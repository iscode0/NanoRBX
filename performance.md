# Performance

Every number here was measured, not estimated. Two of them contradicted the prediction that
motivated the change, which is the main reason this page exists in this form.

## What makes it fast

**Batch everything.** One `{32, 8}` forward is roughly a thirtieth the cost of thirty-two
`{1, 8}` passes, because per-operation overhead dominates at these sizes. A per-sample loop
in your training step *is* your bottleneck.

**Use `noGrad` for anything you will not differentiate.** Rollout collection, evaluation,
deployed inference, every evolutionary fitness evaluation. Tracking off means an operation
allocates its output and nothing else.

**Keep tensors contiguous.** Every fast path checks `_contig`.

**Direct arithmetic beats closures.** Luau cannot fastcall or inline an indirect call, so a
closure passed as a parameter costs a full CALL per element. Hot arithmetic and the common
activations use opcode-selected loops where the branch happens once per operation rather
than once per element.

**`--!native` on the numeric modules.** Verify with the Script Profiler that `Adam.step` and
the loss forwards show `<native>`; if not, a type misprediction is de-optimizing them.
Watch the budget with `debug.dumpcodesize()` in Server view. Wrong annotations are worse
than none — under native codegen they dictate the generated machine code.

Native codegen is server-side only. Client scripts run the same code interpreted.

## Yielding

Yield on a **time budget**, never a step count:

```lua
local budget = os.clock() + 0.008
if os.clock() > budget then
    task.wait()
    budget = os.clock() + 0.008
end
```

A step triggering a gradient update costs ~100x a plain environment step, so a fixed step
count either stalls the server during update-heavy stretches or wastes most of the frame
during cheap ones. Measuring elapsed time adapts automatically.

Use a coarser budget (~30 ms) during update phases than during rollout — nothing is
watching the world while it learns. `train.Trainer` handles this for you.

## Measured results

Post-migration, on an 8-64-64-2 network at batch 32:

| | before | after | gain |
| --- | --- | --- | --- |
| Train step | 2.32 ms | **0.83 ms** | **2.80x** |
| Forward, no grad | 0.79 ms | **0.24 ms** | **3.29x** |
| Updates/sec | 431 | **1208** | 2.80x |

Per-change, measured in isolation:

| change | gain |
| --- | --- |
| `transpose` {64,64} forward (2D path) | **9.9x** |
| Fused LayerNorm, forward | **9.1x** |
| Fused LayerNorm, fwd+bwd | **6.4x** |
| `transpose` {64,64} fwd+bwd | **5.5x** |
| `nn.compile` rewrite (batch-1, hidden 64) | **2.45x** |
| Huber loss fwd+bwd (fused) | **5.8x** |
| MSE loss fwd+bwd (fused) | **3.2x** |
| `zeroGrad` set-to-none | 2.7x (both sides near timer resolution) |
| Linear fwd+bwd (fused) | 1.06x |
| Matmul inner loop (4x unroll) | 1.02x |
| dW loop order (i-outer) | 1.01x |

## The two that did not go as predicted

**The buffer migration overdelivered.** `BenchBuffer` measured matmul alone with
pre-allocated buffers and predicted 1.5-1.8x end to end; the actual was 2.80x. The isolated
benchmark could not see the allocation and locality gains across the rest of the graph.

**The `nn.compile` rewrite was worth 2.45x, and the reason was embarrassing.** `compile`
was supposed to beat a `noGrad` forward at batch 1. It lost to it, by 1.7x. The cause was
not subtle: `F.linear` unrolls its inner loop 4x and skips zero multiplicands, and
`compile` did neither — so the "optimized" inference path ran a naive matmul while the
ordinary path ran the tuned one. Nothing caught it because nothing compared the two.

Porting the unroll, and hoisting the per-step table lookups and buffer ping-pong into
compile-time parallel arrays, took it from 10.5 us to **4.29 us** on an 8-64-64-2 batch-1
forward — 2.45x against its old self, and 1.50x against `noGrad`, which is what it was
always supposed to be. The lesson is not about unrolling; it is that a fast path nobody
benchmarks against the slow path is an assumption, not an optimisation.

**The dW loop-order fix was a nothing.** Switching from p-outer to i-outer makes reads of
`x` sequential instead of strided by `k`, which sounded like an 8x reduction in cache lines
touched. It measured 1.01x.

The strided read happens once per (i, p) and is amortised over `m` inner iterations, so it
is about 1.5% of the work — the cache-line analysis was applied to the wrong loop level,
and a 64 KB working set mostly sits in L2 regardless. The swap is kept because it is never
slower and the stride would matter at large `k`, but it is not a win.

The pattern is consistent and worth internalising: **fusion pays where node count dominates
arithmetic.** Huber composed was 7 graph nodes over 512 elements, so removing 6 removed 6/7
of the cost. A `Linear` layer is one matmul where the fused bias is 1.6% of the work, so
fusing it saved the node, not the math.

## Fusing LayerNorm

The composed form was nine graph nodes — `mean`, `sub`, `mul`, `mean`, `add`, `sqrt`,
`div`, `mul`, `add` — each allocating an intermediate tensor and a backward closure over
every element. At `{32,64}` that cost **568 us** forward+backward, which was nine tenths of
a whole batch-32 training step through an 8-64-64-2 network. One normalisation layer cost
about as much as training the network it sat in.

| | before | after | gain |
| --- | --- | --- | --- |
| Forward only | 191.6 us | **21.1 us** | 9.1x |
| `nn.LayerNorm` fwd+bwd | 568.0 us | **88.2 us** | 6.4x |

Measured in the same run, the surviving composed chain costs 490.7 us against the fused
kernel's 88.2 us — **5.6x**, with both paths on identical inputs.

This is the Huber result again (7 nodes, 7.11x) and the rule it established holds: fusion
pays where node count dominates arithmetic. The kernel keeps `xhat` and a per-row inverse
standard deviation from the forward, costing `n + rows` of scratch and saving the backward
two full reduction passes.

**One checksum moved, by exactly one ulp.** The fused forward differs from the composed
chain in the last bit: it computes `1/sqrt(var + eps)` once per row and multiplies, where
the composition divides per element. Multiply-by-reciprocal and divide disagree on about
27% of inputs by 1 ulp. That is a real numerical difference, correctly flagged by the
benchmark checksums, and the right call is to accept it — the equivalence tests agree to
1e-9 and the division count per row dropped from H to 1.

## Transposing without index arithmetic

`transpose {64,64}` cost **153 us** to move 4,096 numbers — 38 ns each, and within 4% of a
full `{32,64}@{64,64}` matmul doing 131,072 multiply-adds.

The forward called `unravel` per element, a division and modulo per dimension, then walked
the stride array to build an offset. The backward was worse: it called `table.clone` once
per element, allocating 4,096 throwaway tables on every backward pass.

| | before | after | gain |
| --- | --- | --- | --- |
| `{64,64}` forward | 153.2 us | **15.5 us** | 9.9x |
| `{64,64}` fwd+bwd | 451.0 us | **81.6 us** | 5.5x |
| `{8,8,8}` 3D, general path | 22.9 us | 22.9 us | unchanged |

For two dimensions the source offset advances by a fixed stride down each column, so it is
a plain double loop with no division and no index array. The 3D case is the control: it
still takes the general path and did not move, which is what rules out the 2D win having
been bought at the general path's expense.

## Compiled inference

`nn.compile` against a `noGrad` forward, batch 1, after the rewrite:

| network | forward | compile | gain |
| --- | --- | --- | --- |
| 8-32-32-2 | 3.7 us | **1.46 us** | ~2.5x |
| 8-64-64-2 | 6.4 us | **4.24 us** | ~1.5x |
| 8-128-128-2 | 15.6 us | **11.3 us** | ~1.4x |

Two independent runs agreed to within 4%. The compiled column is the steadier of the two
(4-6% CV against 12-18% for the forward), so the uncertainty in the ratio sits almost
entirely in the denominator — round these rather than quoting them.

The gain **shrinks as the network widens**, and that is the diagnostic, not a
disappointment. What `compile` removes is per-operation overhead — a fixed cost per layer,
independent of width. Once arithmetic dominates, both paths run the same unrolled inner
loop and converge. If the ratio ever grows with width, something is wrong with the ordinary
path.

It also allocates **nothing** per call, measured at 0.00 kb/op across every compiled case,
which is why 100 sequential compiled NPCs (437-444 us) now beat one batched forward over
the same 100 observations (510-537 us) despite making a hundred separate calls.

One incidental result worth keeping: the compiled function and the batched forward returned
**bit-identical** output on the same input — agreement to all 17 digits across two
unrelated code paths. That is a stronger correctness signal than the tolerance check in
`EdgeSuite`, and it came free from the benchmark's checksums.

## Where the time goes now

| | ms | share |
| --- | --- | --- |
| Forward (nograd) | 0.81 | 34% |
| Graph + backward + optimizer | 1.51 | 66% |

Backward is almost exactly 2x forward flops — it computes both `dX` and `dW` — which the
measurement confirms at 1.88x.

**The train step is now matmul, end to end**, at roughly 4.6 ns per multiply-add. That is
about 14 cycles, which is not the arithmetic. It is the read, multiply-add, and write.

Unrolling the inner loop 4x gained 6% at width 64 and 7% at width 128. Identical gain at
both widths means loop overhead is only ~8% of the body; the rest is storage access.
**Loop-level optimisation is exhausted.**

## Why buffers, and why f64

Measured before migrating, not after:

| access pattern | table vs buffer |
| --- | --- |
| Sequential read | 1.02x |
| Sequential write | 1.01x |
| Read-modify-write | 1.53x |
| Full matmul | **1.82x** |

Pure sequential access is a wash. The win appears exactly where Nano lives, since
`out[i] += a * b[j]` is a read-modify-write inside a matmul, and matmul is essentially 100%
of a training step.

**It had to be f64.** float32 puts 0.14% error on a central difference against a 2e-4
gradcheck tolerance, so every one of the 110 gradient checks would have failed. f64 proved
**bit-identical** to table arithmetic — the agreement check reported a worst element
difference of exactly zero — which is what made a 2,200-line rewrite verifiable rather than
hopeful. Memory halved as a side effect: 8 bytes per element against a 16-byte TValue.

A partial migration was tested and rejected: converting at the boundary costs 8.7% overall,
but a 64→2 layer pays 55% overhead to save 45%. It is all-or-nothing.

## Remaining headroom

**Tensor pooling — open.** Every op allocates a fresh table, and a high allocation rate
forces GC assists that interrupt the script. Shapes are fixed in a training loop, so a free
list keyed by element count would recycle them. Needs an explicit arena scope with a hard
contract about escaping values; recycling a still-referenced tensor corrupts silently.

**Static graph replay — open.** Record the op sequence once for a fixed-shape loop and
replay it against pooled buffers. Largest possible win, largest effort. Only worth it if
profiling shows graph construction, not arithmetic, dominating — which it currently does
not.

## Measuring your own changes

`BenchPerf` measures the optimisations end to end; `BenchBuffer` compares storage
strategies in isolation.

Two Roblox-specific caveats. `collectgarbage("collect")` is blocked — only `gcinfo()` and
`"count"` are available, so allocation measurements have no clean baseline. And an isolated
microbenchmark systematically under-predicts allocation and locality effects, as the buffer
migration demonstrated by a wide margin. Measure end to end before believing a number.
