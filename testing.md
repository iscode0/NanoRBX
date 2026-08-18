# Tensors and autograd

What a tensor is in Nano, how the computation graph is built, and what `backward()`
actually does. This explains the design decisions behind
[`Tensor`](../reference/tensor.md) — read it once and the reference page reads as obvious.

## Shapes are `{batch, features}`

Every layer expects a batch dimension. A single sample is a batch of one:

```lua
local one   = T.new({{0.3, 0.7}})   -- {1, 2}  correct
local wrong = T.new({0.3, 0.7})     -- {2}     a vector, not a sample
```

Double braces. `matmul` requires two 2D tensors, so `{2}` fails outright — which is
merciful, because the shapes that *silently* broadcast wrong are much harder to find.

## Flat storage with strides

A tensor is not a table of tables. A `{2,3}` tensor is one `buffer` of six f64 values
plus a description of how to read them:

```
data   = buffer of 6 f64 (48 bytes)
shape  = {2, 3}
stride = {3, 1}
element (i, j) = offset + (i-1)*stride[1] + (j-1)*stride[2]   -- in ELEMENTS
```

One multiply-add per access instead of a pointer chase per dimension, one allocation per
operation instead of one per row. Offsets are counted in elements; every access
multiplies by 8 to get bytes, and hot loops carry byte offsets directly so that multiply
never appears inside one.

**`t.data` is a buffer, not a table.** `t.data[i]` is not an error — it returns `nil`.
The accessors are:

| you want | use |
|---|---|
| One element | `t:at(i)` / `t:setAt(i, v)` |
| One element by dimension | `t:get(i, j)` |
| A Lua array | `t:toTable()` |
| A contiguous buffer | `t:toFlat()` |
| The single value of a scalar | `t:item()` |

`toFlat()` returns a **buffer**. Indexing it with a number raises "attempt to index buffer
with number", and passing it where an array is expected fails silently in any loop using
`#`. `toTable()` is the interop boundary, and it always copies.

### Why f64, and why buffers

Measured on this workload, table array reads and `buffer.readf64` are roughly equal for
pure sequential access — 1.02x. But the inner operation of a matmul is
`out[i] += a * b[j]`, a read-modify-write, which measures 1.53x. A full matmul measures
1.82x, and matmul is essentially 100% of a training step.

f32 was not an option. It puts 0.14% error on a central difference against a 2e-4
gradcheck tolerance, so every gradient check would have failed. f64 proved *bit-identical*
to the old table arithmetic — the agreement check reported a worst element difference of
exactly zero — which is what made a 2,200-line storage rewrite verifiable rather than
hopeful. Memory halved as a side effect: 8 bytes per element against a 16-byte TValue.

### Every op materializes

No operation in Nano returns a view. `transpose`, `reshape`, `select` and `narrow` all
copy into a fresh contiguous tensor, so `_contig` is always true.

The stride machinery buys cheap indexing and correct broadcasting, not zero-copy views.
That is a deliberate trade: materializing keeps every backward pass a simple contiguous
loop, and the fast paths that check `_contig` always hit.

## Broadcasting

Operations between different shapes follow NumPy rules: dimensions are matched from the
right, and a size-1 or missing dimension is stretched.

```lua
local x = T.randn({4, 2})      -- a batch of 4 samples
local b = T.randn({2})         -- one bias per feature
local y = T.add(x, b)          -- {4, 2}: b is added to every row
```

The important half is that **the gradient reduces correctly back down**. The gradient
arriving at `y` is `{4,2}`; the gradient stored into `b` is `{2}`, summed over the batch.
That is what makes batched training work at all — without it, a bias would need one copy
per sample.

Nano picks one of four dispatch paths once per operation: scalar → identical contiguous
shapes → `{n,m}` with `{m}` (the bias case) → general strided broadcast. The common cases
never pay for the general one.

## The computation graph

Every tensor produced by an operation records three things:

| field | holds |
|---|---|
| `_prev` | The input tensors it was computed from |
| `_backward` | A closure that pushes gradient into those inputs |
| `grad` | The accumulated gradient, once `backward()` has run |

So this code:

```lua
local a = T.new({{2}}, nil, true)
local b = T.new({{3}}, nil, true)
local c = T.mul(a, b)
local d = T.add(c, a)
```

builds a small graph where `d._prev` is `{c, a}` and `c._prev` is `{a, b}`. Calling
`d:backward()` walks it in reverse, calling each `_backward` closure, and every leaf ends
up with the right gradient — including `a`, which contributed through two different
paths.

That last point is the whole reason for the next section.

## Gradients accumulate

`backward()` **adds into** `grad`. It never overwrites.

This is not an implementation quirk. It is what makes a graph like the one above correct:
`a` feeds both `c` and `d`, so its true gradient is the sum of the two contributions. If
the second write replaced the first, every model with a skip connection, a shared trunk,
or a tensor used twice would silently compute the wrong gradient.

The cost is that stale gradients pile up across training steps, which is why every loop
starts with:

```lua
opt:zeroGrad()
```

Forgetting produces no error. Just loss behaviour you cannot explain.

### zeroGrad does not zero anything

`Optimizer:zeroGrad` marks each gradient buffer **stale**, and the next accumulation
overwrites instead of adding. Same result, without an O(params) pass every step — for a
4,866 parameter network, 4,866 writes replaced by 6 flag writes.

One visible consequence: between `zeroGrad()` and `backward()`, a parameter's `.grad`
still holds the *previous* step's values, not zeros. Nothing inside Nano reads gradients
in that window. If your own code does, use `opt:hardZeroGrad()`.

## backward() needs a scalar

```lua
loss:backward()          -- fine, loss is {1}
prediction:backward()    -- error, prediction is {32, 4}
```

You can only differentiate a single number. If you have a tensor, reduce it first with
`T.sum` or `T.mean` — which is exactly what every loss function does.

The traversal is iterative, not recursive. A deep RL rollout can build a graph thousands
of nodes deep, and a recursive implementation would blow the Luau stack.

## Leaves vs results

A tensor is a **leaf** if you constructed it, and a **result** if an operation produced
it. `t._isLeaf` tells you which.

In-place operations — `addInPlace`, `mulInPlace`, `fillInPlace`, `copyFrom` and friends —
refuse to run on a result. This is a guard, not a limitation: mutating a tensor the graph
still references would change the values that a `_backward` closure reads later, and the
gradients would be quietly wrong. Refusing to run is the only safe option, because there
is no correct backward rule for "somebody changed this after the fact".

Related, and not guarded because it cannot be: **inputs are not copied.** Backward
closures read source arrays directly, which is the same tradeoff PyTorch makes. If you
mutate an input tensor between `forward` and `backward`, you get wrong gradients with no
error.

## Turning the graph off

Graph construction is gated globally by `Config.gradEnabled`. When it is off, an operation
allocates its output and does nothing else — no `_prev`, no closure, no gradient buffers.

```lua
nano.noGrad(function()
    return model:forward(obs)
end)
```

Use it for every forward pass you will not differentiate: evaluation, deployed inference,
rollout collection, evolutionary fitness evaluation. It is a large speedup, not a micro
one — measured at 3.29x on a batch-32 forward.

`noGrad` restores the *previous* state rather than setting `true`, so nested calls
compose, and it restores on error too — a throw inside your function cannot leave
gradients globally disabled for the rest of the session.

## Verifying a backward pass

A wrong backward pass never errors. It trains slightly wrong, and you spend a week
blaming hyperparameters. So Nano ships a checker:

```lua
local ok, worstError = T.gradcheck(function(t)
    return T.sum(myCustomOp(t))
end, T.randn({3, 4}))
assert(ok, "backward is wrong, worst relative error " .. worstError)
```

It nudges each input element up and down, measures how the output actually changed, and
compares that against what your backward pass claims. Disagreement means a bug.

Two requirements. `fn` must be **deterministic** — it is called three times per element,
so dropout or unseeded sampling makes the comparison meaningless. And it must return a
**scalar**.

Run it on anything you write yourself. See [Extending Nano](../guides/extending.md).

## Next

- [`Tensor` reference](../reference/tensor.md) — every operation
- [Extending Nano](../guides/extending.md) — custom ops, gradchecked
- [Performance](../performance.md) — what the storage design bought
- [Gotchas](../gotchas.md) — the autograd mistakes that do not error
