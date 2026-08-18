# Testing

Five scripts, **341 assertions, all passing**.

| script | assertions | covers |
| --- | --- | --- |
| `TestTensor` | 142 | The Tensor core alone, bypassing `init.lua` |
| `TestGradients` | 113 | Every differentiable op, plus fused-vs-composed equivalence |
| `TestAlgorithms` | 23 | Agent contracts, and does-it-actually-learn |
| `TestAdditions` | 47 | Seeding, Trainer, diagnose, PER, compile |
| `TestRecurrent` | 16 | RecurrentPPO on a provable memory task |

All are Scripts in `ServerScriptService`.

## When to run what

**`TestTensor` first**, on a fresh install. It requires `Nano.Tensor` directly rather than
`init.lua`, so a failure is a Tensor bug and not fallout from somewhere else in the load
chain.

**`TestGradients` after any change to a backward pass.** This is the important one. A wrong
backward pass never errors — it trains slightly wrong — so this is the only thing standing
between you and a week of blaming hyperparameters.

The other three after changes to their respective areas.

## TestTensor

Covers construction, every arithmetic dispatch path, reductions, views, in-place guards,
set-to-none semantics, and a matmul sweep across every unroll remainder width against a
naive reference.

The matmul sweep matters because the inner loop is unrolled 4x. Widths that are not
multiples of 4 take the remainder path, and an off-by-one there would be invisible at width
64 and wrong at width 65.

## TestGradients

Verifies every differentiable op against a central finite difference — **110 checks** —
covering core `Tensor`, all of `functional` including the fused kernels, `nn` layers and
losses, all three RNN cells, and the RL distributions.

### Equivalence tests

It also runs three **equivalence tests**, which catch something gradcheck structurally
cannot.

A fused kernel can be internally consistent — its backward matching its own forward
perfectly — while computing the wrong function entirely. Gradcheck would pass. So each
fused op is additionally compared against the composed chain it replaced.

### Two specific guards

**`nn.Embedding` with a repeated index.** The backward must `+=`, not `=`. With `=`, a
batch containing the same index twice trains on only one of them, and the bug is invisible
in any test that happens to use distinct indices.

**A diamond graph** where one tensor feeds two branches that later merge. This is the case
that requires gradients to accumulate rather than overwrite, and it is the reason
`backward()` adds into `grad`.

### Stricter than Tensor.gradcheck

The checker in `TestGradients` restores `requiresGrad` and `grad` on inputs, and asserts
each gradient array has exactly `numel` entries.

That second check matters because `Tensor.accumulate` loops to `#values`, not `#grad`. A
short gradient array from a hand-written backward is silently *partially* applied — no
error, no warning, just a model that trains slightly wrong. This is now caught.

### Kinks

Ops with a non-differentiable point — `relu`, `abs`, `clamp`, `elu`, `hardSigmoid`, `L1`,
`BCE`, `nano.min` — get inputs offset away from the discontinuity. Central differences are
simply wrong *at* a kink, so testing there would report failures that are measurement
artifacts rather than bugs.

A failure on one of these with offset inputs is a real bug.

## TestAlgorithms

Checks the agent contracts — `act` before `observe`, `terminated` and `truncated` handled
separately, `ready`/`update` behaviour — and then that each agent actually learns on a
small task.

The second half exists because contract tests pass on an agent that does everything
correctly except improve. A learning check is the only thing that catches an update path
that runs, reports statistics, and moves the policy nowhere.

## TestAdditions

Seeding reproducibility, the `Trainer`, `diagnose`, prioritized replay, and `nn.compile`.

The seeding tests assert that two runs with the same seed produce identical weights. This
is the test that would have caught the drifting-RNG bug that once made `keepBest` track the
luckiest *evaluation* rather than the best *policy*.

## TestRecurrent

RecurrentPPO on a task where a cue appears on step 1 and never again, with reward only on
the final step.

The task is constructed so a memoryless policy is **provably** bounded. It sees an identical
final observation every episode, so it can only emit a constant, which bounds it
analytically at -0.2133. Trained feedforward PPO scores -0.2328. RecurrentPPO scores
-0.0058, closing 97% of the gap to perfect recall.

The analytic bound is what makes this a test rather than a demo. Without it, a score of
-0.0058 is just a number.

An earlier version of this test was wrong in an instructive way: it hid the target but left
the agent's position visible, so a memoryless policy encoded the target into its own
position on step 1 and read it back forever after — using the world as memory. It scored
well and remembered nothing. **Hiding a variable does not make a memory task**; it must not
be recoverable from anything the agent can observe or influence.

## Adding tests

When you write a custom op, add it to `TestGradients.server.lua`:

```lua
check("myOp", function(t)
    return Tensor.sum(myOp(t))
end, Tensor.randn({3, 4}))
```

If it has a kink, offset the inputs away from it. If it replaces a composed chain, add an
equivalence test against that chain as well — gradcheck alone cannot tell you the fused
version computes the same function.

See [Extending Nano](guides/extending.md).
