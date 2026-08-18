# optim

Optimizers, gradient clipping and learning-rate schedules.

```lua
local optim = nano.optim
```

## Optimizers

| Member | Description |
| --- | --- |
| `optim.SGD(params, lr?, momentum?)` | Stochastic gradient descent. Defaults `0.01`, `0`. |
| `optim.Adam(params, lr?, beta1?, beta2?, epsilon?)` | Adam. Defaults `0.001`, `0.9`, `0.999`, `1e-8`. The usual choice. |
| `optim.AdamW(params, lr?, weightDecay?, beta1?, beta2?, epsilon?)` | Adam with decoupled weight decay. Defaults `0.001`, `0.01`. |
| `optim.RMSprop(params, lr?, alpha?, epsilon?)` | RMSprop. Defaults `0.01`, `0.99`, `1e-8`. |
| `optim.Optimizer` | The shared base class, exposed for custom optimizers. |

### Properties

| Member | Description |
| --- | --- |
| `opt.lr: number` | Learning rate. Writable at any time. |
| `opt.stepCount: number` | Number of `step` calls so far. |
| `opt.parameters: {Tensor}` | The parameter list the optimizer was given. |

### Methods

| Member | Description |
| --- | --- |
| `opt:zeroGrad()` → `()` | Ready every gradient for the next backward pass. Call before every backward. |
| `opt:hardZeroGrad()` → `()` | Actually zero-fill every gradient buffer. |
| `opt:step()` → `()` | Apply one update. |
| `opt:clipGradNorm(maxNorm)` → `number` | Clip by global gradient norm. Returns the **pre-clip** norm. |
| `optim.Optimizer.init(self, parameters, lr)` → `self` | Initialise a custom optimizer. |

## zeroGrad does not zero anything

It marks each gradient buffer stale, and the next accumulation overwrites instead of
adding — same result, without an O(params) pass on every step. For a 4,866 parameter
network that is 4,866 writes replaced by 6 flag writes.

The one visible difference: between `zeroGrad()` and `backward()`, a parameter's `.grad`
still holds the *previous* step's values rather than zeros. Nothing in Nano reads
gradients in that window — `clipGradNorm` and `step` both run after `backward` — but if
your own code inspects them there, use `hardZeroGrad()`.

## Adam vs AdamW

Classic L2-in-the-gradient decay gets divided by `sqrt(v)` along with everything else,
making regularisation strength inversely proportional to gradient magnitude. AdamW applies
decay straight to the weight, which is what you actually meant.

Use AdamW when you want regularisation. For RL, plain Adam is usually right — the data
distribution shifts constantly and weight decay fights that rather than helping.

## Gradient clipping

Not optional for PPO. One freak advantage estimate takes the policy out, and the next
rollout is garbage — collected by a broken policy, so recovery is not automatic.

```lua
opt:zeroGrad()
loss:backward()
opt:clipGradNorm(0.5)
opt:step()
```

Order matters: clip after `backward` and before `step`. The return value is the pre-clip
norm, which is worth logging — a norm that spikes by orders of magnitude means something
upstream is wrong.

## Schedules

| Member | Description |
| --- | --- |
| `optim.schedule(opt, fn)` → `Scheduler` | Drive `opt.lr` from a function of step count. `fn` returns a multiplier on the base lr. |
| `optim.Scheduler` | The scheduler class. |
| `sched:step()` → `number` | Advance one step and return the new lr. Call after `opt:step()`. |
| `sched:reset()` → `()` | Reset the count and restore the base lr. |
| `sched.baseLR: number` | The lr captured when the schedule was created. |
| `sched.count: number` | Steps taken. |

| Member | Description |
| --- | --- |
| `optim.linearDecay(total, finalFraction?)` → `function` | Linear decay to `finalFraction` over `total` steps. The PPO default. |
| `optim.cosine(total, finalFraction?)` → `function` | Cosine decay to `finalFraction` over `total` steps. |
| `optim.stepDecay(every, factor)` → `function` | Multiply by `factor` every `every` steps. |
| `optim.warmup(steps, after?)` → `function` | Ramp from 0 to full lr over `steps`, then hand off to `after`. |

```lua
local sched = optim.schedule(opt, optim.linearDecay(totalSteps))

-- in the loop, after opt:step()
sched:step()
```

Warmup exists because Adam's `v` is unreliable for the first few dozen steps — it has seen
too few gradients to know each parameter's scale, so early updates can be enormous.
Essential for anything deep; usually unnecessary for a three-layer MLP.
