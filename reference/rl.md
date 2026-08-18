# rl

The pieces every reinforcement learning algorithm rebuilds — not the algorithms
themselves. For those see [`algorithms`](algorithms.md). For what any of this means, see
the [RL primer](../concepts/rl-primer.md).

```lua
local rl = nano.rl
```

## RunningNorm

Streaming observation normalisation via Welford's algorithm.

| Member | Description |
| --- | --- |
| `rl.RunningNorm.new(features, clip?)` → `RunningNorm` | `clip` bounds the normalised output, default 10. |
| `norm.frozen: boolean` | When `true`, `step` normalises without updating the statistics. Set before evaluation. |
| `norm:observe(x)` → `()` | Update the running mean and variance. |
| `norm:normalize(x)` → `{number}` | Normalise without observing. |
| `norm:step(x)` → `{number}` | Observe then normalise. The usual call. |
| `norm:state()` → `table` | Serialisable statistics. |
| `norm:load(state)` → `()` | Restore statistics from `state`. |

Normalising inputs is routinely the difference between learning and not: a feature in
`[0, 80]` dominates every dot product against one in `[-1, 1]`, and the weights take most
of training to compensate.

Save the state with the model. A deployed policy that normalises with different statistics
than it trained with is a different policy.

## ReplayBuffer

Uniform-sampling circular buffer for off-policy algorithms.

| Member | Description |
| --- | --- |
| `rl.ReplayBuffer.new(capacity, obsSize, actionSize)` → `ReplayBuffer` | Fixed-capacity circular storage. |
| `buffer.count: number` | Transitions currently stored. |
| `buffer:add(obs, action, reward, nextObs, terminated)` → `()` | Store one transition. |
| `buffer:ready(minimum)` → `boolean` | Whether at least `minimum` transitions are stored. |
| `buffer:sample(batch)` → `table` | Uniform sample. Fields: `.obs`, `.action`, `.nextObs` (Tensors), `.reward`, `.done`, `.obsRows`, `.actionRows` (arrays). |

The `done` argument **must mean termination, not truncation**. A time limit is not the end
of the world — the next state still has value. Storing a truncation as `done` teaches the
critic that value is zero at every episode boundary.

## RolloutBuffer

On-policy storage for PPO-style algorithms.

| Member | Description |
| --- | --- |
| `rl.RolloutBuffer.new(horizon)` → `RolloutBuffer` | Storage for one rollout. |
| `rollout.n: number` | Transitions currently stored. |
| `rollout.obs` · `.action` · `.logp` · `.reward` · `.value` · `.done` · `.bootstrap` | The parallel arrays, readable directly. |
| `rollout:add(obs, action, logp, reward, value, done, bootstrap?)` → `()` | Store one transition. |
| `rollout:clear()` → `()` | Reset to empty. |
| `rollout:full()` → `boolean` | Whether the horizon has been reached. |

## PrioritizedReplay

Replay that draws in proportion to TD error, so the batch is made of what the network is
still getting wrong. Priorities live in a sum tree, so sampling and updating are O(log n).

| Member | Description |
| --- | --- |
| `rl.PrioritizedReplay.new(capacity, obsSize, actionSize, alpha?, beta?)` → `PrioritizedReplay` | `alpha` defaults to 0.6, `beta` to 0.4. |
| `per:add(obs, action, reward, nextObs, terminated)` → `()` | Store one transition at maximum priority. |
| `per:ready(minimum)` → `boolean` | Whether at least `minimum` transitions are stored. |
| `per:sample(batch)` → `table` | Same fields as `ReplayBuffer:sample`, plus `.indices` and `.weights`. |
| `per:updatePriorities(indices, errors)` → `()` | Set new priorities from fresh TD errors. |
| `per:setProgress(fraction)` → `()` | Anneal `beta` toward 1 as training progresses. |

```lua
local per = rl.PrioritizedReplay.new(50000, obsSize, actionSize)
per:add(obs, action, reward, nextObs, terminated)

local batch = per:sample(64)
-- ... compute tdErrors, weighting the loss by batch.weights ...
per:updatePriorities(batch.indices, tdErrors)
per:setProgress(step / totalSteps)
```

Two corrections make it correct rather than merely faster. `alpha` controls how greedily
priority is followed — fully greedy overfits the few highest-error transitions. `beta` is
the importance-sampling weight that cancels the bias non-uniform sampling introduces; it
anneals to 1 because the bias matters least early, when the value function is wrong
anyway. Weights are normalised by the largest, so they only ever scale gradients **down**.

Sampling from an empty tree errors rather than returning garbage.

## Advantages and returns

| Member | Description |
| --- | --- |
| `rl.gae(rewards, values, dones, bootstraps, lastValue, gamma, lambda)` → `({number}, {number})` | Generalized advantage estimation. Returns advantages and returns. |
| `rl.normalizeAdvantages(advantages)` → `()` | Zero mean, unit variance, in place. |
| `rl.discountedReturns(rewards, dones, gamma)` → `{number}` | Plain discounted return per step. |
| `rl.explainedVariance(values, returns)` → `number` | How much of the return variance the critic accounts for. |

`dones` cuts the trace; `bootstraps` supplies `γ·V(s')` at a time-limit boundary. **Both
are needed.** A cut without a bootstrap is the most common silent bug in PPO
implementations, and it looks exactly like the algorithm being broken.

`explainedVariance` near 0 or negative means the critic is useless, so every advantage is
noise and the policy gradient cannot work. It is the fastest single number for telling a
broken critic from a broken policy.

Normalise advantages **per batch, not per minibatch** — otherwise the same transition gets
a different advantage depending on which minibatch it lands in.

## Distributions

| Member | Description |
| --- | --- |
| `rl.Normal.new(mean, logStd)` → `Normal` | Diagonal Gaussian over continuous actions. |
| `dist:logProb(actions)` → `Tensor` | Batched, differentiable log-probability. |
| `dist:entropy()` → `Tensor` | Differentiable entropy. |
| `dist:sample()` → `{{number}}` | Plain Lua sample, for rollouts. |
| `rl.Categorical.new(logits)` → `Categorical` | Distribution over discrete actions. |
| `categorical:logProb(actions)` → `Tensor` | Log-probability of the given 1-based indices. |
| `categorical:entropy()` → `Tensor` | Differentiable entropy. |
| `categorical:sample()` → `{number}` | One action index per row. |
| `rl.SquashedNormal.new(mean, logStd)` → `SquashedNormal` | Tanh-squashed Gaussian. The SAC policy. |
| `squashed:rsample(rows, dims)` → `(Tensor, Tensor)` | Reparameterised sample and its log-probability. |

`SquashedNormal` sampling is reparameterised with noise held constant, so the action is
differentiable in mean and std and the gradient of `Q(s,a)` reaches the actor. The
`log(1 - tanh(u)²)` correction is not optional — without it a learned temperature tunes
against an entropy it cannot measure.

**Never clamp an action before scoring it.** Scoring a clamped action with an unclamped
Gaussian density makes every stored log-prob wrong. Bound actions in the environment
instead.

## Noise and target networks

| Member | Description |
| --- | --- |
| `rl.gaussianNoise()` → `number` | One standard normal, seeded via `nano.seed`. |
| `rl.OUNoise.new(dims, theta?, sigma?)` → `OUNoise` | Ornstein-Uhlenbeck, temporally correlated. |
| `ou:sample()` → `{number}` | Next correlated noise vector. |
| `ou:reset()` → `()` | Return the process to zero. |
| `rl.polyak(targetParams, liveParams, tau)` → `()` | Soft update: `target = (1-tau)·target + tau·live`. |
| `rl.hardCopy(targetParams, liveParams)` → `()` | Copy live parameters straight into the target. |

OU noise drifts rather than jittering. White noise averages out over consecutive steps, so
an agent with heading inertia barely moves under it — worth trying whenever your
environment blends headings between steps.
