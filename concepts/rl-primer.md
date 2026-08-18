# Reinforcement learning primer

Supervised learning needs correct answers. Reinforcement learning is what you use when
you do not have them — you know what a *good outcome* looks like, but not what the right
action was at each step.

Training an NPC to navigate, fight, gather or flee is RL. This page explains the pieces
and, more importantly, the ways RL goes wrong that supervised learning does not.

## The setup

An **agent** observes the world, takes an **action**, and receives a **reward**. That
repeats until the episode ends.

```lua
local action = agent:act(obs)
local nextObs, reward, terminated, truncated = env:step(action)
agent:observe(reward, terminated, truncated, nextObs)
```

| term | meaning |
|---|---|
| observation | What the agent can see, as `{number}` |
| action | What it does — a vector (continuous) or an index (discrete) |
| reward | A number scoring what just happened |
| episode | One run from reset to termination |
| return | The total reward over an episode |
| policy | The network mapping observation → action |
| value | The expected future return from a state |

The goal is a policy that maximises return. Nobody tells the agent which action was
correct — it has to work that out from rewards that arrive later, often much later.

## Why this is harder than supervised learning

Four things break that do not break when you have labels.

**The data depends on the policy.** A supervised dataset sits still. Here, the agent
generates its own training data by acting, so improving the policy changes the
distribution of what it sees next. A bad early policy can wander into a region of the
world it then never escapes.

**Rewards are delayed.** The action that lost the fight happened thirty steps before the
death. Assigning credit backwards through time is the central problem, and it is what
`gamma` and GAE exist to handle.

**Exploration is your problem.** A policy that only ever does what currently looks best
never discovers anything better. Every algorithm here handles this differently: PPO and
SAC add noise to a sampled action, DQN uses epsilon-greedy.

**There is no loss curve to trust.** In supervised learning, falling loss means progress.
In RL the loss is computed against a moving target and routinely goes *up* while the
policy improves. You have to look at episode return instead.

## Rewards

The reward function is the single highest-leverage thing you write. The agent will
optimise exactly what you wrote, not what you meant.

Practical rules:

**Dense beats sparse.** A reward only at the goal means the agent must stumble onto it
by chance before it learns anything. Reward progress toward the goal too.

**Keep the scale sane.** Roughly `[-1, 1]` per step, or `[-10, 10]` per episode. Rewards
in the thousands produce enormous value targets and unstable critics.

**Penalise time, lightly.** A small negative per step (`-0.01`) makes the agent prefer
finishing sooner, which removes a lot of dithering.

**Expect to be exploited.** Reward proximity to an enemy and the agent will learn to hover
just outside attack range forever. If the behaviour is strange, the reward function
usually explains it before the algorithm does.

## terminated vs truncated

Nano requires these separately and refuses to guess.

```lua
env:step(action) -> obs, reward, terminated, truncated
```

**`terminated`** means the state genuinely has no future. The agent died, the goal was
reached, the episode is over on its own terms. Future value from here is zero.

**`truncated`** means you stopped watching. A step limit, a timeout, a training-loop
boundary. The world would have continued and the next state still has value.

Collapsing them into one `done` teaches the critic that value is zero at every episode
boundary, including the artificial ones. The critic becomes systematically pessimistic
near time limits, every advantage estimate downstream is wrong, and it looks exactly like
"the algorithm does not work on my task."

This is the most common silent bug in RL implementations. It is why the arguments are
required rather than optional, and why a truncated step without `nextObs` errors instead
of bootstrapping zero.

## Discounting and gamma

A reward now is worth more than a reward later. `gamma` sets how much more — the return
is `r₁ + γ·r₂ + γ²·r₃ + ...`.

`gamma = 0.99` is the default and means a reward 100 steps away counts for about a third
of an immediate one. Lower it (`0.95`) for short episodes where distant rewards are
irrelevant; raise it (`0.999`) for long-horizon tasks. Values at or above 1 do not
converge.

## The critic and advantages

Most algorithms train two networks. The **actor** is the policy. The **critic** estimates
the value of a state — expected future return.

The critic exists to reduce noise. Knowing an action returned +10 tells you little;
knowing it returned +10 from a state usually worth +2 tells you a lot. That difference is
the **advantage**, and it is what the policy gradient actually uses.

[`rl.gae`](../reference/rl.md) computes advantages with generalized advantage estimation,
trading bias against variance via `lambda` (default `0.95`).

`rl.explainedVariance` is the fastest single number for diagnosing a run. It measures how
much of the return variance the critic accounts for. Near zero or negative means the
critic is useless, so every advantage is noise and the policy gradient cannot work — that
distinguishes a broken critic from a broken policy in one number.

## Observation normalisation

A feature ranging over `[0, 80]` dominates every dot product against one in `[-1, 1]`, and
the weights take most of training just to compensate.

[`rl.RunningNorm`](../reference/rl.md) tracks a streaming mean and variance and normalises
as it goes. Every agent in [`algorithms`](../reference/algorithms.md) uses it by default
via `normalizeObs = true`. Freeze it before evaluation (`norm.frozen = true`) so
evaluation does not shift the statistics.

Normalising inputs is routinely the difference between learning and not learning.

## Picking an algorithm

| your actions are | try | why |
|---|---|---|
| Continuous (movement, aim, throttle) | `PPO` | Stable, forgiving, the sensible default |
| Continuous, and samples are expensive | `SAC` | Off-policy, reuses a replay buffer, far more sample-efficient |
| Discrete (a fixed list of moves) | `DQN` | Simple, well understood, Double DQN by default |
| Discrete or continuous, but the observation is incomplete | `RecurrentPPO` | Adds memory |

Start with PPO unless you have a specific reason not to.

**Use RecurrentPPO when the observation does not contain everything the agent needs.** A
target that leaves view, occlusion, a signal shown once at the start of the episode. If
the current observation is sufficient, memory is expensive and pointless.

One warning on testing memory: **hiding a variable does not make a memory task.** It must
not be *recoverable* from anything the agent can observe or influence. An early version of
Nano's recurrent test hid the target but left the agent's position visible — so a
memoryless policy learned to encode the target into its own position on step 1 and read it
back forever after, using the world as memory. It scored well and remembered nothing.

## What a training loop looks like

```lua
local agent = nano.algorithms.PPO.new({
    obsSize = 8, actionDim = 2, hidden = 64,
})

local trainer = nano.train.Trainer.new(agent, env, {
    steps = 100000,
    evalEvery = 5000,
    evalEnv = makeEnv(),      -- a SECOND instance
    keepBest = true,
})

local summary = trainer:run()
trainer:restoreBest()
```

[`Trainer`](../reference/train.md) owns the rules that are easy to get wrong: yield on a
time budget rather than a step count, update at the right moment for the algorithm, and
keep the best weights rather than the last.

Two things it needs from you. Give your environment a `seed(n)` method, or evaluation is
not reproducible and `keepBest` tracks the luckiest evaluation rather than the best
policy. And pass a separate `evalEnv` — evaluating on the training environment resets it
mid-rollout, so the loop resumes with an observation that no longer describes the world.

Full walkthrough in [Training an NPC with RL](../guides/rl-agent.md).

## Reading the numbers

`agent:update()` returns statistics. What to watch:

| stat | healthy | what it means |
|---|---|---|
| `meanReward` | rising | The only thing that ultimately matters |
| `explainedVariance` | > 0.3 and rising | Critic quality. Near 0 means it is useless |
| `clipFraction` | 0.05 – 0.3 | Fraction of updates hitting the PPO clip |
| `approxKL` | < 0.02 | How far the policy moved. Spikes mean instability |
| `entropy` | falling slowly | Exploration. A cliff means premature convergence |

`clipFraction` near zero means your learning rate is so small nothing is happening. Above
0.5 means updates are far too aggressive.

`entropy` collapsing early is the classic failure: the policy commits to one action before
it has explored, and never recovers. Raise `entropyCoef`.

## Debugging checklist

When it does not learn, in this order:

1. **Is the reward function what you think?** Print it. Log episode returns from a random
   policy for a baseline.
2. **Are `terminated` and `truncated` correct?** Run `train.checkEnv(env, sampleAction)`.
3. **Is `explainedVariance` moving?** If not, the critic is the problem, not the policy.
4. **Are observations normalised and finite?** A `nan` in one observation poisons
   everything downstream.
5. **Did any update actually run?** `summary.updates` — a rollout agent whose horizon
   exceeds the total step count collects forever and never learns. The Trainer warns
   about this loudly.
6. **Is the task actually solvable from the observation?** If a human could not do it from
   what the agent sees, you need `RecurrentPPO` or a better observation.

## Next

- [Training an NPC with RL](../guides/rl-agent.md) — full walkthrough
- [`algorithms` reference](../reference/algorithms.md) — every agent and option
- [`rl` reference](../reference/rl.md) — buffers, GAE, distributions
- [`train` reference](../reference/train.md) — the environment protocol and Trainer
