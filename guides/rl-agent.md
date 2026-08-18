# Training an NPC with RL

An environment, an agent, a training run, and how to tell whether it worked. Read the
[RL primer](../concepts/rl-primer.md) first if the vocabulary is unfamiliar.

## The task

An NPC has to reach a target on a flat plane. It observes its own position and velocity
and the direction to the target, and outputs a movement vector. Continuous actions, so
PPO.

## The environment

An environment is any table with three methods. There is no base class to inherit.

```lua
local Env = {}
Env.__index = Env

function Env.new()
    return setmetatable({
        rng = Random.new(),
        maxSteps = 300,
    }, Env)
end

--- Optional but strongly recommended. Without it, evaluation is not reproducible.
function Env:seed(n)
    self.rng = Random.new(n)
end

function Env:reset()
    self.pos = Vector2.new(0, 0)
    self.vel = Vector2.new(0, 0)
    self.target = Vector2.new(
        self.rng:NextNumber(-50, 50),
        self.rng:NextNumber(-50, 50)
    )
    self.steps = 0
    return self:observe()
end

function Env:observe()
    local toTarget = self.target - self.pos
    return {
        self.pos.X / 50, self.pos.Y / 50,
        self.vel.X / 10, self.vel.Y / 10,
        toTarget.X / 50, toTarget.Y / 50,
        toTarget.Magnitude / 50,
        self.steps / self.maxSteps,
    }
end

function Env:step(action)
    local before = (self.target - self.pos).Magnitude

    local accel = Vector2.new(
        math.clamp(action[1], -1, 1),
        math.clamp(action[2], -1, 1)
    )
    self.vel = (self.vel + accel) * 0.9      -- drag
    self.pos = self.pos + self.vel
    self.steps += 1

    local after = (self.target - self.pos).Magnitude

    -- dense shaping: reward progress, not just arrival
    local reward = (before - after) * 0.1 - 0.01

    local terminated = after < 2
    if terminated then
        reward += 10
    end

    local truncated = self.steps >= self.maxSteps

    return self:observe(), reward, terminated, truncated
end
```

Four things in there are the difference between a task that trains and one that does not.

**Clamp the action in the environment, not before scoring it.** The agent samples from an
unclamped Gaussian and stores that log-prob. Clamping the action before it reaches the
agent's bookkeeping makes every stored log-prob describe a different action than the one
taken.

**Observations are pre-scaled to roughly `[-1, 1]`.** The agent normalises too — every
agent has `normalizeObs = true` by default — but raw values in the hundreds make the
running statistics take a long time to settle.

**The reward is dense.** `(before - after) * 0.1` rewards *progress* every step. With
reward only on arrival, the agent has to stumble onto the target by chance before it
learns anything at all, and on a 100x100 plane that may never happen. The `-0.01` per step
makes it prefer arriving sooner.

**`terminated` and `truncated` are distinct.** Reaching the target terminates — there is no
future from there. Hitting the step limit truncates — the world would have continued, and
the next state still has value. Collapsing them teaches the critic that value is zero at
every episode boundary, which is the single most common silent bug in RL.

## Check it before you train

```lua
local env = Env.new()
local ok, problems = nano.train.checkEnv(env, {0, 0})

if not ok then
    for _, p in ipairs(problems) do warn(p) end
    return
end
```

One step, and it names anything wrong: missing methods, wrong return count, non-finite
observations, a `done` returned where two booleans were expected. Cheap insurance against
an afternoon spent debugging an agent that was never getting valid data.

## The agent

```lua
local agent = nano.algorithms.PPO.new({
    obsSize = 8,
    actionDim = 2,
    actionType = "continuous",
    hidden = 64,
    depth = 2,
    horizon = 2048,
    gamma = 0.99,
    entropyCoef = 0.005,
})
```

`obsSize` must match your observation length exactly. `actionDim` is the number of
continuous dimensions — for discrete actions it would be the number of choices, with
`actionType = "discrete"`.

`horizon` is how many steps are collected before an update. **It must be smaller than your
total step count**, or the buffer never fills and no update ever runs — the loop executes,
returns log, throughput reports, and the policy never moves.

Everything else has a defensible default. See
[`algorithms`](../reference/algorithms.md#options) for the full list.

## Training

```lua
nano.seed(20240607)

local trainer = nano.train.Trainer.new(agent, env, {
    steps = 200000,
    evalEvery = 10000,
    evalEpisodes = 20,
    evalEnv = Env.new(),      -- a SECOND instance
    keepBest = true,
    timeBudget = 0.008,
})

local summary = trainer:run()
trainer:restoreBest()

print(("%d steps, %d updates, %d episodes, %.1f steps/sec, best %.2f")
    :format(summary.steps, summary.updates, summary.episodes,
            summary.stepsPerSecond, summary.bestScore))
```

**Pass `evalEnv` as a separate instance.** Evaluating on the training environment calls
`reset` on it mid-rollout, so the training loop resumes with an observation that no longer
describes the world, and every transition after an evaluation is garbage.

**`keepBest` needs `env:seed`.** Evaluation reseeds with a fixed `evalSeed` so every
evaluation faces the same twenty episodes and scores are comparable. Without a `seed`
method, scores vary by luck and `keepBest` tracks the luckiest evaluation rather than the
best policy. `evaluate` returns a second value telling you whether the score was
deterministic, precisely so an incomparable one does not drive `keepBest`.

**`timeBudget`, not a step count.** A step that triggers a gradient update costs roughly
100x a plain environment step. `0.008` seconds keeps the server responsive; raise it if
nothing else is running.

**Check `summary.updates`.** Zero means the agent collected 200,000 transitions and learned
from none of them. The Trainer warns loudly about this, but check anyway.

## Reading the run

Each log line reports the agent's update statistics:

| stat | healthy | what a bad value means |
|---|---|---|
| `meanReward` | rising | The only thing that ultimately matters |
| `explainedVariance` | > 0.3, rising | Near 0: the critic is useless, so every advantage is noise |
| `clipFraction` | 0.05 – 0.3 | ~0: lr too small, nothing happening. > 0.5: updates far too aggressive |
| `approxKL` | < 0.02 | Spikes mean the policy is jumping and about to destabilise |
| `entropy` | falling slowly | A cliff means premature convergence — raise `entropyCoef` |

`explainedVariance` is the fastest diagnostic in the list. If it is stuck near zero, the
problem is the critic, not the policy, and no amount of tuning `clip` or `entropyCoef` will
help.

## When it does not learn

Work through this in order. Most failures are in the first two.

**1. Is the reward what you think it is?** Print it for a few episodes. Run a random policy
and record the mean return — that is your baseline, and if the agent is not beating it,
nothing is working.

**2. Are `terminated` and `truncated` right?** Run `checkEnv`. Then check that reaching the
goal sets `terminated` and only the step limit sets `truncated`.

**3. Is `explainedVariance` moving?** If not, the critic cannot predict returns. Usually the
reward scale is wild — returns in the thousands make value targets the critic cannot fit.
Keep per-step rewards roughly in `[-1, 1]`.

**4. Are observations finite?** One `nan` poisons everything downstream, silently.

```lua
nano.train.printDiagnosis(nano.diagnose(agent.actor))
```

**5. Did updates actually run?** `summary.updates`.

**6. Is the task solvable from the observation?** If a human could not do it from what the
agent sees, no algorithm can. Either add to the observation, or use `RecurrentPPO` if the
missing information is something the agent saw earlier.

## Adding memory

If the target can leave view, or a cue appears once at the start of an episode, the current
observation is insufficient and feedforward PPO cannot solve it — no matter how long you
train.

```lua
local agent = nano.algorithms.RecurrentPPO.new({
    obsSize = 8,
    actionDim = 2,
    hidden = 64,
    cell = "gru",
    layers = 1,
    horizon = 512,
    seqLen = 16,        -- must divide horizon
    seqPerBatch = 8,
})
```

Same `act`/`observe`/`ready`/`update` shape, plus `agent:resetState()` at each episode
boundary. Start with `"gru"` — one update gate instead of the LSTM's separate input and
forget gates, ~25% fewer parameters, comparable quality on most tasks.

One warning when testing this: **hiding a variable does not make a memory task.** It must
not be recoverable from anything the agent can observe or influence. An early version of
Nano's own recurrent test hid the target but left the agent's position visible — so a
memoryless policy encoded the target into its own position on step 1 and read it back
forever after, using the world as memory. It scored well and remembered nothing.

## Other algorithms

Same environment, different agent:

```lua
-- Off-policy continuous. Far more sample-efficient when env steps are expensive.
local agent = nano.algorithms.SAC.new({
    obsSize = 8, actionDim = 2, bufferSize = 100000, warmup = 1000,
})

-- Discrete actions. Double DQN by default.
local agent = nano.algorithms.DQN.new({
    obsSize = 8, actionCount = 4, epsilonDecay = 50000,
})
```

SAC and DQN update inside `observe`, so they have no `ready`/`update` in the loop — the
Trainer detects this automatically via `train.updateMode`.

For DQN, set `epsilonDecay` relative to your run length. Decaying over 20,000 steps in a
500,000-step run means exploration stops in the first 4% of training.

## Saving the agent

```lua
local state = nano.algorithms.save(agent)
-- ... later
nano.algorithms.load(agent, state)
```

Use these rather than `nano.save` on the networks. `algorithms.save` round-trips the
observation-normaliser state too, and a policy restored without its normaliser sees inputs
on a different scale than it trained on.

See [Deployment](deployment.md) for running the trained policy on many NPCs at once.

## Next

- [Deployment](deployment.md) — running the result at scale
- [`algorithms` reference](../reference/algorithms.md) — every agent and option
- [`train` reference](../reference/train.md) — Trainer and diagnostics
- [`rl` reference](../reference/rl.md) — buffers, GAE, distributions
