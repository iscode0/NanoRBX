# algorithms

PPO, RecurrentPPO, SAC and DQN, assembled so every correctness fix lives in the library
rather than in a script you have to remember to copy.

```lua
local algorithms = nano.algorithms
```

New to RL? Read the [primer](../concepts/rl-primer.md) first — it explains what these
options mean and how to tell whether a run is working.

## The interaction pattern

All four agents share the same shape:

```lua
local agent = algorithms.PPO.new({ obsSize = 8, actionDim = 2 })

local action = agent:act(obs)
local nextObs, reward, terminated, truncated = env:step(action)
agent:observe(reward, terminated, truncated, nextObs)

if agent:ready() then                       -- PPO and RecurrentPPO only
    local stats = agent:update(nextObs)
end
```

**Act, then observe.** `act` caches the observation, log-prob and value internally, so a
log-prob can never be paired with the wrong observation.

`terminated` and `truncated` are **separate required arguments**. The library refuses to
guess. A truncated step without `nextObs` errors rather than silently bootstrapping zero.

SAC and DQN update from their replay buffer inside `observe`, so they have no `ready` or
`update(finalObs)` in the loop. `train.updateMode(agent)` reports which kind you have.

Every default here was paid for: gradient clipping always on, `logStd` clamped inside the
graph, actions never clamped before scoring, `parameters()` copied before extending,
advantages normalised per batch, policy head scaled to 0.01.

## Shared members

Present on every agent.

| Member | Description |
| --- | --- |
| `agent.config: table` | The resolved configuration, defaults merged with your overrides. |
| `agent.kind: string` | `"ppo"`, `"recurrentPPO"`, `"sac"` or `"dqn"`. |
| `agent.updates: number` | Gradient updates performed so far. |
| `agent.stats: table` | Statistics from the most recent update. |
| `agent.obsNorm: RunningNorm?` | The observation normaliser, or `nil` when `normalizeObs` is false. |
| `agent:act(obs)` | Sample an action and remember what produced it. |
| `agent:actGreedy(obs)` | Act without exploration. Use for evaluation and deployment. |
| `agent:observe(reward, terminated, truncated, nextObs?)` → `()` | Record the outcome of the last action. |

| Member | Description |
| --- | --- |
| `algorithms.save(agent)` → `table` | Serialise every network plus observation-normaliser state. |
| `algorithms.load(agent, state)` → `()` | Restore from `save`. |

Use these rather than `serialize` directly for agents — a policy restored without its
observation normaliser sees inputs on a different scale than it trained on.

## PPO

On-policy, continuous or discrete. The sensible default.

| Member | Description |
| --- | --- |
| `algorithms.PPO.new(config?)` → `PPO` | Create an agent. See `PPO.defaults`. |
| `algorithms.PPO.defaults: table` | Every option with its default value. |
| `ppo.actor` · `ppo.critic` | The two networks. |
| `ppo.logStd: Tensor?` | Learnable log standard deviation, one per dimension. Continuous only. |
| `ppo.actorOpt` · `ppo.criticOpt` | The two Adam optimizers. |
| `ppo.buffer: RolloutBuffer` | The rollout storage. |
| `ppo:act(obs)` | Sample an action. Continuous returns `{number}`, discrete returns a number. |
| `ppo:actGreedy(obs)` | The distribution mean, or the argmax action. |
| `ppo:valueOf(obs)` → `number` | The critic's estimate for one observation. |
| `ppo:ready()` → `boolean` | Whether the rollout buffer is full. |
| `ppo:update(finalObs)` → `table` | Run the full PPO update. |
| `ppo:updateMinibatch(obsRows, actions, oldLogps, advantages, returns)` → `(number, number, number)` | One minibatch. Called by `update`. |

`update` returns `meanReward`, `explainedVariance`, `clipFraction`, `approxKL`, `entropy`
and `updates`. What to watch for is in the [primer](../concepts/rl-primer.md#reading-the-numbers).

### Options

| Option | Default | Description |
| --- | --- | --- |
| `obsSize` | `8` | Observation length. |
| `actionDim` | `2` | Action dimensions (continuous) or action count (discrete). |
| `actionType` | `"continuous"` | `"continuous"` or `"discrete"`. |
| `hidden` | `64` | Hidden width. |
| `depth` | `2` | Hidden layer count. |
| `horizon` | `512` | Steps collected before an update. |
| `epochs` | `10` | Passes over each rollout. |
| `minibatch` | `64` | Transitions per gradient step. |
| `gamma` | `0.99` | Discount factor. |
| `lambda` | `0.95` | GAE trace decay. |
| `clip` | `0.2` | PPO ratio clip. |
| `entropyCoef` | `0.005` | Entropy bonus weight. Raise if the policy converges too early. |
| `valueCoef` | `0.5` | Value loss weight. |
| `maxGradNorm` | `0.5` | Gradient clipping threshold. |
| `actorLR` | `3e-4` | Actor learning rate. |
| `criticLR` | `1e-3` | Critic learning rate — the value function can and should learn faster. |
| `initLogStd` | `-0.7` | Initial log std, `exp(-0.7) ≈ 0.5`. |
| `logStdMin` · `logStdMax` | `-2.5` · `0.5` | Clamped **inside the graph**, not after the step. |
| `normalizeObs` | `true` | Wrap observations in a `RunningNorm`. |
| `normalizeAdvantages` | `true` | Normalise per batch, not per minibatch. |

**`horizon` must be smaller than your total step count.** A rollout agent whose buffer
never fills collects forever and never updates — no error, the loop runs, returns log, and
the policy never moves. The Trainer warns loudly when this happens.

## RecurrentPPO

PPO with memory, for partially observable tasks — occlusion, a target that leaves view, a
signal shown once at the start of an episode.

| Member | Description |
| --- | --- |
| `algorithms.RecurrentPPO.new(config?)` → `RecurrentPPO` | Create an agent. Errors if `horizon` is not divisible by `seqLen`. |
| `algorithms.RecurrentPPO.defaults: table` | Every option with its default value. |
| `agent.trunk` | The recurrent stack, shared between actor and critic. |
| `agent.actorHead` · `agent.criticHead` | The two output layers. |
| `agent.state` | The current hidden state. |
| `agent:resetState()` → `()` | Zero the hidden state. Call at an episode boundary. |
| `agent:clearBuffer()` → `()` | Reset the rollout storage. |
| `agent:valueOf(obs)` → `number` | Critic estimate for one observation. |
| `agent:ready()` → `boolean` | Whether the horizon has been reached. |
| `agent:update(finalObs)` → `table` | Run the sequence-based update. |
| `agent:updateSequences(starts, L, componentCount, advantages, returns)` → `(number, number, number)` | One minibatch of sequences. Called by `update`. |

### Options

Everything PPO has except `depth`, `minibatch`, `actorLR` and `criticLR`, plus:

| Option | Default | Description |
| --- | --- | --- |
| `cell` | `"gru"` | `"gru"`, `"lstm"` or `"rnn"`. |
| `layers` | `1` | Recurrent layer count. |
| `seqLen` | `16` | BPTT window. Must divide `horizon`. |
| `seqPerBatch` | `8` | Sequences per minibatch. |
| `epochs` | `4` | Passes over each rollout — lower than PPO's 10. |
| `lr` | `3e-4` | One learning rate for the whole shared network. |

### What differs from feedforward PPO

Three things, and each is where a careless implementation goes wrong.

**Minibatches are sequences, not timesteps.** Feedforward PPO shuffles individual
transitions freely because each is independent. Here a timestep is meaningless without the
hidden state that preceded it, so the rollout is chunked into fixed-length sequences and
whole sequences are shuffled. Shuffling timesteps would destroy the very thing being
learned.

**The hidden state at each sequence start is stored and replayed.** The update cannot
recompute it — the parameters have changed since the rollout — so it is recorded during
collection and fed back as a constant.

**State is masked at episode boundaries.** A sequence can straddle a reset, and memory must
not leak between episodes. The mask multiplies state by zero, which is differentiable;
resetting in place would break the graph.

The trunk is shared between actor and critic: memory is expensive to learn and duplicating
it doubles that cost for no benefit.

Verified on a task where a cue appears on step 1 and never again, with reward only on the
final step. A memoryless policy sees an identical final observation every episode, so it
can only emit a constant — bounding it analytically at -0.2133. Trained feedforward PPO
scores -0.2328; RecurrentPPO scores -0.0058, closing 97% of the gap to perfect recall.

## SAC

Off-policy, continuous actions only. Twin critics with a learned entropy temperature. Far
more sample-efficient than PPO when environment steps are expensive.

| Member | Description |
| --- | --- |
| `algorithms.SAC.new(config?)` → `SAC` | Create an agent. |
| `algorithms.SAC.defaults: table` | Every option with its default value. |
| `sac.actor` · `sac.q1` · `sac.q2` | The policy and twin critics. |
| `sac:alpha()` → `number` | The current entropy temperature. |
| `sac:act(obs)` → `{number}` | Sample a squashed action. |
| `sac:actGreedy(obs)` → `{number}` | The squashed mean. |
| `sac:sampleWithLogProb(states, count)` → `(Tensor, Tensor)` | Reparameterised batch sample and its log-probability. |
| `sac:update()` → `table` | One gradient update. Called automatically from `observe`. |

### Options

| Option | Default | Description |
| --- | --- | --- |
| `obsSize` · `actionDim` | `8` · `2` | Observation and action sizes. |
| `hidden` · `depth` | `64` · `2` | Network shape. |
| `gamma` | `0.99` | Discount factor. |
| `tau` | `0.01` | Polyak coefficient for target critics. |
| `lr` | `3e-4` | Learning rate for actor, critics and temperature. |
| `batch` | `64` | Replay minibatch size. |
| `bufferSize` | `50000` | Replay capacity. |
| `warmup` | `1000` | Steps collected before the first update. |
| `updateEvery` | `4` | Environment steps per gradient update. |
| `maxGradNorm` | `0.5` | Gradient clipping threshold. |
| `initLogStd` | `-0.7` | Initial log std. |
| `logStdMin` · `logStdMax` | `-2.5` · `0.5` | Clamped inside the graph. |
| `targetEntropy` | `nil` | Defaults to `-actionDim`. |
| `normalizeObs` | `true` | Wrap observations in a `RunningNorm`. |

`Q(s, a)` needs both state and action. Nano has no concatenate op and does not need one: a
linear layer over `[s; a]` is exactly `W_s·s + W_a·a + b`, so two `Linear` layers summed is
the same computation. The action-side layer has no bias, because the state-side one already
supplies it.

## DQN

Discrete actions, **Double DQN** by default.

| Member | Description |
| --- | --- |
| `algorithms.DQN.new(config?)` → `DQN` | Create an agent. |
| `algorithms.DQN.defaults: table` | Every option with its default value. |
| `dqn.q` · `dqn.qTarget` | The live and target Q networks. |
| `dqn.steps: number` | Environment steps seen, which drives epsilon decay. |
| `dqn:epsilon()` → `number` | Current exploration rate, decayed linearly. |
| `dqn:act(obs)` → `number` | Epsilon-greedy action index. |
| `dqn:actGreedy(obs)` → `number` | Argmax action index. |
| `dqn:update()` → `table` | One gradient update. Called automatically from `observe`. |

### Options

| Option | Default | Description |
| --- | --- | --- |
| `obsSize` | `8` | Observation length. |
| `actionCount` | `4` | Number of discrete actions. |
| `hidden` · `depth` | `64` · `2` | Network shape. |
| `gamma` | `0.99` | Discount factor. |
| `lr` | `1e-3` | Learning rate. |
| `batch` | `64` | Replay minibatch size. |
| `bufferSize` | `50000` | Replay capacity. |
| `warmup` | `1000` | Steps collected before the first update. |
| `updateEvery` | `4` | Environment steps per gradient update. |
| `targetSync` | `500` | Hard target copy every N updates. |
| `maxGradNorm` | `10` | Gradient clipping threshold. |
| `epsilonStart` · `epsilonEnd` | `1.0` · `0.05` | Exploration range. |
| `epsilonDecay` | `20000` | Steps to reach `epsilonEnd`. |
| `doubleDQN` | `true` | Select with the live network, evaluate with the target. |
| `normalizeObs` | `true` | Wrap observations in a `RunningNorm`. |

DQN uses `HuberLoss(1)` rather than MSE — far less sensitive to outliers, which matters
when a single bad TD error would otherwise dominate the batch.

Set `epsilonDecay` relative to your total training steps. Decaying to 0.05 over 20,000
steps in a 500,000-step run means the agent stops exploring in the first 4% of training.
