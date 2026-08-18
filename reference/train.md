# train

The environment protocol, the training loop, and diagnostics.

```lua
local train = nano.train
```

## Env protocol

An environment is any table with these methods.

```lua
env:reset()        -> obs
env:step(action)   -> obs, reward, terminated, truncated
env:seed(n)        -> ()        optional, strongly recommended
```

`terminated` means the state has no future. `truncated` means you stopped watching — a
step limit, a timeout. Collapsing them into one `done` teaches the critic that value is
zero at every episode boundary, and looks exactly like "the algorithm does not work on my
task".

`seed` is optional only because not every task can support it. Without it evaluation is
not reproducible, so scores cannot be compared across time and `keepBest` degenerates into
keeping the luckiest evaluation rather than the best policy.

| Member | Description |
| --- | --- |
| `train.checkEnv(env, sampleAction)` → `(boolean, {string})` | Validate a real step against the protocol and name what is wrong. |
| `train.updateMode(agent)` → `string` | `"rollout"` for PPO-style agents, `"internal"` for SAC and DQN. Errors on an agent that cannot learn. |

Run `checkEnv` before your first training run. It costs one step and catches the whole
class of "my environment returns the wrong thing" bugs that otherwise surface as an agent
that mysteriously does not learn.

The update path is detected from the agent's methods, not a tag. A missing tag would
silently disable learning: the loop runs, returns log, throughput reports, and the policy
never moves.

## Trainer

| Member | Description |
| --- | --- |
| `train.Trainer.new(agent, env, config?)` → `Trainer` | Create a trainer. See `Trainer.defaults`. |
| `train.Trainer.defaults: table` | Every option with its default value. |
| `trainer.steps: number` | Environment steps taken. |
| `trainer.episodes: number` | Episodes completed. |
| `trainer.returns: {number}` | Recent episode returns. |
| `trainer.bestScore: number` | Best evaluation score seen. |
| `trainer.bestWeights` | Snapshot taken at `bestScore`, or `nil`. |
| `trainer.history: {table}` | One log entry per `logEvery` interval. |
| `trainer:run()` → `table` | Run the whole training loop. |
| `trainer:evaluate(episodes?, maxSteps?)` → `(number, boolean)` | Greedy evaluation. Returns the mean score and whether it was deterministic. |
| `trainer:snapshot()` → `table` | Capture the agent's current weights. |
| `trainer:restore(snap)` → `()` | Load a snapshot back. |
| `trainer:restoreBest()` → `boolean` | Load the best-scoring snapshot. Returns `false` if there is none. |
| `trainer:printLog(entry)` → `()` | Print one history entry. |

`run` returns `steps`, `updates`, `episodes`, `elapsed`, `stepsPerSecond`, `bestScore` and
`history`.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `steps` | `50000` | Total environment steps. |
| `timeBudget` | `0.008` | Seconds of work between yields. |
| `logEvery` | `5000` | Steps between log entries. |
| `evalEvery` | `0` | Steps between evaluations. `0` disables. |
| `evalEpisodes` | `20` | Episodes per evaluation. |
| `evalMaxSteps` | `500` | Step cap per evaluation episode. |
| `evalEnv` | `nil` | A **second** environment instance. Strongly recommended. |
| `evalSeed` | `20240607` | Fixed, so scores are comparable across time. |
| `keepBest` | `true` | Snapshot weights whenever evaluation improves. |
| `verbose` | `true` | Print log entries as they happen. |
| `onLog` | `nil` | Optional `(stats) -> ()` callback. |

```lua
local trainer = train.Trainer.new(agent, env, {
    steps = 100000,
    evalEvery = 5000,
    evalEnv = makeEnv(),
    keepBest = true,
})
local summary = trainer:run()
trainer:restoreBest()
```

**Pass `evalEnv`.** Evaluating on the training environment resets it mid-rollout, so the
loop resumes with an observation that no longer describes the environment's state, and
every transition immediately after an evaluation is garbage. Without one the trainer
resynchronises afterwards, which is correct but throws away the episode in progress.

Evaluation reseeds via `env:seed(evalSeed)` when the environment supports it, so every
evaluation faces the same episodes. Without that, `keepBest` tracks luck rather than
skill — and `evaluate` returns a second value telling you whether the score was
deterministic, precisely so an incomparable score does not drive `keepBest`.

### What the Trainer owns

The rules that are easy to get wrong:

**Yield on a time budget, not a step count.** A step triggering a gradient update costs
roughly 100x a plain environment step, so any fixed step count either stalls the server
during update-heavy stretches or wastes most of the frame during cheap ones. Measuring
elapsed time adapts automatically.

**Update at the right moment for the algorithm in hand**, detected from the agent's
methods.

**Evaluate greedily.** A policy that only looks competent while adding exploration noise
has not learned much.

**Keep the best weights, not the last.** RL performance is not monotonic; the final policy
is frequently worse than one from 20,000 steps earlier.

**Warn when nothing learned.** A rollout agent that finished a run without a single update
has been fed thousands of transitions and learned from none of them. Nothing errors when
this happens, so it is checked explicitly and warned about by name.

`summary.updates` reports the agent's own counter for internal-update agents, because SAC
and DQN update inside `observe` where the trainer cannot see them — reporting zero there
would be simply wrong.

## Diagnostics

| Member | Description |
| --- | --- |
| `train.diagnose(model, options?)` → `table` | Per-tensor weight and gradient norms, non-finite counts, and named problems. `report.problems` is empty when healthy. |
| `train.printDiagnosis(report)` → `()` | Print a diagnose report. |
| `train.activationReport(model, samples)` → `table` | Dead-ReLU and saturation fractions per layer, from sample rows. |

```lua
train.printDiagnosis(nano.diagnose(model))
train.activationReport(model, sampleRows)
```

Named problems include diverged weights, exploding gradients and all-zero gradients — each
a silent failure that produces plausible-looking training and no error.

`activationReport` catches the two failure modes you cannot see from weights alone: a ReLU
layer where most units output zero for every input (dead, and they never recover), and a
tanh or sigmoid layer saturated at its extremes (gradients near zero, so it stops
learning).
