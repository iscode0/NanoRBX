# Nano

A self-contained machine learning library for Roblox Luau. Neural networks, automatic
differentiation, and reinforcement learning, with no external dependencies and no HTTP calls.
Everything runs inside your experience.

**Version 3.4.0**

---

## What Nano is for

Nano trains and runs neural networks inside Roblox. In practice that means one of four things:

| you want to | you need | start at |
|---|---|---|
| Predict a number or a category from data you already have | Supervised learning | [Your first model](getting-started.md) |
| Teach an NPC to do something by trial and error | Reinforcement learning | [RL primer](concepts/rl-primer.md) |
| Run a trained brain on 100 NPCs at 60 fps | Compiled inference | [Deployment](guides/deployment.md) |
| Evolve behaviour across a population | Parallel evolution | [parallel](reference/parallel.md) |

If you have never done any of this before, read [What machine learning actually
is](concepts/ml-primer.md) first. It assumes no background — no calculus, no statistics, no
prior ML — and everything else in these docs builds on it.

---

## Install

In Studio, insert `NanoRBX` from the Toolbox — or by asset ID `138581366222476` — then move the Nano ModuleScript into ReplicatedStorage.

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
```

Full details, including the manual tree, are in [Installation](installation.md).

## Documentation map

### Start here

| page | what it covers |
|---|---|
| [Installation](installation.md) | Getting Nano into your place, correctly |
| [Getting started](getting-started.md) | Your first working model, line by line |

### Concepts

Read these once. They explain *why* the API looks the way it does, which makes the reference
pages read as obvious rather than arbitrary.

| page | what it covers |
|---|---|
| [What machine learning actually is](concepts/ml-primer.md) | Zero-background introduction: weights, loss, gradients, training |
| [Tensors and autograd](concepts/autograd.md) | Shapes, the computation graph, leaves vs results, `noGrad` |
| [Reinforcement learning primer](concepts/rl-primer.md) | Agents, rewards, episodes, why RL is harder than supervised learning |

### API reference

Every module, every public function and field, each with an immediate description.

| module | contents |
|---|---|
| [nano](reference/nano.md) | The top-level table: shorthand, save/load, version |
| [Config](reference/config.md) | Global gradient switch, seeding, the shared RNG |
| [Tensor](reference/tensor.md) | The core: storage, arithmetic, reductions, autograd |
| [functional](reference/functional.md) | Extra tensor ops, activations, fused kernels, init |
| [nn](reference/nn.md) | Layers, containers, losses, compiled inference |
| [optim](reference/optim.md) | SGD, Adam, AdamW, RMSprop, clipping, LR schedules |
| [rnn](reference/rnn.md) | RNN / LSTM / GRU cells, state helpers, sequence runners |
| [data](reference/data.md) | Datasets, batching, scaling, metrics, early stopping |
| [rl](reference/rl.md) | Buffers, GAE, distributions, normalizers, target networks |
| [algorithms](reference/algorithms.md) | PPO, SAC, DQN, RecurrentPPO — assembled and correct |
| [train](reference/train.md) | Environment protocol, Trainer, diagnostics |
| [serialize](reference/serialize.md) | Saving and loading weights and optimizer state |
| [parallel](reference/parallel.md) | Actor worker pools, weight transport, evolution |

### Guides

| page | what it covers |
|---|---|
| [Supervised learning walkthrough](guides/supervised.md) | A real classifier, from raw data to evaluated model |
| [Training an NPC with RL](guides/rl-agent.md) | Environment, agent, Trainer, and what the numbers mean |
| [Deploying a trained model](guides/deployment.md) | Saving, loading, `nn.compile`, running many NPCs |
| [Extending Nano](guides/extending.md) | Custom ops and custom layers, gradchecked |

### Reference material

| page | what it covers |
|---|---|
| [Performance](performance.md) | What is fast, what is slow, and what was measured |
| [Gotchas](gotchas.md) | Every mistake that produces wrong results without an error |

---

## The shape of every program

Nano has one core loop. Supervised learning, reinforcement learning, and everything else are
variations on it:

```lua
opt:zeroGrad()                              -- clear last step's gradients
local loss = criterion:forward(model:forward(X), Y)   -- how wrong are we?
loss:backward()                             -- how should each weight change?
opt:step()                                  -- change them
```

Four lines. If you understand what each one does, you can read every example in these docs.
[Getting started](getting-started.md) explains them one at a time.

---

## A note on silent failure

Machine learning code fails quietly. A wrong gradient does not throw an error — it trains a
slightly wrong model, and you spend a week blaming your learning rate. Nano is built around that
fact: it validates arguments aggressively, refuses to guess between `terminated` and `truncated`,
errors on architecture mismatch when loading weights, and ships a gradient checker you should
run after touching anything differentiable.

The [Gotchas](gotchas.md) page is a list of every failure mode found so far that produces wrong
answers instead of errors. It is worth reading before you need it.
