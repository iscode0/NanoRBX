# What machine learning actually is

No background assumed. No calculus, no statistics, no prior ML. If you can read the Luau
in these docs, you can read this page.

## The idea

Normally you write a function. You know the rule, you type it out:

```lua
local function isNight(hour)
    return hour < 6 or hour > 20
end
```

Machine learning is for the cases where you *cannot* type out the rule. Not because it is
hard to write, but because you do not know what it is. "Is this player about to quit?"
"Which way should this NPC move to reach cover?" You have examples of the right answer,
but no formula.

So instead of writing the function, you write a *shape* for the function with a few
thousand adjustable numbers in it, and then adjust those numbers until it agrees with
your examples.

That is it. Everything below is detail about how the adjusting works.

## Weights

The adjustable numbers are called **weights** (or parameters). A tiny model might be:

```
output = w1 * input1 + w2 * input2 + b
```

Three numbers: `w1`, `w2`, `b`. Pick good values and this predicts something useful. Pick
random values and it predicts noise. Training is the process of getting from the second
to the first.

A real model has more of them arranged in layers, but the principle does not change. The
XOR model in [Getting started](../getting-started.md) has 33 weights. A model for a
serious game task might have a few thousand. They are all just numbers in a list, and
`model:getFlat()` will hand them to you as one flat Lua array if you want to look.

## Layers

One layer of the shape above is written `nn.Linear(inFeatures, outFeatures)`. It computes
every output as a weighted sum of every input, plus a bias.

Stacking two of them directly gains you nothing — a weighted sum of weighted sums is
still a weighted sum, so two Linear layers in a row collapse mathematically into one.
That is why every stack has an **activation** between the layers:

```lua
nn.Linear(2, 8), nn.Tanh(),
nn.Linear(8, 1), nn.Sigmoid()
```

`nn.Tanh()` bends the output — anything that is not a straight line will do, and there
are a dozen options in [`nn`](../reference/nn.md). The bend is what lets a deep stack
represent something a single layer cannot. XOR is the classic demonstration: one layer
provably cannot solve it, two layers with an activation between them can.

## Loss

To improve the weights you need a number saying how wrong they currently are. That number
is the **loss**, and the function producing it is the **criterion**.

The simplest is mean squared error: for each example, take `(prediction - target)`, square
it, average over all examples.

```lua
local criterion = nn.MSELoss()
local loss = criterion:forward(prediction, target)
```

Squaring does two jobs: it makes errors positive so they cannot cancel out, and it
punishes large errors disproportionately. Being off by 10 is a hundred times worse than
being off by 1, not ten times.

Different tasks need different criteria. Predicting a number → `nn.MSELoss` or
`nn.HuberLoss`. Choosing between categories → `nn.CrossEntropyLoss`. The full list is in
[`nn`](../reference/nn.md).

Training is now a concrete goal: **find the weights that make the loss small.**

## Gradients

You have thousands of weights and a loss. You need to know which way to move each one.

Trying them one at a time would work and would be unusably slow — nudge weight 1, see if
loss improved, put it back, nudge weight 2. Thousands of full forward passes per step.

The **gradient** is the shortcut. For each weight it answers: if I increase this weight
slightly, does the loss go up or down, and how sharply? A positive gradient means
increasing the weight increases the loss, so you should decrease it. Larger magnitude
means this weight matters more.

The remarkable part is that you can compute the gradient for *every* weight in roughly
the cost of two forward passes, no matter how many weights there are. The technique is
called backpropagation, and Nano does it for you when you call:

```lua
loss:backward()
```

After that line, every parameter has its gradient sitting in `.grad`. How that works is
[Tensors and autograd](autograd.md). You do not need to understand it to use it, but you
do need to know it can be silently wrong if you write custom operations — which is why
[`Tensor.gradcheck`](../reference/tensor.md) exists.

## The optimizer

Gradients say which way to move. The **optimizer** decides how far.

The naive version, gradient descent, moves each weight a fixed fraction of its gradient:

```
weight = weight - learningRate * gradient
```

`learningRate` is the step size. Too small and training takes forever. Too large and you
overshoot the target every step and the loss explodes. It is the single hyperparameter
most worth tuning, and 10x changes are the right granularity to try.

In practice everyone uses **Adam**, which keeps a running average of each weight's recent
gradients and scales its step accordingly. A weight with consistently large gradients
gets a smaller relative step; one with small noisy gradients gets a larger one. It is
much less sensitive to a badly chosen learning rate.

```lua
local opt = optim.Adam(model:parameters(), 0.001)
opt:step()
```

`0.001` is Adam's usual default and a reasonable place to start. The options are in
[`optim`](../reference/optim.md).

## The loop

Put those together and you have training:

```lua
for step = 1, 2000 do
    opt:zeroGrad()                                        -- clear old gradients
    local loss = criterion:forward(model:forward(X), Y)   -- how wrong are we?
    loss:backward()                                       -- which way for each weight?
    opt:step()                                            -- move them
end
```

Each pass makes the model slightly less wrong. Run it enough times and the loss falls.

`zeroGrad` is there because Nano *accumulates* gradients — `backward()` adds into `.grad`
rather than replacing it. That is deliberate and necessary for models where one tensor
feeds two branches, but it means stale gradients pile up if you do not clear them. This
produces no error. It is the single most common bug in first training loops.

## Batches

You do not run one example at a time. You run many at once — a **batch** — and average
the loss over them.

```lua
local X = T.new({{0,0}, {0,1}, {1,0}, {1,1}})   -- {4, 2}: 4 samples, 2 features
```

Two reasons. The gradient from a single example is noisy — it points toward what is right
for that one sample, which may be wrong on average. Averaging over 32 gives a far more
useful direction.

And it is dramatically faster. One `{32, 8}` forward pass costs roughly a *thirtieth* of
thirty-two `{1, 8}` passes, because per-operation overhead dominates at these sizes. A
per-sample loop in your training step is your bottleneck, full stop.

This is why every shape in Nano is `{batch, features}` and why a single sample is written
`T.new({{0.3, 0.7}})` with double braces. See [Performance](../performance.md).

## Overfitting

A model with enough weights can memorise your training examples exactly. Loss goes to
zero. It has learned nothing — show it anything new and it fails.

The fix is to hold data back:

```lua
local train, valid = dataset:split(0.8, 12345)
```

Train on the 80%, measure on the 20% the model has never seen. Training loss falling
while validation loss rises means you are memorising, and you should stop.

[`data.EarlyStopping`](../reference/data.md) automates that: it watches validation loss,
tells you when to stop, and — more importantly — snapshots the best weights so you can
restore them. Training past the optimum actively degrades the model, so stopping late
with final weights is worse than stopping late with a saved snapshot.

## Two flavours of learning

Everything above is **supervised learning**: you have inputs paired with correct answers,
and you fit a function between them. Classifying, predicting, estimating. Walkthrough in
[Supervised learning](../guides/supervised.md).

The other flavour is **reinforcement learning**: no correct answers, just a reward signal
and an agent that has to work out what to do. Training an NPC to navigate, fight or
gather is RL. It uses the same tensors, layers and optimizer, but the loop around them is
substantially harder — see the [RL primer](rl-primer.md).

Start with supervised if you have a choice. It is far easier to tell whether it is
working.

## Vocabulary

| term | meaning |
|---|---|
| weight / parameter | An adjustable number inside the model |
| layer | One transformation step, e.g. `nn.Linear(2, 8)` |
| activation | The non-linear bend between layers, e.g. `nn.Tanh()` |
| forward pass | Running data through the model to get predictions |
| loss / criterion | A single number measuring how wrong the model is |
| gradient | Which way to move each weight to reduce the loss |
| backward pass | Computing all the gradients — `loss:backward()` |
| optimizer | Decides how far to move each weight — `optim.Adam` |
| learning rate | Step size |
| batch | A group of samples processed together |
| epoch | One full pass over the training data |
| overfitting | Memorising the training data instead of learning the pattern |
| inference | Using a trained model, no training |

## Next

- [Getting started](../getting-started.md) — build the XOR model
- [Tensors and autograd](autograd.md) — how `backward()` works
- [Supervised learning walkthrough](../guides/supervised.md) — a real task
