# Getting started

This page builds one working model and explains every line. It assumes you have
[installed Nano](installation.md). It does not assume you know what a neural network is —
if you want that explained properly, read [What machine learning actually
is](concepts/ml-primer.md) first, then come back.

## The problem

We will teach a network the XOR function: two inputs, output 1 if exactly one input is 1,
otherwise 0.

| input | output |
|---|---|
| 0, 0 | 0 |
| 0, 1 | 1 |
| 1, 0 | 1 |
| 1, 1 | 0 |

XOR is the standard first example because it is the simplest problem a single layer
*cannot* solve. If your network learns it, your layers, activations, loss and optimizer
are all wired up correctly.

## The whole program

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
local T, nn, optim = nano.Tensor, nano.nn, nano.optim

local X = T.new({{0,0}, {0,1}, {1,0}, {1,1}})
local Y = T.new({{0},   {1},   {1},   {0}})

local model = nn.Sequential(
    nn.Linear(2, 8), nn.Tanh(),
    nn.Linear(8, 1), nn.Sigmoid()
)

local opt = optim.Adam(model:parameters(), 0.05)
local criterion = nn.MSELoss()

for step = 1, 2000 do
    opt:zeroGrad()
    local prediction = model:forward(X)
    local loss = criterion:forward(prediction, Y)
    loss:backward()
    opt:step()

    if step % 500 == 0 then
        print(step, loss:item())
    end
end

nano.noGrad(function()
    print(model:forward(X))
end)
```

Run it. The loss should fall toward zero and the final output should be close to
`{0, 1, 1, 0}`.

## Line by line

### The data

```lua
local X = T.new({{0,0}, {0,1}, {1,0}, {1,1}})
```

`X` has shape `{4, 2}` — **four samples, two features each**. Every tensor in Nano that
represents data is `{batch, features}`, in that order.

The double braces are not decoration. `T.new({0, 0})` gives you a `{2}` tensor — a
vector, not a batch of samples — and `matmul` requires two 2D tensors, so it fails
outright. A single sample is a batch of one: `T.new({{0.3, 0.7}})`.

This is the most common early mistake in the whole library.

### The model

```lua
local model = nn.Sequential(
    nn.Linear(2, 8), nn.Tanh(),
    nn.Linear(8, 1), nn.Sigmoid()
)
```

[`nn.Sequential`](reference/nn.md) runs layers in order, feeding each one's output into
the next.

`nn.Linear(2, 8)` takes 2 features in, produces 8 out. It computes `x @ W + b` — every
output is a weighted sum of every input, plus a bias. `W` and `b` start random and are
what training changes.

`nn.Tanh()` is the activation. Without something non-linear between the two Linear
layers, stacking them is pointless: two matrix multiplies in a row collapse into one
matrix multiply, and you are back to a single layer, which cannot solve XOR. The
activation is what makes depth mean anything.

`nn.Sigmoid()` squashes the final output into `(0, 1)`, which is where our targets live.

Shapes have to chain: the `8` out of the first Linear is the `8` into the second. If they
do not match, `matmul` errors with both shapes printed.

### The optimizer

```lua
local opt = optim.Adam(model:parameters(), 0.05)
```

`model:parameters()` returns a flat list of every learnable tensor in the model — here,
two weight matrices and two bias vectors. [`optim.Adam`](reference/optim.md) is the
optimizer: given gradients, it decides how to change each parameter.

`0.05` is the learning rate. Too small and training crawls; too large and it diverges.
`0.05` is unusually high, which is fine for a four-sample toy problem. For real work,
Adam's default of `0.001` is a better starting point.

One thing to know now: `parameters()` is **cached**. `table.insert(model:parameters(), x)`
mutates the cached list permanently. `table.clone` it first if you need to extend it.

### The loss

```lua
local criterion = nn.MSELoss()
```

Mean squared error: average of `(prediction - target)²`. It is the standard choice when
your target is a number. For classification into categories you would use
`nn.CrossEntropyLoss` instead — and note that one takes **logits**, so you would drop the
final `nn.Sigmoid()`.

The loss is a single number measuring how wrong the model currently is. Training is
entirely about making it smaller.

### The training loop

These four lines are the whole of Nano:

```lua
opt:zeroGrad()
local loss = criterion:forward(model:forward(X), Y)
loss:backward()
opt:step()
```

**`opt:zeroGrad()`** clears the gradients from the previous step. Gradients *accumulate*
in Nano — `backward()` adds into `grad` rather than overwriting it, which is what makes
multi-branch models work. The cost is that you must clear them yourself each step.
Forgetting produces no error, just training that behaves inexplicably.

**`model:forward(X)`** runs the data through the network and produces predictions. As a
side effect it records the computation graph: every intermediate tensor remembers which
operation produced it and which tensors went in.

**`criterion:forward(prediction, Y)`** reduces those predictions and the targets to one
scalar. `backward()` requires a scalar — you cannot differentiate from a whole tensor.

**`loss:backward()`** walks that graph backwards and works out, for every parameter, which
direction would make the loss smaller. It stores the answer in each parameter's `.grad`.
This is [automatic differentiation](concepts/autograd.md), and it is the reason the
library exists.

**`opt:step()`** actually moves the parameters using those gradients.

Miss any one of the four and you get no error — just a model that does not learn. That is
worth internalising early.

### Inference

```lua
nano.noGrad(function()
    print(model:forward(X))
end)
```

`noGrad` turns off graph construction. When you are not going to call `backward()`,
building the graph is pure waste — allocations and bookkeeping for gradients nobody will
use. Wrap every forward pass you will not differentiate: evaluation, deployed inference,
rollout collection.

It restores the previous state rather than setting it back to `true`, so nested calls
compose, and it restores even if your function throws.

## Making runs reproducible

```lua
nano.seed(12345)
```

Put this before you build anything. It seeds tensor initialisation, dropout masks,
exploration noise, replay sampling and epsilon-greedy from one stream. Two runs with the
same seed produce identical weights.

Without it you cannot tell a real improvement from a lucky seed. That is not
hypothetical — an evaluation running on a drifting RNG once made a "keep best weights"
feature track the luckiest *evaluation* rather than the best *policy*.

Two shuffles are **not** covered and take their own seed: `Dataset:split(fraction, seed)`
and `data.Loader:iter()`.

## When it does not work

| symptom | usual cause |
|---|---|
| Loss does not move at all | Missing one of the four loop lines, usually `opt:step()` |
| Loss is `nan` | Learning rate too high, or `exp` overflowing somewhere |
| Loss falls then explodes | Learning rate too high — try 10x smaller |
| Loss stalls around 0.25 on XOR | No activation between the Linear layers |
| `matmul` shape error | Missing the batch dimension — check your braces |

`nano.train.printDiagnosis(nano.diagnose(model))` prints per-tensor weight and gradient
norms and names the usual problems: diverged weights, exploding gradients, all-zero
gradients.

## Where to go next

- [Tensors and autograd](concepts/autograd.md) — what `backward()` actually does
- [Supervised learning walkthrough](guides/supervised.md) — the same loop on real data,
  with batching, validation and metrics
- [Reinforcement learning primer](concepts/rl-primer.md) — if you want to train an NPC
- [Gotchas](gotchas.md) — every silent failure mode, in one list
