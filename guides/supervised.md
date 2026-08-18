# Supervised learning walkthrough

A complete classifier, from raw rows to an evaluated model. This assumes you have read
[Getting started](../getting-started.md) — the four-line loop here is the same one, with
everything real training needs wrapped around it.

## The task

Predict whether a player will finish a level, from five features you can measure in the
first thirty seconds. Two classes, so this is binary classification.

The data is `{{number}}` rows plus a matching array of class indices.

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
local T, nn, optim, data = nano.Tensor, nano.nn, nano.optim, nano.data

nano.seed(12345)

local inputs = {
    {12, 340, 0.8,  3, 1},
    {45, 120, 0.2, 11, 0},
    -- ... several thousand rows
}
local targets = { 1, 2, ... }   -- 1-based class indices
```

Class indices are **1-based**, matching Luau. `nn.CrossEntropyLoss` expects exactly this.

## Split before you do anything else

```lua
local dataset = data.Dataset.new(inputs, targets)
local train, valid = dataset:split(0.8, 12345)
```

80% to train on, 20% held back to measure honestly. The explicit seed makes the split
reproducible across runs, and independent of how much randomness was drawn before it.

The split shuffles, which matters more than it sounds. Data arrives sorted far more often
than expected — by time, by player, by outcome — and a tail split on sorted data hands you
a validation set full of cases the model never trained on.

## Scale the features

Look at the ranges above: one feature is `0.2–0.8`, another is `120–340`. The large one
dominates every dot product, and the network spends most of training just learning to
compensate.

```lua
local scaler = data.Scaler.new("standard")
local trainX = scaler:fitTransform(train.inputs)
local validX = scaler:transform(valid.inputs)
```

**Fit on training data only.** `fitTransform` on train, `transform` on validation. Fitting
on everything leaks validation statistics into training and makes your score optimistic by
an unknown amount — worse than being wrong, because you cannot tell how wrong.

Keep `scaler` around. A deployed model without the scaler it trained with receives inputs
on a different scale and produces confident nonsense.

```lua
local trainSet = data.Dataset.new(trainX, train.targets)
local validSet = data.Dataset.new(validX, valid.targets)
```

## The model

```lua
local model = nn.Sequential(
    nn.Linear(5, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 2)
)
```

Five features in, two classes out. Two hidden layers of 32 is a reasonable default for
tabular data with a few thousand rows — start small, and only grow it if the *training*
loss refuses to fall.

**No activation on the output layer.** `nn.CrossEntropyLoss` takes **logits** and applies
log-softmax internally, fused for numerical stability. Adding `nn.Softmax` before it
applies the transform twice and flattens your gradients.

```lua
local criterion = nn.CrossEntropyLoss()
local opt = optim.Adam(model:parameters(), 0.001)
```

`0.001` is Adam's default and the right place to start. If loss oscillates or explodes, go
down by 10x.

## The training loop

```lua
local loader = data.Loader.new(trainSet, 32)
local stopper = data.EarlyStopping.new(10, 1e-4)

for epoch = 1, 200 do
    model:train()

    local total, batches = 0, 0
    for X, Y in loader:iter() do
        opt:zeroGrad()
        local loss = criterion:forward(model:forward(X), Y)
        loss:backward()
        opt:step()

        total += loss:item()
        batches += 1
    end

    -- validation, no gradients
    model:eval()
    local validLoss, accuracy = nano.noGrad(function()
        local X = T.new(validSet.inputs)
        local logits = model:forward(X)
        return criterion:forward(logits, validSet.targets):item(),
               data.metrics.accuracy(logits, validSet.targets)
    end)

    if epoch % 10 == 0 then
        print(("epoch %d  train %.4f  valid %.4f  acc %.3f")
            :format(epoch, total / batches, validLoss, accuracy))
    end

    if stopper:update(validLoss, model) then
        print("early stop at epoch", epoch)
        break
    end
end

stopper:restore(model)
```

Several things are doing real work here.

**`data.Loader` batches for you.** Batch 32 is a sensible default. One `{32, 5}` forward
costs roughly a thirtieth of thirty-two `{1, 5}` passes, because per-operation overhead
dominates at these sizes. `Y` comes back as a plain array of class indices, which is what
`CrossEntropyLoss` wants — for row targets it would come back as a Tensor.

**`model:train()` and `model:eval()`** flip the `training` flag across the whole tree.
`nn.Dropout` is the only layer here that cares, and this model has none — but get in the
habit, because forgetting `eval()` on a model with dropout means you evaluate a randomly
crippled network and cannot work out why the score is bad.

**`nano.noGrad` around validation.** You are not going to call `backward()` on it, so
building the graph is pure waste — allocations and bookkeeping for gradients nobody uses.

**`EarlyStopping` snapshots.** `update` returns `true` when validation loss has not
improved for 10 epochs. Passing `model` makes it snapshot on every improvement, which is
what `restore` puts back at the end.

That last part matters more than the stop signal. Training past the optimum actively
degrades the model, so stopping late with the final weights is worse than stopping late
with a saved snapshot.

## Reading the two losses

| training | validation | meaning |
|---|---|---|
| falling | falling | Working. Keep going. |
| falling | flat, then rising | Overfitting. This is what EarlyStopping catches. |
| flat | flat | Underfitting — model too small, lr too low, or features carry no signal |
| `nan` | `nan` | lr too high, or a `nan` in the input data |
| falling | wildly noisy | Validation set too small to measure anything |

The gap between them is the thing to watch. A small gap means the model generalises. A
large and growing gap means it is memorising.

## Evaluating properly

Accuracy alone lies when classes are imbalanced. 95% accuracy on data that is 95% one
class means the model learned to always guess that class.

```lua
nano.noGrad(function()
    local logits = model:forward(T.new(validSet.inputs))

    print("accuracy", data.metrics.accuracy(logits, validSet.targets))

    local m = data.metrics.confusion(logits, validSet.targets, 2)
    for i = 1, 2 do
        print(("true class %d: %s"):format(i, table.concat(m[i], "  ")))
    end
end)
```

The confusion matrix shows it immediately — a row of zeros for one class means that class
is never predicted.

For regression tasks — predicting a number rather than a category — use `nn.MSELoss` and
report `data.metrics.rmse`, `mae` and `r2` instead.

## When it does not work

Run the diagnostics before changing hyperparameters:

```lua
nano.train.printDiagnosis(nano.diagnose(model))
nano.train.activationReport(model, validSet.inputs)
```

`diagnose` reports weight and gradient norms and names problems: diverged weights,
exploding gradients, all-zero gradients. All-zero gradients usually means a missing
`backward()` or a detached graph.

`activationReport` catches what weights alone cannot show. A ReLU layer where most units
output zero for every input is **dead** — those units contribute nothing and never
recover, because a ReLU at zero has zero gradient. If a large fraction is dead, lower the
learning rate or switch to `nn.LeakyReLU`.

Common fixes, in the order worth trying:

| symptom | try |
|---|---|
| Training loss will not fall | Larger model, higher lr, check your labels are right |
| Overfitting immediately | Smaller model, add `nn.Dropout(0.2)`, more data |
| Loss explodes | 10x lower lr, or `opt:clipGradNorm(1.0)` before `step` |
| Everything is `nan` | Check inputs for `nan`/`inf`; scaler on constant column divides by zero |
| Dead ReLU units | Lower lr, `nn.LeakyReLU`, or He init: `nn.Linear(5, 32, true, "he")` |

## Saving it

```lua
local state = nano.save(model, {
    note = "level-completion classifier",
    scalerMean = scaler.mean,
    scalerStd = scaler.std,
})
```

Save the scaler statistics alongside the weights. A model loaded without them is a
different model.

`nano.save` writes a plain table with an architecture fingerprint, so loading it into a
mismatched network fails cleanly rather than producing confident nonsense. See
[Deployment](deployment.md) for running it at scale.

## Next

- [Deployment](deployment.md) — `nn.compile`, saving, running many at once
- [`data` reference](../reference/data.md) — every loader, scaler and metric option
- [`nn` reference](../reference/nn.md) — all layers and losses
- [Gotchas](../gotchas.md) — the mistakes that do not error
