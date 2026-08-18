# data

Datasets, batching, feature scaling, metrics and early stopping.

```lua
local data = nano.data
```

## Dataset

Rows of features with matching targets. Targets may be rows (regression) or plain numbers
(class indices for `CrossEntropyLoss`).

| Member | Description |
| --- | --- |
| `data.Dataset.new(inputs, targets?)` → `Dataset` | Build from `{{number}}` rows. Errors if the counts disagree. |
| `dataset.inputs: {{number}}` | The feature rows. |
| `dataset.targets: {any}?` | The targets, or `nil`. |
| `dataset.count: number` | Row count. |
| `dataset:size()` → `number` | Row count. |
| `dataset:split(trainFraction, seed?)` → `(Dataset, Dataset)` | Shuffled train/validation split. |

**The split shuffles.** Data arrives sorted far more often than expected, and a naive tail
split on sorted data gives a validation set containing classes the model never trained on
— which reports as a catastrophically broken model rather than a data-handling mistake.

Pass `seed` to give the split a private stream, so it stays identical even if the amount
of global randomness drawn beforehand changes. With no seed it draws from the shared
stream, so `nano.seed` still covers it.

## Loader

| Member | Description |
| --- | --- |
| `data.Loader.new(dataset, batchSize, shuffle?)` → `Loader` | Minibatch iterator. `shuffle` defaults to `true`. |
| `loader.batchSize: number` | Rows per batch. |
| `loader.shuffle: boolean` | Whether to reshuffle each `iter` call. |
| `loader.dropLast: boolean` | Whether to discard a short final batch. Defaults to `false`. |
| `loader:iter()` → `function` | Generic-for iterator yielding `(X, Y)` per batch. |
| `loader:batchCount()` → `number` | Number of batches per epoch. |

```lua
for X, Y in data.Loader(train, 32):iter() do
    opt:zeroGrad()
    local loss = criterion:forward(model:forward(X), Y)
    loss:backward()
    opt:step()
end
```

`Y` comes back as a Tensor for row targets and a plain array for class indices, matching
what `MSELoss` and `CrossEntropyLoss` respectively expect.

`iter` shuffles through the shared seeded stream, so `nano.seed` covers minibatch order.

## Scaler

| Member | Description |
| --- | --- |
| `data.Scaler.new(mode?)` → `Scaler` | `"standard"` (default, zero mean and unit variance) or `"minmax"`. |
| `scaler:fit(rows)` → `self` | Learn the statistics. |
| `scaler:transform(rows)` → `{{number}}` | Apply learned statistics. |
| `scaler:fitTransform(rows)` → `{{number}}` | Fit then transform, in one call. |

```lua
local scaler = data.Scaler.new("standard")
local trainX = scaler:fitTransform(train.inputs)
local validX = scaler:transform(valid.inputs)   -- transform only
```

**Fit on training data only.** Fitting on everything leaks validation statistics into
training and makes the reported score optimistic by an unknown amount — which is worse
than being wrong, because you cannot tell how wrong.

Keep the scaler alongside the model. A model deployed without the scaler that was fit
during training receives inputs on a different scale and produces confident nonsense.

## Metrics

| Member | Description |
| --- | --- |
| `data.metrics.accuracy(logits, targets)` → `number` | Fraction of rows whose argmax matches the class index. |
| `data.metrics.rmse(prediction, target)` → `number` | Root mean squared error. |
| `data.metrics.mae(prediction, target)` → `number` | Mean absolute error. |
| `data.metrics.r2(prediction, target)` → `number` | Coefficient of determination. |
| `data.metrics.confusion(logits, targets, classes)` → `{{number}}` | Confusion matrix, rows are true classes. |

Accuracy alone hides class imbalance — 95% accuracy on a dataset that is 95% one class
means the model learned to always say that class. `confusion` shows it immediately.

## EarlyStopping

| Member | Description |
| --- | --- |
| `data.EarlyStopping.new(patience?, minDelta?)` → `EarlyStopping` | Stop when validation loss has not improved for `patience` checks by at least `minDelta`. |
| `stopper:update(loss, model?)` → `boolean` | Record one validation loss. Returns `true` when you should stop. Snapshots `model` on improvement if given. |
| `stopper:restore(model)` → `()` | Load the best snapshot back into the model. |

```lua
local stopper = data.EarlyStopping.new(10, 1e-4)

-- after each epoch
if stopper:update(validLoss, model) then break end
stopper:restore(model)
```

`restore` matters more than the stop signal. Training past the optimum actively degrades
the model, so stopping late with the final weights is worse than stopping late with a
snapshot. Pass the model to `update` so there is something to restore.
