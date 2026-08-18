# nano

The table returned by `require(ReplicatedStorage.Nano)`. Holds every submodule plus
shorthand for the most-used functions.

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
```

## Modules

| Member | Description |
| --- | --- |
| `nano.Config` | Global gradient switch and the shared seedable RNG. [Reference](config.md) |
| `nano.Tensor` | Strided storage and reverse-mode autograd. The core. [Reference](tensor.md) |
| `nano.F` · `nano.functional` | Extra tensor ops, activations, fused kernels. The same table under both names. [Reference](functional.md) |
| `nano.nn` | Layers, containers, losses, compiled inference. [Reference](nn.md) |
| `nano.optim` | Optimizers, clipping, LR schedules. [Reference](optim.md) |
| `nano.rnn` | Recurrent cells, state helpers, sequence runners. [Reference](rnn.md) |
| `nano.data` | Datasets, batching, scaling, metrics, early stopping. [Reference](data.md) |
| `nano.rl` | Buffers, GAE, distributions, normalizers. [Reference](rl.md) |
| `nano.algorithms` | PPO, RecurrentPPO, SAC, DQN. [Reference](algorithms.md) |
| `nano.train` | Environment protocol, Trainer, diagnostics. [Reference](train.md) |
| `nano.serialize` | Save and load weights and optimizer state. [Reference](serialize.md) |
| `nano.parallel` | Actor worker pools and evolution. [Reference](parallel.md) |

## Properties

| Member | Description |
| --- | --- |
| `nano.VERSION: string` | Version string, currently `"3.4.0"`. |

## Construction shorthand

| Member | Description |
| --- | --- |
| `nano.tensor(data, shape?, requiresGrad?)` → `Tensor` | Alias for `Tensor.new`. |
| `nano.zeros(shape, requiresGrad?)` → `Tensor` | Alias for `Tensor.zeros`. |
| `nano.ones(shape, requiresGrad?)` → `Tensor` | Alias for `Tensor.ones`. |
| `nano.full(shape, value, requiresGrad?)` → `Tensor` | Alias for `Tensor.full`. |
| `nano.randn(shape, requiresGrad?)` → `Tensor` | Alias for `Tensor.randn`. |
| `nano.rand(shape, requiresGrad?)` → `Tensor` | Alias for `Tensor.rand`. |

## Math shorthand

| Member | Description |
| --- | --- |
| `nano.matmul(a, b)` → `Tensor` | Matrix multiply. Alias for `Tensor.matmul`. |
| `nano.exp(a)` → `Tensor` | Elementwise `e^x`. |
| `nano.log(a)` → `Tensor` | Elementwise natural log. |
| `nano.sqrt(a)` → `Tensor` | Elementwise square root. |
| `nano.abs(a)` → `Tensor` | Elementwise absolute value. |
| `nano.clamp(a, lo?, hi?)` → `Tensor` | Clamp into range. |
| `nano.softmax(a, dim?)` → `Tensor` | Softmax, `dim` defaults to the last dimension. |
| `nano.sum(a, dim?, keepdim?)` → `Tensor` | Sum over everything or one dimension. |
| `nano.mean(a, dim?, keepdim?)` → `Tensor` | Mean over everything or one dimension. |
| `nano.cat(tensors, dim?)` → `Tensor` | Concatenate along an existing dimension. |
| `nano.stack(tensors, dim?)` → `Tensor` | Stack along a new dimension. |
| `nano.min(a, b)` → `Tensor` | Elementwise minimum of two tensors, differentiable. Computed via `(a+b-|a-b|)/2`, which is exact rather than an approximation. What PPO's clipped surrogate and SAC's twin-critic minimum both need. |
| `nano.max2(a, b)` → `Tensor` | Elementwise maximum of two tensors, differentiable. Named `max2` because `Tensor.max` is the reduction. |

## Gradient control

| Member | Description |
| --- | --- |
| `nano.noGrad(fn)` → `T` | Run `fn` with graph construction disabled, then restore the previous state. Nested calls compose, and it restores on error. |
| `nano.gradcheck(fn, ...)` → `(boolean, number)` | Compare a backward pass against a central finite difference. Returns whether it passed and the worst relative error. |

## Randomness

| Member | Description |
| --- | --- |
| `nano.seed(value?)` → `()` | Seed every random source in Nano. Pass nothing to reseed unpredictably. |
| `nano.random()` → `number` | Uniform in `[0, 1)` from the shared seeded stream. |
| `nano.gaussian()` → `number` | One standard normal from the shared seeded stream. |

`nano.seed` covers tensor initialisation, dropout masks, exploration noise, replay
sampling, epsilon-greedy, GA mutation, minibatch order and dataset splits. Passing an
explicit seed to `Dataset:split` gives that split a private stream instead, so it stays
identical even if the amount of global randomness drawn beforehand changes.

## Diagnostics and deployment

| Member | Description |
| --- | --- |
| `nano.diagnose(model, options?)` → `table` | Alias for `train.diagnose`. Weight and gradient norms, non-finite counts, named problems. |
| `nano.compile(model)` → `function?` | Alias for `nn.compile`. Compile a `Sequential` into an allocation-free forward, or `nil` if unsupported. |

## Save and load

| Member | Description |
| --- | --- |
| `nano.save(model, metadata?)` → `table` | Alias for `serialize.toTable`. |
| `nano.load(model, state)` → `boolean` | Alias for `serialize.fromTable`. |

For agents rather than bare models use
[`algorithms.save` / `algorithms.load`](algorithms.md), which also round-trip
observation-normaliser state.
