# nn

Layers, containers, losses and compiled inference.

```lua
local nn = nano.nn
```

Every layer class is callable, so `nn.Linear(2, 8)` constructs, and every instance is
callable, so `layer(x)` is `layer:forward(x)`. Prefer `:forward` in hot loops — method
syntax gets a specialized instruction sequence.

## Module

The base class every layer inherits. Exposed for building custom layers; see
[Extending Nano](../guides/extending.md).

| Member | Description |
| --- | --- |
| `m:forward(x)` · `m(x)` → `Tensor` | Run the layer. |
| `m:parameters()` → `{Tensor}` | Flat list, depth-first, stable order. **Cached** — copy with `table.clone` before mutating. |
| `m:zeroGrad()` → `()` | Zero every parameter's gradient. |
| `m:numParameters()` → `number` | Total scalar parameter count. |
| `m:train()` → `()` | Set `training = true` across the whole tree. |
| `m:eval()` → `()` | Set `training = false` across the whole tree. |
| `m:getFlat()` → `{number}` | Every parameter as one Lua array. The genome interface for evolution. |
| `m:setFlat(values)` → `()` | Load parameters from a Lua array. |
| `m:getFlatBuffer()` → `buffer` | Every parameter as one buffer, block-copied. For target networks and snapshots. |
| `m:setFlatBuffer(values)` → `()` | Load parameters from a buffer. |
| `m:flushParameterCache()` → `()` | Invalidate the cached parameter list. Rarely needed. |
| `m:registerParameter(name, tensor)` → `()` | Declare a learnable tensor on a custom layer. |
| `m:registerModule(name, module)` → `()` | Declare a child module on a custom layer. |
| `nn.Module.init(self)` → `self` | Initialise a table as a Module. Call in a custom layer's constructor. |

The cache is why `table.insert(model:parameters(), extra)` is a bug: it mutates the cached
list, so every later call returns the polluted one.

`getFlat`/`setFlat` do **not** check architecture. Loading a 64-hidden genome into a
48-hidden network produces a model that runs and outputs confident nonsense — use
[`serialize`](serialize.md), whose formats all carry an architecture fingerprint.

## Layers

| Member | Description |
| --- | --- |
| `nn.Linear(inFeatures, outFeatures, bias?, init?)` | Fully connected layer, `x @ W + b`. `bias` defaults to `true`, `init` is `"he"` or Xavier. |
| `nn.LayerNorm(features, eps?)` | Per-sample normalisation with learnable scale and shift. |
| `nn.Dropout(p?)` | Zero a fraction of activations during training only. |
| `nn.Embedding(count, dim)` | Lookup table from 1-based indices to vectors. |
| `nn.Flatten()` | Collapse everything but the batch dimension. |
| `nn.Sequential(layer1, layer2, ...)` | Run layers in order, feeding each output into the next. |

| Member | Description |
| --- | --- |
| `linear.weight: Tensor` · `linear.bias: Tensor?` | The learnable tensors. |
| `linear:scaleWeights(factor)` → `self` | Multiply the weights by `factor` and zero the bias. |
| `sequential.layers: {Module}` | The layers, in order. |
| `nn.dropoutFn(x, p, training)` → `Tensor` | The dropout op, for use outside a module. |

`linear:scaleWeights(0.01)` on a policy head makes the policy start undecided rather than
opinionated — small change, large effect on early PPO.

**LayerNorm, not BatchNorm.** BatchNorm's statistics depend on the other samples in the
batch, so a network behaves differently at batch 1 than at 32 — fatal for RL, where you
act on one observation and train on many. LayerNorm's statistics come from within the
sample.

## Activations

Each is a stateless module wrapping the corresponding tensor or functional op.

| Member | Description |
| --- | --- |
| `nn.ReLU()` | `max(0, x)`. |
| `nn.Tanh()` | Hyperbolic tangent. |
| `nn.Sigmoid()` | Logistic sigmoid. |
| `nn.LeakyReLU(slope?)` | Leaky ReLU, `slope` defaults to `0.01`. |
| `nn.Softmax(dim?)` | Softmax, `dim` defaults to the last dimension. |
| `nn.GELU()` | Gaussian error linear unit. |
| `nn.SiLU()` · `nn.Swish()` | `x · sigmoid(x)`. The same op under both names. |
| `nn.Mish()` | `x · tanh(softplus(x))`. |
| `nn.ELU(alpha?)` | Exponential linear unit, `alpha` defaults to 1. |
| `nn.Softplus(beta?)` | Smooth ReLU, `beta` defaults to 1. |
| `nn.HardSigmoid()` | Piecewise-linear sigmoid. |

## Losses

| Member | Description |
| --- | --- |
| `nn.MSELoss(reduction?)` | Mean squared error. `reduction` is `"mean"` (default) or `"sum"`. |
| `nn.HuberLoss(delta?)` | Quadratic within `delta`, linear beyond. Far less sensitive to outliers than MSE. |
| `nn.CrossEntropyLoss()` | `forward(logits, targets)` where `targets` are 1-based class indices. |
| `nn.BCELoss(reduction?)` | Binary cross-entropy. |
| `nn.L1Loss(reduction?)` | Mean absolute error. |
| `nn.KLDivLoss()` | `forward(logPrediction, logTarget)` — both arguments are log-probabilities. |

`CrossEntropyLoss` and `BCELoss` both take **logits**, not probabilities. Do not put an
`nn.Softmax` or `nn.Sigmoid` in front of them — the softmax is fused into the loss for
numerical stability, and applying it twice silently flattens your gradients.

## Init helpers

| Member | Description |
| --- | --- |
| `nn.xavier(fanIn, fanOut, shape)` → `Tensor` | Xavier/Glorot initialisation. For tanh and sigmoid networks. |
| `nn.he(fanIn, shape)` → `Tensor` | He initialisation. For ReLU-family activations. |

## Compiled inference

| Member | Description |
| --- | --- |
| `nn.compile(model)` → `(({number}) -> {number})?` | Compile a `Sequential` into a weights-only forward for single observations. Returns `nil` if the model contains anything it cannot compile. |
| `nn.noGrad(fn)` → `T` | Re-export of `Config.noGrad`. |

```lua
local fast = nn.compile(model)     -- nil if the model is unsupported
local out = fast(obs)              -- {number} -> {number}
```

Walks a `Sequential` once and returns a plain function holding only weights and two
reusable scratch buffers, ping-ponged between layers. No Tensor objects, no per-call
allocation, no graph. Measured **1.3-1.5x** on a batch-1 forward, which takes 100 NPCs at
60fps from ~4% of the frame budget to ~2.8%.

Supports `Linear` plus `ReLU`, `Tanh`, `Sigmoid`, `LeakyReLU`, `GELU`, `SiLU`/`Swish` and
`Softplus`. Anything else — including `Mish`, `ELU`, `HardSigmoid`, `Dropout`,
`LayerNorm` and `Embedding` — returns `nil`.

Returning `nil` is deliberate. A compiled path that silently skipped the layer it could
not handle would be quietly wrong, and dropping a `Dropout` layer at inference would even
look correct. Check the return value and fall back to `model:forward` when it is `nil`.

Inference only. Training needs the graph. See [Deployment](../guides/deployment.md).
