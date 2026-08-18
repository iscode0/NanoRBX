# functional

Additional tensor operations, fused kernels and initialisers.

```lua
local F = nano.functional   -- also nano.F
```

Requiring `functional` **installs these onto `Tensor`**. `T.cat` and `F.cat` are the same
function. Existing `Tensor` names are never overwritten, so the core cannot be shadowed.
`init.lua` requires it before `nn`, `rnn` and `rl`, which assume these exist.

## Joining and splitting

| Member | Description |
| --- | --- |
| `F.cat(tensors, dim?)` → `Tensor` | Concatenate along an existing dimension. All other dimensions must agree. |
| `F.stack(tensors, dim?)` → `Tensor` | Stack along a **new** dimension. |
| `F.chunk(t, count, dim?)` → `{Tensor}` | Split into `count` equal parts. This is the LSTM gate split. |
| `F.split(t, sizes, dim?)` → `{Tensor}` | Split into explicitly sized parts, e.g. `{2, 3, 5}`. |

## Indexing and masking

| Member | Description |
| --- | --- |
| `F.gather(t, dim, indices)` → `Tensor` | One element per row, by index. |
| `F.oneHot(indices, classes)` → `Tensor` | One-hot encoding. Constant, not differentiable. |
| `F.maskedFill(t, mask, value)` → `Tensor` | Replace masked positions with a constant. |
| `F.where(cond, a, b)` → `Tensor` | Elementwise select between two tensors. |

`gather` is the discrete-RL workhorse: `Q(s, a_taken)`, `log π(a_taken|s)`. One graph node
regardless of batch size, versus one per row with a `select` loop.

## Statistics

| Member | Description |
| --- | --- |
| `F.var(t, dim?, unbiased?)` → `Tensor` | Variance, two-pass form. |
| `F.std(t, dim?, unbiased?)` → `Tensor` | Standard deviation. |
| `F.norm(t)` → `Tensor` | L2 norm over all elements. |
| `F.layerNorm(x, gamma?, beta?, eps?)` → `Tensor` | Normalise each row to zero mean and unit variance, then scale and shift. One fused node, not nine. |

`var` uses the two-pass form deliberately. The `E[x²] - E[x]²` shortcut can return
negative variance for large means, and the `sqrt` in `std` then gives `nan`.

`layerNorm` is **one fused node, not nine**. The composed chain cost 568 us at `{32,64}`
forward+backward — nine tenths of a whole batch-32 training step — against 88.2 us fused,
a 6.4x gain, and 9.1x on the forward alone. It normalises over the last dimension, so a
`{batch, features}` tensor is normalised per sample.

Its output differs from a hand-composed chain in the **last bit**: it computes
`1/sqrt(var + eps)` once per row and multiplies, where the composition divides per element.
That is one ulp, and it is the reason a division count per row dropped from `features` to
one.

## Activations

Each is one fused op rather than a composition — a composed GELU would be six graph nodes
and six intermediate tensors.

| Member | Description |
| --- | --- |
| `F.softplus(t, beta?)` → `Tensor` | Smooth ReLU, `log(1 + e^(βx)) / β`. |
| `F.gelu(t)` → `Tensor` | Gaussian error linear unit, tanh approximation. |
| `F.silu(t)` → `Tensor` | `x · sigmoid(x)`. Also known as Swish. |
| `F.elu(t, alpha?)` → `Tensor` | Exponential linear unit, `alpha` defaults to 1. |
| `F.mish(t)` → `Tensor` | `x · tanh(softplus(x))`. |
| `F.hardSigmoid(t)` → `Tensor` | Piecewise-linear sigmoid approximation. |
| `F.elementwise(t, code, arg?)` → `Tensor` | The opcode-dispatched kernel the above are built on. Exposed for custom elementwise ops. |

`softplus` is the numerically safe way to produce a positive quantity from an
unconstrained one. Prefer it to `exp` for standard deviations: `exp` overflows, `softplus`
grows linearly.

## Fused kernels

| Member | Description |
| --- | --- |
| `F.linear(x, w, bias?)` → `Tensor` | `y = x @ W + b` in one graph node. Unrolls its three inner loops 4x and skips zero multiplicands. |
| `F.mseLoss(prediction, target, reduction?)` → `Tensor` | Mean squared error. `reduction` is `"mean"` (default) or `"sum"`. |
| `F.huber(prediction, target, delta?)` → `Tensor` | Huber loss, quadratic within `delta` and linear beyond it. |

`nn.Linear`, `nn.MSELoss` and `nn.HuberLoss` use these automatically — you rarely call
them directly. They exist because the composed forms allocate one intermediate tensor and
one graph node per step: Huber composed was seven nodes, and fusing it measured **7.11x**
faster forward+backward.

Skipping zero multiplicands in `F.linear` is worth ~2x on a post-ReLU layer and costs
~1/m when it never fires.

## Linear algebra

| Member | Description |
| --- | --- |
| `F.outer(a, b)` → `Tensor` | Outer product. |
| `F.batchDot(a, b)` → `Tensor` | Row-wise dot product across a batch. |
| `F.cosineSimilarity(a, b, eps?)` → `Tensor` | Row-wise cosine similarity. |

## Initialisation

| Member | Description |
| --- | --- |
| `F.orthogonal(shape, gain?)` → `Tensor` | Orthogonal initialisation, 2D only. |
| `F.constant(shape, value, requiresGrad?)` → `Tensor` | Filled with a constant. |
| `F.uniform(shape, lo, hi, requiresGrad?)` → `Tensor` | Uniform in `[lo, hi)`. |

`orthogonal` is essential for recurrent layers, not a nicety. `W` is applied once per
timestep, so over T steps signal is multiplied by `W^T`; any eigenvalue above 1 explodes
and below 1 vanishes, exponentially in sequence length. Orthogonal puts every eigenvalue
at magnitude exactly 1.
