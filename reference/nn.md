# nn

Layers, containers, transformers, losses and compiled inference.

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
| `m:parameters()` → `{Tensor}` | Flat list, depth-first, **alphabetical by registered name** at each level. **Cached** — copy with `table.clone` before mutating. See [Parameter order](#parameter-order). |
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

### Parameter order

`parameters()` sorts each module's own parameter names, then sorts its submodule names, and
recurses. It is **alphabetical, not registration order**. That is invisible until you need
to line the list up against something outside Nano — an external exporter, a genome layout,
a hand-packed buffer — at which point it is the whole game:

| module | yields, in this order |
| --- | --- |
| `Linear` | `bias`, `weight` |
| `LayerNorm` | `beta`, `gamma` |
| `MultiheadAttention` | `out.bias`, `out.weight`, `qkv.weight` |
| `TransformerBlock` | `attn.*`, `fc1.*`, `fc2.*`, `norm1.*`, `norm2.*` |

Every one of those is the reverse of, or unrelated to, the order the constructor registers
them in. `Sequential` registers its children as `"001"`, `"002"`, … precisely so the
alphabetical sort reproduces the layer order — up to 999 layers.

The failure mode is not a crash. A `Linear` bias and a `LayerNorm` beta of the same width
are the same shape, so a swapped pair loads cleanly and the model is wrong forever. This is
why [`serialize`](serialize.md) fingerprints shapes, and why anything writing weights from
outside must reproduce this order exactly — see
[Training externally](../guides/training-externally.md).

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

`nn.LayerNorm` runs on the fused kernel: 88.2 us forward+backward at `{32,64}`, against
568 us for the composition it replaced.

**LayerNorm, not BatchNorm.** BatchNorm's statistics depend on the other samples in the
batch, so a network behaves differently at batch 1 than at 32 — fatal for RL, where you
act on one observation and train on many. LayerNorm's statistics come from within the
sample.

## Positional encoding

Attention is a weighted sum over positions, and the weights depend only on the *content* of
Q and K. Shuffle the rows of the input and every output row is the same row, shuffled — the
layer cannot tell `cat sat mat` from `mat sat cat`. A causal mask restricts *which*
positions are visible; it says nothing about their order among the visible ones.

So a transformer without positional encoding is a bag of tokens. It trains, the loss falls,
and it has learned something strictly weaker than you think. Nothing errors. These modules
are not optional garnish.

| Member | Description |
| --- | --- |
| `nn.SinusoidalPositions(dim, maxLen)` | Fixed sine/cosine encoding, added to the input. Parameter-free. |
| `nn.LearnedPositions(maxLen, dim)` | One trainable vector per position. **Note the reversed argument order.** |

| Member | Description |
| --- | --- |
| `pos:forward(x, offset?)` → `Tensor` | Add the encoding for positions `offset+1 .. offset+T` to `x`. `x` is `{T, dim}`. |
| `sinusoidal.table: buffer` | The precomputed table. Read-only in practice. |

`SinusoidalPositions` computes its table once at construction, so it costs one add per
element at runtime and nothing at all in gradients. It holds **no parameters**, so it
contributes nothing to `parameters()` — worth knowing when you are matching a parameter
list against an external exporter.

It is the default over a learned table because it needs no training signal to be correct
and it extrapolates past the longest sequence seen in training. `LearnedPositions` does
neither, and cannot represent a position beyond `maxLen` at all, where the sinusoidal form
merely extrapolates imperfectly. At Roblox model sizes there is rarely enough data to learn
a position table well.

**The argument orders are inconsistent** — `SinusoidalPositions(dim, maxLen)` against
`LearnedPositions(maxLen, dim)`. Swapping them does not error at construction; it errors
later, at the first `forward`, complaining about a feature width you never typed.

### offset, and why generation needs it

`offset` defaults to `0`. When you decode one token at a time, pass the number of positions
already generated:

```lua
local x = positions:forward(embed:forward({ id }), n)   -- n = tokens so far
```

Omit it and every generated token is encoded as position 1, so the model sees a constant
where the position signal should be. The output stays fluent and stops tracking order.

## Transformer

| Member | Description |
| --- | --- |
| `nn.MultiheadAttention(dim, heads, causal?)` | Multi-head self-attention. Pass `causal = true` so position `t` cannot see `t+1`. |
| `nn.TransformerBlock(dim, heads, causal?, mult?)` | Pre-norm block: attention, then a GELU MLP, each with a residual. `mult` is the feed-forward expansion, default `4`. |

| Member | Description |
| --- | --- |
| `attn:forward(x)` → `Tensor` | `x` is `{T, dim}`, returns `{T, dim}`. |
| `attn.qkv: Linear` · `attn.out: Linear` | The fused QKV projection and the output projection. |
| `block:forward(x)` → `Tensor` | `x` is `{T, dim}`, returns `{T, dim}`. |
| `block.norm1` · `block.attn` · `block.norm2` · `block.fc1` · `block.fc2` | The pieces, for inspection. |

`dim` must divide evenly by `heads`. That is checked rather than floored: silently rounding
the head dimension changes the model you think you built, and `1/sqrt(D)` with the wrong
`D` is a scale error no gradcheck would notice.

**These take `{T, dim}` — one sequence, no batch dimension**, unlike every other layer in
`nn`. Batched attention would need a 3D kernel. Until then, loop over sequences.

### One projection, not three

`MultiheadAttention` produces Q, K and V from a single `{dim, 3*dim}` weight, sliced
afterwards. Three separate `{dim, dim}` `Linear`s measured **73% of a whole forward**:
three matmuls of width `dim` cost more than one of width `3*dim` — the same arithmetic,
three times the per-call and per-row overhead, and two extra graph nodes. Slicing is free
by comparison, at 1% of the same forward.

Heads are slices of that projection rather than separate parameters, for the same reason.
The QKV projection has **no bias**; the output projection does.

### The block

```
x = x + attn(norm1(x))
x = x + mlp(norm2(x))
```

Pre-norm rather than post-norm, because the residual path stays a clean identity — which is
what lets a stack train without a warmup schedule. Post-norm needs one, and a warmup
schedule is a thing to get wrong.

The MLP expands by `mult` and uses `F.gelu`, the tanh approximation, matching every
reference transformer implementation. `LayerNorm` defaults to `eps = 1e-5`.

```lua
local blocks = table.create(3)
for i = 1, 3 do
    blocks[i] = nn.TransformerBlock.new(128, 4, true)   -- dim, heads, causal
end
```

## KV caching

| Member | Description |
| --- | --- |
| `attn:newCache(maxLen)` · `block:newCache(maxLen)` → `table` | Allocate a cache for up to `maxLen` positions. |
| `attn:resetCache(cache)` · `block:resetCache(cache)` → `()` | Forget every stored position, keeping the allocation. |
| `attn:decode(x, cache)` · `block:decode(x, cache)` → `Tensor` | Feed one position, `{1, dim}`. Returns `{1, dim}`. |
| `attn:compileDecode(maxLen)` → `(step, reset, length)` | Specialised buffer-only single-token decode. |

Generating `T` tokens without a cache re-runs attention over the entire prefix at every
step, so the whole generation is **O(T³)**. Measured: 8 tokens 2.9 ms, 16 tokens 11.0 ms,
32 tokens 43.7 ms — 4x per doubling, exactly cubic. Long replies get visibly slower as they
go.

A cache stores K and V for every position seen so far, so a step projects only the new
token and attends over the stored prefix: **O(T) per token, O(T²) overall**. On the
attention layer alone, 64 tokens went from 182 ms to **3.1 ms**.

`TransformerBlock:decode` is bit-identical to `forward` over the same sequence — verified as
equality in the test suite, not as a tolerance.

**One cache belongs to one sequence.** Two NPCs generating at once need two caches. Sharing
one interleaves their contexts, which reads as each NPC finishing the other's sentences.

**Inference only, and it errors otherwise.** `decode` raises if the graph is on. Decoding
with gradients enabled would retain every step's graph through a buffer that later steps
overwrite in place, so the backward would differentiate against values that no longer
exist — wrong numbers, no error. Wrap generation in `nano.noGrad`.

```lua
local caches = table.create(#blocks)
for i, b in blocks do
    caches[i] = b:newCache(256)
end

nano.noGrad(function()
    local n = 0
    local function step(id: number)
        local x = positions:forward(embed:forward({ id }), n)
        for i, b in blocks do
            x = b:decode(x, caches[i])
        end
        n += 1
        return head:forward(x)             -- {1, vocab}
    end
    -- ...
end)
```

See [Transformers](../guides/transformers.md) for the full generation loop.

### compileDecode

`attn:compileDecode(maxLen)` is to `decode` what `nn.compile` is to `forward`: everything
fixed at compile time is hoisted, the scratch is allocated once, and the step works in raw
buffers.

One cached token at `dim = 64` is 20,480 multiply-adds, which at the library's 1.26 ns/MAC
should take 25.9 us. `decode` measured **42.7 us** — 39% of it per-call cost on a call doing
very little arithmetic: graph checks, a Tensor per intermediate, four narrows that each
allocate and copy, and an `F.cat` to rejoin the heads. `compileDecode` skips all of it; the
head slices are offsets into one buffer, and each head writes straight into its slice of the
concat scratch, because concatenating along the feature axis *is* writing at an offset.

```lua
local step, reset, length = attn:compileDecode(256)

nano.noGrad(function()
    local out = step(features)      -- {number} of dim -> {number} of dim
    print(length())                 -- positions cached so far
    reset()
end)
```

It takes and returns plain Lua arrays, builds no graph, and holds its own cache internally —
there is no `cache` argument. Inference only, for the same reason `decode` is. It covers
attention only, not a whole block; a stacked model still runs its norms, residuals and MLP
through the ordinary path.

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
reusable scratch buffers. Everything fixed at compile time — the layer kind, the weight
buffers, the widths, and which scratch buffer each step reads and writes — is resolved
into parallel arrays, so the forward has no hash lookups, no string compares and no buffer
swapping. The inner loop is 4x unrolled with a zero-skip, matching `F.linear`.

No Tensor objects, no graph, and **zero allocation per call** (measured at 0.00 kb/op).

Measured against a `noGrad` forward at batch 1:

| network | forward | compile | gain |
| --- | --- | --- | --- |
| 8-32-32-2 | 3.7 us | **1.46 us** | ~2.5x |
| 8-64-64-2 | 6.4 us | **4.24 us** | ~1.5x |
| 8-128-128-2 | 15.6 us | **11.3 us** | ~1.4x |

Two independent runs agreed to within 4%. The compiled column is the steadier of the two
(4-6% CV against 12-18% for the forward), so the uncertainty in the ratio sits almost
entirely in the denominator — round these rather than quoting them.

The gain shrinks as the network widens, which is the shape you should expect: what
`compile` removes is per-operation overhead, and that is a fixed cost. Once the arithmetic
dominates, both paths run the same unrolled loop. It pays most where it is most often
used — many small networks.

Supports `Linear` plus `ReLU`, `Tanh`, `Sigmoid`, `LeakyReLU`, `GELU`, `SiLU`/`Swish` and
`Softplus`. Anything else — including `Mish`, `ELU`, `HardSigmoid`, `Dropout`,
`LayerNorm` and `Embedding` — returns `nil`.

Returning `nil` is deliberate. A compiled path that silently skipped the layer it could
not handle would be quietly wrong, and dropping a `Dropout` layer at inference would even
look correct. Check the return value and fall back to `model:forward` when it is `nil`.

Inference only. Training needs the graph. See [Deployment](../guides/deployment.md).
