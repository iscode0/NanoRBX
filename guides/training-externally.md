# Training externally

Train on a GPU in PyTorch, ship the weights into Roblox.

---

## When to do this

Nano trains inside Roblox, which is the point of it. For a policy network or a small
classifier that is the right call: the whole loop lives in one place, there is nothing to
export, and training takes seconds.

Transformers are the case where it stops being the right call. A character-level model worth
listening to is a few hundred thousand parameters trained for thousands of steps over a real
corpus. In Luau that is hours. On a GPU it is minutes, and the GPU model is better because
you could afford to train it properly.

Nothing about Nano changes. You are replacing the training loop, not the model — the weights
land in the same `TransformerBlock`s, and inference runs on the same path it always did.

| you have | do this |
| --- | --- |
| A small net and a few thousand steps | Train in Nano. Stop reading. |
| A transformer, a corpus, and a GPU | Train in PyTorch, export, load |
| A transformer and no GPU | Train in Nano overnight, or shrink the model |

---

## The pipeline

```
corpus.txt ──▶ train_nano_lm.py ──▶ NanoWeights.lua ──▶ ReplicatedStorage
                    (GPU)            (ModuleScript)      serialize.from*
```

`train_nano_lm.py` is a single self-contained script. It needs only `torch`:

```bash
pip install torch
python train_nano_lm.py --corpus mytext.txt --steps 4000 --dim 128 --amp
```

It writes `NanoWeights.lua`, a ModuleScript you paste into Studio next to `Nano`. That
module carries the weights, the exact vocabulary, and a `build` function that reconstructs
the matching architecture.

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
local W    = require(script.Parent.NanoWeights)

local model = W.build(nano)     -- same architecture the weights were trained for
W.load(nano, model)             -- picks fromBase64 or fromQuantized

local ids = W.encode("hello")   -- string -> 1-based ids
local char = W.decode(id)       -- id -> string
```

`W.build` returns a holder with `embed`, `positions`, `blocks`, `head`, a `parameters()`
that yields them in export order, and a `forward`. Generation from there is the loop in
[Transformers](transformers.md#with-a-kv-cache).

### GPU flags worth knowing

| flag | effect |
| --- | --- |
| `--amp` | bf16 on Ampere and newer, fp16 otherwise. Roughly 2x. |
| `--compile` | `torch.compile`; slow first step, faster after. |
| `--batch` | Sequences per step. Raise until reported peak memory is most of the card. |
| `--quantize int8` | 4x smaller ModuleScript. See below. |

The batching matters more than any flag. Each step stacks `--batch` windows into one
`(B, T)` tensor and does a single forward and backward; looping over the batch one sequence
at a time leaves a GPU almost entirely idle, because each kernel is tiny and launch overhead
dominates.

---

## Size, and the ceiling

f32 base64 costs **5.33 bytes per parameter**. Roblox will not comfortably hold an
arbitrarily large script, and past roughly 4 MB you are gambling.

`--quantize int8` stores each weight as one byte plus one f32 scale per row: **1.33 bytes
per parameter**, 4x smaller, so 4x the model fits in the same script.

```bash
python train_nano_lm.py --corpus mytext.txt --quantize int8
```

`W.load` reads `W.quantized` and dispatches to
[`serialize.fromQuantized`](../reference/serialize.md#quantized-transport) on its own.

**Quantization is transport only.** Nano dequantizes into the same f64 buffers `fromBase64`
fills, so inference afterwards is bit-for-bit the normal path — same speed, same arithmetic,
no quantized kernels. The only cost is a one-time rounding at export, and the script prints
the worst weight error it introduced.

If you are still over the limit after quantizing, reduce `--dim` or `--blocks`, or split the
base64 chunks across several ModuleScripts and concatenate at load.

---

## The export contract

This is the part that repays reading before you debug it. A mismatch here does not error in
Studio — the payload loads cleanly and the model is subtly wrong forever.

`serialize.fromBase64` and `fromQuantized` both validate every shape against the model, so
a *wrong architecture* fails loudly. What they cannot catch is a correct set of shapes in
the wrong order, or the right numbers transposed, or an off-by-one in your token ids.

### Parameter order is alphabetical

`model:parameters()` sorts each module's own parameter names, then sorts its submodule
names, and recurses. It is **not** registration order:

| module | yields |
| --- | --- |
| `Linear` | `bias`, `weight` |
| `LayerNorm` | `beta`, `gamma` |
| `MultiheadAttention` | `out.bias`, `out.weight`, `qkv.weight` |
| `TransformerBlock` | `attn.*`, `fc1.*`, `fc2.*`, `norm1.*`, `norm2.*` |

The exporter reproduces this exactly. Its first draft assumed registration order and
produced twelve shape mismatches — which is the *good* outcome. The bad one is when the
shapes happen to collide and a bias loads into a gamma.

### Five places PyTorch and Nano differ

| # | PyTorch | Nano | what the exporter does |
| --- | --- | --- | --- |
| 1 | `Linear` stores `{out, in}`, computes `x @ Wᵀ` | stores `{in, out}`, computes `x @ W` | transposes every weight |
| 2 | Q/K/V usually three matrices | one fused `{dim, 3*dim}`, no bias, ordered Q, K, V | fuses; output projection keeps its bias |
| 3 | `F.gelu` defaults to the exact erf form | tanh approximation | passes `approximate="tanh"` |
| 4 | — | pre-norm, 4x MLP, LayerNorm `eps = 1e-5` | matches the block exactly |
| 5 | positional table often half-split | **interleaved**: even sin, odd cos, shared frequency per pair | replicates it for training, exports nothing |

`SinusoidalPositions` holds no parameters, so it never appears in the payload. It is
reproduced in the PyTorch model only so training sees the signal inference will.

Number 3 is the one to watch. The erf and tanh GELUs differ by a small amount at every
activation, which is invisible in any single value and compounds through a stack.

### Token ids are 1-based

`nn.Embedding` indexes from 1. Python vocabularies index from 0. Feed a 0-based id straight
in and every character reads the wrong row — shapes all match, nothing errors, and the model
emits fluent nonsense.

`W.chars` is written in Nano's 1-based order and `W.encode` / `W.decode` are the only
correct way to cross that boundary. Use them rather than indexing a vocabulary yourself.

This also lines up with the rest of the library: `rl.Categorical:sample()` returns 1-based
class indices, so a sampled id feeds straight back into `embed:forward`.

### Validation

A full forward pass through the PyTorch model and through Nano loaded from the exported
weights agree to **2.4e-07**, which is f32 transport precision. If you change either side,
reproduce that number before trusting anything downstream. A model that is merely
*plausible* is the failure mode this whole page exists to prevent.

---

## Checklist

- [ ] Architecture in `W.build` matches what you trained
- [ ] `W.load` succeeded — it errors on shape mismatch, so do not swallow it in a `pcall`
- [ ] Ids go through `W.encode` / `W.decode`, never a hand-built table
- [ ] Generation runs under `nano.noGrad`
- [ ] Script size is under ~4 MB; `--quantize int8` if not
- [ ] A known prompt produces the same output it did in Python

---

## Next

- [Transformers](transformers.md) — the generation loop the weights plug into
- [serialize](../reference/serialize.md#quantized-transport) — the payload formats
- [Parameter order](../reference/nn.md#parameter-order) — the ordering rule in full
- [Gotchas](../gotchas.md#transformers) — the silent failures
