# serialize

Saving and loading weights and optimizer state.

```lua
local serialize = nano.serialize
```

## Functions

| Member | Description |
| --- | --- |
| `serialize.toTable(model, metadata?)` → `table` | Full-precision plain table, with optional metadata attached. Also `nano.save`. |
| `serialize.fromTable(model, state)` → `boolean` | Load from a table. Also `nano.load`. **Errors** on architecture mismatch rather than returning `false` — wrap in `pcall` if a mismatch is expected. |
| `serialize.toString(model, metadata?)` → `string` | JSON-encoded string, for DataStores. |
| `serialize.fromString(model, encoded)` → `boolean` | Load from a string. |
| `serialize.toBase64(model)` → `{shapes: {{number}}, data: string}` | Compact float32 encoding. |
| `serialize.fromBase64(model, payload)` → `boolean` | Load from a base64 payload. |
| `serialize.fromQuantized(model, payload)` → `boolean` | Load int8 weights with per-row scales. Load-only; there is no `toQuantized`. |
| `serialize.optimizerToTable(opt)` → `table` | Capture optimizer moments and step count. |
| `serialize.optimizerFromTable(opt, state)` → `boolean` | Restore optimizer state. |

```lua
local state = nano.save(model, { note = "gen 400" })
nano.load(model, state)

local text = serialize.toString(model)
serialize.fromString(model, text)

local packed = serialize.toBase64(model)
serialize.fromBase64(model, packed)
```

## Choosing a format

| format | precision | use for |
|---|---|---|
| `toTable` / `fromTable` | f64, exact | Resuming training, in-memory snapshots |
| `toString` / `fromString` | f64, exact | DataStore persistence |
| `toBase64` / `fromBase64` | float32, lossy | Deployment, weight transport between Actors |
| `fromQuantized` | int8 + per-row f32 scales, lossy | Shipping a large externally trained model inside a ModuleScript |

Base64 is roughly half the size and loses precision below float32. That is fine for a
deployed policy and wrong for a checkpoint you intend to resume exact training from.

## Quantized transport

Weights ship as text inside a ModuleScript, and f32 base64 costs **5.33 bytes per
parameter**. A 600k-parameter model is 3.3 MB of source, already near what Roblox will
comfortably hold. int8 with per-row scales costs **1.33 bytes per parameter** — 4x smaller,
so 4x the model fits in the same script.

```lua
serialize.fromQuantized(model, {
    shapes = W.shapes,      -- {{number}}, parameter order
    scales = W.scales,      -- base64 f32, one per row, flattened
    data   = W.data,        -- base64 int8, one byte per weight
})
```

**This is transport only.** Values are dequantized into the same f64 buffers `fromBase64`
fills, so inference afterwards is bit-for-bit the normal path: same speed, same arithmetic,
no quantized kernels anywhere. The only cost is a one-time rounding at export.

Scales are **per row**, not per tensor. A single outlier row would otherwise set the scale
for the whole matrix and quantize every other row into a handful of levels. Per row,
`w ≈ scale * q` with `scale = max|row| / 127` for that row. Rows are the leading dimension;
a 1D tensor is one row.

There is no `toQuantized`. The format exists to receive weights from an external trainer —
see [Training externally](../guides/training-externally.md), which ships a PyTorch exporter
that writes exactly this payload. Quantizing weights that are already inside Roblox would
cost precision and buy nothing, since they are already f64 in memory.

`fromQuantized` checks the same architecture fingerprint as every other loader and errors on
mismatch.

## Every format checks architecture

Every one of them carries an architecture fingerprint and **refuse to load on mismatch**. The refusal
is an **error**, not a `false` return, and it names the offending parameter and both sizes.
Wrap the load in `pcall` if a mismatch is possible — for example when loading a model whose
architecture may have changed since it was saved.

This is the reason to use `serialize` over `model:setFlat`. Plain `getFlat`/`setFlat` will
happily load a 64-hidden genome into a 48-hidden network and produce a model that runs and
outputs confident nonsense — no error, no warning, just wrong answers forever.

## Save the optimizer too

For any run you intend to resume:

```lua
local checkpoint = {
    model = nano.save(model),
    opt   = serialize.optimizerToTable(opt),
}
```

Adam's `m` and `v` are built up over thousands of steps. Resuming without them leaves the
first updates effectively unscaled, which can undo a substantial amount of training in a
handful of steps.

## Agents

For an RL agent, use [`algorithms.save` / `algorithms.load`](algorithms.md) instead. They
round-trip every network plus the observation-normaliser state — a policy restored without
its normaliser sees inputs on a different scale than it trained on, and behaves like a
different policy.

If you use a `data.Scaler` for supervised inputs, save its statistics alongside the model
for the same reason. See [Deployment](../guides/deployment.md).
