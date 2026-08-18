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

Base64 is roughly half the size and loses precision below float32. That is fine for a
deployed policy and wrong for a checkpoint you intend to resume exact training from.

## Every format checks architecture

All three carry an architecture fingerprint and **refuse to load on mismatch**. The refusal
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
