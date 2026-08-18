# Deploying a trained model

Getting a trained model out of training and into a live experience: saving it, loading it,
and running it fast enough that a hundred NPCs do not cost you the frame.

## Save everything the model needs

A model is not just weights. Anything that transformed the inputs during training must
travel with it, or the deployed model receives data on a different scale and behaves like a
different model.

**Supervised — save the scaler:**

```lua
local state = nano.save(model, {
    scalerMean = scaler.mean,
    scalerStd = scaler.std,
    note = "v3, trained 2026-08-01",
})
```

**RL — use the agent serialiser:**

```lua
local state = nano.algorithms.save(agent)
```

`algorithms.save` round-trips every network plus the observation-normaliser statistics.
`nano.save` on `agent.actor` alone would lose the normaliser, and a policy that normalises
with the wrong statistics is a different policy.

## Persisting to a DataStore

```lua
local text = nano.serialize.toString(model)
DataStore:SetAsync("model_v3", text)

-- later
local loaded = DataStore:GetAsync("model_v3")
local ok = nano.serialize.fromString(model, loaded)
if not ok then
    warn("architecture mismatch — model shape changed since save")
end
```

Every format carries an **architecture fingerprint and refuses to load on mismatch**,
returning `false`. Always check the return value. This is the reason to use `serialize`
over `model:setFlat`, which will happily load a 64-hidden genome into a 48-hidden network
and produce a model that runs and outputs confident nonsense.

For large models, `toBase64` is roughly half the size:

```lua
local packed = nano.serialize.toBase64(model)     -- float32, compact
nano.serialize.fromBase64(model, packed)
```

It is lossy below float32 precision. Fine for a deployed policy, wrong for a checkpoint you
intend to resume exact training from — for that, keep the f64 `toTable` format.

## Rebuild the exact same architecture

Loading needs a model of matching shape to load *into*. Write the architecture once and
call it from both the training script and the runtime:

```lua
-- ReplicatedStorage.Architecture
return function()
    return nano.nn.Sequential(
        nano.nn.Linear(8, 64), nano.nn.ReLU(),
        nano.nn.Linear(64, 64), nano.nn.ReLU(),
        nano.nn.Linear(64, 2)
    )
end
```

Two hand-maintained copies of a layer stack will drift, and the fingerprint check will
catch it — but at load time in production rather than when you made the change.

## Always run inference under noGrad

```lua
local output = nano.noGrad(function()
    return model:forward(obs)
end)
```

When graph construction is off, an operation allocates its output and does nothing else —
no `_prev`, no closures, no gradient buffers. Measured at **3.29x** on a batch-32 forward.
There is no reason to build a graph you will never differentiate.

## Compile for single-observation inference

The usual deployment shape is one NPC, one observation, every frame. That is a batch-1
forward, where per-operation overhead completely dominates the arithmetic.

```lua
local fast = nano.nn.compile(model)

if fast then
    local action = fast(obs)          -- {number} -> {number}
else
    -- unsupported layer somewhere; fall back
    local action = nano.noGrad(function()
        return model:forward(nano.tensor({obs})):toTable()
    end)
end
```

`compile` walks a `Sequential` once and returns a plain function holding only weights and
two scratch buffers, ping-ponged between layers. No Tensor objects, no per-call allocation,
no graph. Measured **1.3-1.5x** over a `noGrad` forward — which takes 100 NPCs at 60fps
from about 4% of the frame budget to 2.8%.

**Check for `nil`.** It supports `Linear` plus `ReLU`, `Tanh`, `Sigmoid`, `LeakyReLU`,
`GELU`, `SiLU`/`Swish` and `Softplus`. Anything else — `Mish`, `ELU`, `HardSigmoid`,
`Dropout`, `LayerNorm`, `Embedding` — returns `nil`.

Returning `nil` rather than skipping the layer is deliberate. A compiled path that silently
dropped a `Dropout` layer would even look correct, and one that dropped a `LayerNorm` would
be quietly, unfixably wrong.

Recompile after loading new weights — the compiled function captured the buffers that
existed at compile time.

## Many NPCs

Two shapes, depending on whether they act on the same tick.

**Batched.** If your NPCs can act together, one batched forward is dramatically cheaper
than N single ones — roughly a thirtieth of the cost at N=32, because per-operation
overhead dominates at these sizes.

```lua
local rows = table.create(#npcs)
for i, npc in ipairs(npcs) do
    rows[i] = npc:observe()
end

local actions = nano.noGrad(function()
    return model:forward(nano.tensor(rows)):toTable()
end)
```

`toTable()` returns a flat Lua array in row-major order, so NPC `i`'s output starts at
index `(i-1) * outputSize + 1`.

**Staggered.** If they act independently, use the compiled function and spread the work
across frames:

```lua
local fast = nano.nn.compile(model)
local cursor = 1

RunService.Heartbeat:Connect(function()
    local budget = os.clock() + 0.002
    repeat
        local npc = npcs[cursor]
        npc:apply(fast(npc:observe()))
        cursor = cursor % #npcs + 1
    until os.clock() > budget
end)
```

A time budget rather than a fixed count, for the same reason the Trainer uses one: the cost
per NPC is not constant, and a fixed count either overruns on a bad frame or wastes the
budget on a good one.

## Act greedily

For RL agents, `act` samples from the policy distribution — that is exploration, and it
belongs in training.

```lua
local action = agent:actGreedy(obs)      -- deployment
```

Also freeze the observation normaliser so live data does not shift the statistics the
policy was trained against:

```lua
if agent.obsNorm then
    agent.obsNorm.frozen = true
end
```

## Server or client

**Native codegen is server-side only.** The `--!native` numeric modules run interpreted on
the client, so identical code is slower there.

| where | when |
|---|---|
| Server | Shared NPCs, anything authoritative, anything exploitable |
| Client | Cosmetic behaviour, per-client effects, offloading a busy server |

Running policies client-side means shipping the weights to clients, so `Nano` must be in
`ReplicatedStorage`. Never put anything authoritative there — the client can read and
modify the weights.

## Budgeting

Rough numbers on an 8-64-64-2 network:

| operation | cost |
|---|---|
| Batch-32 forward, `noGrad` | 0.24 ms |
| Batch-32 full train step | 0.83 ms |
| Batch-1 forward, compiled | ~0.04 ms |

At 60fps you have 16.6 ms per frame. A hundred compiled batch-1 forwards is roughly 4 ms —
substantial but workable if nothing else is competing. Batch them or stagger them if it is
not.

If you train live in a running experience, yield on a time budget:

```lua
local budget = os.clock() + 0.008
if os.clock() > budget then
    task.wait()
    budget = os.clock() + 0.008
end
```

A step that triggers a gradient update costs roughly 100x a plain environment step, so a
fixed step count between yields either stalls during update-heavy stretches or wastes most
of the frame during cheap ones. `train.Trainer` does this for you.

## Checklist

- [ ] Scaler statistics or observation normaliser saved with the weights
- [ ] Load return value checked for architecture mismatch
- [ ] Architecture defined in one shared place, not duplicated
- [ ] All inference wrapped in `noGrad`
- [ ] `nn.compile` used for batch-1, with the `nil` case handled
- [ ] `model:eval()` called if the model contains `Dropout`
- [ ] `actGreedy` for RL policies, normaliser frozen
- [ ] Frame budget measured with the real NPC count

## Next

- [`serialize` reference](../reference/serialize.md) — every format
- [`nn` reference](../reference/nn.md) — `compile` details
- [Performance](../performance.md) — the full measurement set
- [`parallel` reference](../reference/parallel.md) — many independent NPCs across Actors
