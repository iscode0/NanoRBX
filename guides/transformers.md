# Transformers

Building, training and generating from a transformer in Nano.

This page assumes you have read [Getting started](../getting-started.md) and
[Tensors and autograd](../concepts/autograd.md). The reference pages for the pieces are
[nn](../reference/nn.md#transformer) and [functional](../reference/functional.md#fattention).

---

## What a transformer is, in one paragraph

A transformer reads a whole sequence at once. Each position produces a query, a key and a
value; each position's output is a weighted sum of every value, weighted by how well its
query matches each key. That is attention. Stack a few of those with a small MLP between
them, add a positional signal so the layer can tell order from content, and you have a model
that predicts the next token from all the previous ones.

In Nano that is four objects: an embedding, a positional encoding, some blocks, and a linear
head.

---

## The model

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
local nn = nano.nn

local VOCAB   = 96      -- distinct characters
local DIM     = 128     -- model width
local HEADS   = 4       -- DIM must divide evenly by this
local BLOCKS  = 3
local WINDOW  = 64      -- training context length

local embed     = nn.Embedding.new(VOCAB, DIM)
local positions = nn.SinusoidalPositions.new(DIM, WINDOW + 256)
local head      = nn.Linear.new(DIM, VOCAB)

local blocks = table.create(BLOCKS)
for i = 1, BLOCKS do
    blocks[i] = nn.TransformerBlock.new(DIM, HEADS, true)   -- causal
end
```

`causal = true` is what makes this a language model rather than an encoder: position `t`
cannot see `t+1`, so predicting the next token is not trivially solved by reading it.

`positions` is built with headroom past `WINDOW` so the same model can generate sequences
longer than it trained on. It holds no parameters, so it costs nothing to oversize.

### Collecting the parameters

The pieces are separate objects, so gather them once:

```lua
local params = {}
for _, p in embed:parameters() do table.insert(params, p) end
for _, b in blocks do
    for _, p in b:parameters() do table.insert(params, p) end
end
for _, p in head:parameters() do table.insert(params, p) end

local opt = nano.optim.AdamW.new(params, 3e-4)
```

Order matters if you ever load weights from outside Nano. See
[Parameter order](../reference/nn.md#parameter-order).

### Forward

```lua
local function forward(ids: {number}): Tensor
    local x = positions:forward(embed:forward(ids))
    for _, b in blocks do
        x = b:forward(x)
    end
    return head:forward(x)               -- {T, VOCAB} logits
end
```

`ids` are **1-based** — `nn.Embedding` indexes from 1, not 0. A vocabulary built with
0-based ids reads the wrong row for every character, and nothing errors.

---

## Training

One sequence at a time: attention takes `{T, dim}` with no batch dimension.

```lua
local criterion = nn.CrossEntropyLoss.new()

for step = 1, 4000 do
    local start   = math.random(1, #corpus - WINDOW - 1)
    local inputs  = table.create(WINDOW)
    local targets = table.create(WINDOW)
    for i = 1, WINDOW do
        inputs[i]  = corpus[start + i - 1]
        targets[i] = corpus[start + i]       -- shifted by one
    end

    opt:zeroGrad()
    local loss = criterion:forward(forward(inputs), targets)
    loss:backward()
    opt:clipGradNorm(1.0)
    opt:step()
end
```

`CrossEntropyLoss` takes **logits**, so there is no softmax on `head`. Targets are 1-based
class indices, which lines up with the embedding.

Clip the gradient norm. Transformers produce occasional large gradients early in training,
and one unclipped step can undo a great deal of progress.

### This will be slow

Training a transformer inside Roblox is possible and rarely what you want. A `{64, 128}`
sequence through three blocks is a few million multiply-adds per step, and you need
thousands of steps. Budget for minutes-to-hours, run it on the server, and yield — see
[Performance](../performance.md#yielding).

If you have a GPU, [train externally](training-externally.md) and ship the weights in. That
path takes minutes and produces a strictly better model.

---

## Generation

The naive loop re-runs the whole forward over the whole prefix for every character:

```lua
-- DON'T: O(T³) over the whole reply
for i = 1, 200 do
    local logits = forward(generated)
    -- ...
end
```

Attention is O(T²) in context length, so the last character of a reply costs hundreds of
times the first. Measured on the attention layer alone: 8 tokens 2.9 ms, 16 tokens 11.0 ms,
32 tokens 43.7 ms. Exactly cubic, and very visible.

### With a KV cache

Each block keeps the keys and values it has already computed, so a step touches one position
instead of the whole prefix. O(T) per token, O(T²) overall. On the attention layer, 64
tokens went from 182 ms to **3.1 ms**.

```lua
local MAXLEN = 256

local caches = table.create(BLOCKS)
for i, b in blocks do
    caches[i] = b:newCache(MAXLEN)
end

local function generate(prompt: {number}, count: number, temperature: number): {number}
    return nano.noGrad(function()
        for i, b in blocks do
            b:resetCache(caches[i])
        end

        local n = 0
        local out = {}

        local function step(id: number): Tensor
            local x = positions:forward(embed:forward({ id }), n)
            for i, b in blocks do
                x = b:decode(x, caches[i])
            end
            n += 1
            return head:forward(x)              -- {1, VOCAB}
        end

        -- prime on the prompt, keeping only the last set of logits
        local logits
        for _, id in prompt do
            logits = step(id)
        end

        for _ = 1, count do
            -- temperature scales the logits; Categorical takes logits, not probabilities
            local dist = nano.rl.Categorical.new(nano.Tensor.mul(logits, 1 / temperature))
            local id = dist:sample()[1]        -- 1-based class index, one row
            table.insert(out, id)
            logits = step(id)
        end

        return out
    end)
end
```

Three things in there are load-bearing:

**`nano.noGrad`.** `decode` errors if the graph is on, rather than silently producing wrong
gradients. The cache is overwritten in place, so a graph built across steps would
differentiate against values that no longer exist.

**The `n` offset.** `positions:forward(x, n)` encodes the token as position `n+1`. Drop the
offset and every generated token is encoded as position 1 — the model sees a constant where
the order signal should be, and the output stays fluent while quietly losing track of
position.

**`resetCache` at the start.** A cache carries the previous generation's context. Reusing
one without resetting reads as the model finishing someone else's sentence.

### One cache per sequence

A cache belongs to one sequence. Two NPCs generating simultaneously need two sets:

```lua
local function newCaches()
    local c = table.create(BLOCKS)
    for i, b in blocks do
        c[i] = b:newCache(MAXLEN)
    end
    return c
end

local perNPC = {}
for _, npc in npcs do
    perNPC[npc] = newCaches()
end
```

Caches are buffers of `maxLen * dim * 8` bytes per block, twice over for K and V. At
`dim = 128` and `maxLen = 256` that is 512 KB per NPC across three blocks — real memory.
Size `maxLen` to the longest reply you actually want, and `resetCache` between replies
rather than allocating fresh ones.

`decode` errors when the cache is full rather than overwriting the oldest position. Sliding
a context window is a decision about what your model forgets, so it is left to you.

---

## Checklist

- [ ] `DIM` divides evenly by `HEADS`
- [ ] `causal = true` on every block, for a language model
- [ ] A positional encoding is actually applied — no error tells you if it is missing
- [ ] Token ids are **1-based**
- [ ] No softmax before `CrossEntropyLoss`
- [ ] Gradient norm clipped
- [ ] Generation wrapped in `nano.noGrad`
- [ ] `positions:forward(x, n)` passes the offset during decode
- [ ] `resetCache` between sequences, one cache per sequence
- [ ] `maxLen` big enough for the longest reply

---

## Next

- [Training externally](training-externally.md) — train on a GPU, ship the weights in
- [Deployment](deployment.md) — saving, loading, running many NPCs
- [nn reference](../reference/nn.md#transformer) — every member
- [Gotchas](../gotchas.md#transformers) — the silent failures
