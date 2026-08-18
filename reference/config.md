# Config

Global switches and the shared random source. Dependency-free, so every other module can
require it without a cycle.

```lua
local Config = nano.Config
```

Everything used to call `math.random` directly, which meant no run could be reproduced: a
change that looked like an improvement might just have been a lucky seed, and a bug that
appeared once might never appear again. One shared, seedable source fixes both.

## Properties

| Member | Description |
| --- | --- |
| `Config.gradEnabled: boolean` | When `false`, operations skip building the autograd graph entirely — no `_prev`, no `_backward` closure, no gradient buffers. Defaults to `true`. Prefer `noGrad` over setting this by hand. |

## Functions

| Member | Description |
| --- | --- |
| `Config.isGradEnabled()` → `boolean` | Whether graph construction is currently on. |
| `Config.noGrad(fn)` → `T` | Run `fn` with the graph disabled and return its result. |
| `Config.seed(value?)` → `()` | Seed every random source in Nano. Pass nothing to reseed unpredictably. |
| `Config.getSeed()` → `number?` | The seed currently in force, or `nil` if unseeded. |
| `Config.isSeeded()` → `boolean` | Whether an explicit seed is in force. |
| `Config.random()` → `number` | Uniform in `[0, 1)`. |
| `Config.uniform(lo, hi)` → `number` | Uniform in `[lo, hi)`. |
| `Config.randomInt(lo, hi?)` → `number` | Uniform integer in `[lo, hi]`, or `[1, lo]` when `hi` is omitted. |
| `Config.gaussian()` → `number` | One standard normal, via Box-Muller. |
| `Config.stream(seed)` → `Random` | A private `Random` for anything needing its own reproducible stream without disturbing the global one — an environment, a dataset split. |

## Notes

**`noGrad` restores the previous state, not `true`.** Nested `noGrad` calls are common
once helper functions use it internally, so setting `true` on exit would re-enable
gradients inside an outer `noGrad` block. It also restores on error, so a throw inside
`fn` cannot leave gradients globally disabled for the rest of the session.

**`gaussian` discards the second Box-Muller value rather than caching it.** Caching would
halve the trig cost but makes the stream stateful in a way that breaks reproducibility
across differently-shaped calls: the same seed would give different numbers depending on
how many samples were drawn before. Determinism is worth more than the saved cosine.

**Every random source in the library routes through here.** `Dataset:split` and
`data.Loader:iter()` included — they used to call `Random.new` and `math.random` directly,
which meant a seeded run reproduced its weights but not the order they were trained on.

Worker Actors are covered too: `parallel.Pool.new` forwards the seed across the actor
boundary, since each Actor loads its own copy of this module with its own unseeded `Random`.

`Config.stream(seed)` is the deliberate exception: `Dataset:split(fraction, seed)` uses it
so a split stays identical even if the amount of global randomness drawn before it
changes.
