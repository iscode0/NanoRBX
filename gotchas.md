# Gotchas

Every mistake found so far that produces wrong results without an error. Nearly all of
these train a subtly wrong model, report plausible numbers, and cost you a week.

## Shapes and storage

**Shapes are `{batch, features}`.** A single sample is `T.new({{0.3, 0.7}})`. Double
braces. `T.new({0.3, 0.7})` is a `{2}` vector, and `matmul` requires two 2D tensors.

**`t.data` is a buffer.** `t.data[i]` is not an error, it is `nil`. Use `t:at(i)`,
`t:setAt(i, v)`, `t:toTable()`, or `buffer.readf64(t.data, (i-1)*8)`.

**`toFlat()` returns a buffer, `toTable()` a Lua array.** Passing the former where an array
is expected fails silently in any loop that uses `#`. Indexing it raises "attempt to index
buffer with number" — `action[1]` on a `toFlat()` result. The failure is easy to miss for a
long time, because the loop that swallows it does not complain.

## Autograd

**`opt:zeroGrad()` first, every time.** Gradients accumulate; forgetting produces no error,
just inexplicable loss behaviour.

**`backward()` requires a scalar.** Reduce with `mean` or `sum`.

**`zeroGrad` leaves stale values in `.grad` until `backward` runs.** It marks buffers for
overwrite rather than zeroing them. Use `hardZeroGrad()` if you inspect gradients between
the two.

**`parameters()` is cached.** `table.insert(model:parameters(), extra)` pollutes the cache
permanently. `table.clone` first.

**Inputs are not copied.** Backward closures read source arrays directly — the same tradeoff
PyTorch makes. Mutating an input between forward and backward gives wrong gradients
silently.

**A short gradient array is caught, not silently applied.** `accumulate` errors if a
backward returns the wrong number of entries. This is the failure mode of a hand-written
backward, and it was previously invisible.

**`gradcheck` requires `fn` to be deterministic.** It calls `fn` three times per element.
Build models outside the closure, and do not gradcheck through dropout or unseeded
sampling — otherwise the finite difference compares two different networks.

## Losses and activations

**`CrossEntropyLoss` and `BCELoss` take logits.** No softmax or sigmoid before them. The
transform is fused into the loss for stability, and applying it twice flattens your
gradients.

**IEEE edges do not throw.** `exp(1000)` is `inf`, `log(0)` is `-inf`, `log(-1)` and
`sqrt(-1)` are `nan`, `x/0` is `inf`. A single `nan` entering a layer comes out of every
output and keeps going. Use `safeExp` where an exponent can spike, and
`train.diagnose(model)` to find non-finite parameters after the fact.

**Softmax `dim` defaults to the last dimension.** A softmax normalising over the whole
tensor makes every row depend on every other — silently wrong, hard to see.

**Clamp `logStd` inside the graph, not just after the step.** Clamping only in `.data`
afterwards means a bad update can drive it far negative mid-update, `std` collapses, `exp`
overflows to `inf`, and every parameter becomes `nan` with no error.

**Forgetting `model:eval()`** on a model with dropout means you evaluate a randomly
crippled network. The score is bad and nothing explains why.

## Data handling

**Fit the scaler on training data only.** Fitting on everything leaks validation statistics
into training and makes the score optimistic by an unknown amount.

**Save the scaler with the model.** A deployed model without the scaler it trained with
receives inputs on a different scale and produces confident nonsense.

**Use `serialize`, not `setFlat`, to load weights.** Every `serialize` format carries an
architecture fingerprint and refuses to load on mismatch. `setFlat` will happily load a
64-hidden genome into a 48-hidden network and produce a model that runs and outputs
confident nonsense.

## Reinforcement learning

**`terminated` and `truncated` are different things.** A time limit is not the end of the
world; the next state still has value. Collapsing them teaches the critic that value is
zero at every episode boundary, and looks exactly like the algorithm being broken.

**GAE needs both `dones` and `bootstraps`.** A cut without a bootstrap is the most common
silent bug in PPO implementations.

**Never clamp an action before scoring it.** Scoring a clamped action against an unclamped
Gaussian density makes every stored log-prob wrong. Bound the action in the environment.

**`horizon` must be smaller than your total step count.** Otherwise the rollout buffer never
fills, no update ever runs, and the loop executes happily forever. Check
`summary.updates` — the Trainer warns, loudly.

**Give your env a `seed(n)` method.** Without it, evaluation is not reproducible and
`keepBest` tracks the luckiest evaluation rather than the best policy.

**Pass `evalEnv` to the Trainer.** Evaluating on the training env resets it mid-rollout and
corrupts every transition that follows.

**Save the observation normaliser with the policy.** Use `algorithms.save`, not
`nano.save` on the actor. A policy restored without its normaliser sees inputs on a
different scale than it trained on.

**Hiding a variable does not make a memory task.** It must not be *recoverable* from
anything the agent can observe or influence. An earlier version of the recurrent test hid
the target but left the agent's position visible — so a memoryless policy encoded the
target into its own position on step 1 and read it back forever after, using the world as
memory.

## Transformers

**A transformer without positional encoding is a bag of tokens.** Attention weights depend
only on the content of Q and K, so shuffling the input rows shuffles the output rows and
changes nothing else. The model cannot tell `cat sat mat` from `mat sat cat`. It trains, the
loss falls, and it has learned something strictly weaker than you think. A causal mask does
not substitute — it restricts which positions are visible, not their order.

**`positions:forward(x, offset)` needs the offset during generation.** Omit it and every
generated token is encoded as position 1, so the model sees a constant where the order
signal should be. Pass the number of tokens generated so far.

**`SinusoidalPositions(dim, maxLen)` and `LearnedPositions(maxLen, dim)` take their
arguments in opposite orders.** Both are two numbers, so a swap constructs fine and errors
later, at the first `forward`, about a feature width you never typed.

**Attention takes `{T, dim}`, with no batch dimension** — unlike every other layer in `nn`.
Loop over sequences.

**One KV cache belongs to one sequence.** Two NPCs sharing a cache interleave their
contexts, which reads as each finishing the other's sentence. `resetCache` between
sequences; a cache carries the previous generation's context.

**`decode` errors if the graph is on, and that is the feature.** The cache is overwritten in
place, so a graph built across decode steps would differentiate against values that no
longer exist. Wrap generation in `nano.noGrad`.

**`compileDecode` covers attention only, not a whole block.** It holds its own cache
internally and takes no cache argument, so mixing it with `block:newCache` gives you two
independent caches that disagree about how many positions exist.

**Embedding ids are 1-based.** Nano indexes from 1; most tooling outside it indexes from 0.
An off-by-one reads the wrong row for every token — no error, and output fluent enough to
look like a training problem rather than an indexing one.

**`dim` must divide evenly by `heads`.** This one *is* checked, and deliberately: silently
flooring the head dimension changes the model you think you built, and `1/sqrt(D)` with the
wrong `D` is a scale error no gradcheck would notice.

## Loading external weights

**`parameters()` is alphabetical, not registration order.** `Linear` yields `bias` then
`weight`; `LayerNorm` yields `beta` then `gamma`; `MultiheadAttention` yields `out` before
`qkv`. Anything packing weights from outside must reproduce that order. Shapes are
fingerprinted, so a wrong architecture errors — but a bias and a beta of the same width are
the same shape, and a swapped pair loads clean.

**Nano's `Linear` stores `{in, out}`, PyTorch stores `{out, in}`.** Every weight needs
transposing on the way in. A square layer makes this silent.

**`F.gelu` is the tanh approximation.** PyTorch's default is the exact erf form. The
difference is small at every activation and compounds through a stack, and it looks exactly
like an undertrained model.

**Quantized loading is lossy at export, not at inference.** `fromQuantized` dequantizes into
the same f64 buffers as every other loader, so nothing downstream is approximate. If output
degraded after quantizing, the rounding happened in the exporter — check the worst-error
figure it printed.

## Parallel

**`require()` is unavailable in a parallel phase.** Require in serial, at the worker's top
level.

**Everything a parallel handler touches must be resolved in the serial phase.** A module
resolved lazily on first use works perfectly in every serial test and then fails inside an
Actor. `nn` used to resolve `functional` on demand, which put a `require` inside
`Linear.forward` and made every fused layer unusable in a worker.

**Worker pools must be parented where Scripts run.** `Pool.new` defaults to
`ServerScriptService` and errors on `ReplicatedStorage`, `ServerStorage` and other inert
containers. A pool parented into one looks completely healthy — actors exist, clones are
enabled, destroy works — and every job silently times out.

**`parallel.serve` takes the Actor as its first argument.** `script` is a per-script global;
inside `parallel.lua` it refers to that ModuleScript, so `GetActor()` there always returns
`nil`.

**Seed before you build the pool.** Each Actor is a separate Lua VM with its own `Config`,
so `nano.seed` after `Pool.new` reaches the server and not the workers. `Pool.new`
propagates whatever seed is in force at construction time.

**An Actor drops any message whose topic has no handler bound yet.** No queue, no error on
either side. `Pool.new` calls `waitReady` for you; if you construct actors by hand, wait
before dispatching.

## Deployment

**`nn.compile` returns `nil` for unsupported models.** Check the return value. It supports
`Linear` plus `ReLU`, `Tanh`, `Sigmoid`, `LeakyReLU`, `GELU`, `SiLU`/`Swish` and
`Softplus` — and nothing else. Returning `nil` is deliberate: a compiled path that skipped
a `Dropout` layer would even look correct.

**Recompile after loading new weights.** The compiled function captured the buffers that
existed when it was built.

**Use `actGreedy` in deployment.** `act` samples from the policy distribution — that is
exploration, and it belongs in training.

## Reproducibility

**Seed before you build anything.** `nano.seed` covers every random source in the library,
including tensor init — which happens at construction, so seeding after you build the model
does not reproduce its starting weights.

## Luau and Roblox

**Class tables need `__call` on their metatable.** A plain table is not callable, so
`nn.Linear(2, 8)` fails without it. A custom layer needs two: one on the metatable to
construct, one on the class to forward.

**`local function` is invisible above its definition.**

**A statement starting with `(` glues to the previous line.** `x = a + b` then `(c):d()`
parses as `b(c):d()`.

**`collectgarbage("collect")` is blocked in Roblox.** Only `gcinfo()` and `"count"` are
available, so allocation measurements have no clean baseline.

**Native codegen is server-side.** Client scripts run the same code interpreted. Wrong
`--!native` type annotations are worse than none — under native codegen they dictate the
generated machine code.
