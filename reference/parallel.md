# parallel

Actor worker pools, weight transport and evolutionary training.

```lua
local parallel = nano.parallel
```

## What Actors can and cannot do here

Measured, not assumed:

| Operation | Cost |
| --- | --- |
| Shipping a `{64,64}` tensor through SharedTable | ~12 ms |
| Computing a full batch-32 training step | ~1.7 ms |

**Parallelising a single gradient update is strictly slower than not doing it.** Transfer
dominates by roughly 7x. Actors pay off only when work per message is large and payload is
small:

- ✅ evolutionary populations (genome in, fitness scalar out)
- ✅ rollout collection across many environments
- ✅ hyperparameter and seed sweeps
- ✅ many independent NPCs each running a small network
- ❌ splitting one matmul, one minibatch, or one backward pass

Servers expose roughly 2 worker threads, clients up to 8. Use **more actors than threads**
so the scheduler can balance them.

## Pool

| Member | Description |
| --- | --- |
| `parallel.Pool.new(workerScript, workerCount?, parent?)` → `Pool` | Clone a worker template into Actors. `workerCount` defaults to 8, `parent` to `ServerScriptService`. Blocks until every worker is ready. |
| `pool.count: number` | Worker count. |
| `pool.actors: {Actor}` | The Actor instances. |
| `pool.container: Folder` | The folder holding the actors, named `NanoPool`. |
| `pool:waitReady(timeout?)` → `boolean` | Block until every worker has bound its handlers. Called automatically by `new`. |
| `pool:send(workerIndex, topic, ...)` → `()` | Send one message to one worker. |
| `pool:broadcast(topic, ...)` → `()` | Send the same message to every worker. |
| `pool:map(topic, jobs, timeout?)` → `{any}` | Dispatch jobs round-robin and yield until every result returns. |
| `pool:seedWorkers(base)` → `()` | Give each worker its own reproducible stream. Worker `i` receives `base + i`. Called automatically by `Pool.new` when a seed is in force. |
| `pool:destroy()` → `()` | Tear down the actors and the container. |
| `parallel.suggestedWorkers()` → `number` | A reasonable worker count for the current context. |

```lua
local pool = parallel.Pool.new(script.NanoWorker, 8)
local results = pool:map("EvaluateGenome", genomes)
pool:broadcast("UpdateWeights", packed)
pool:destroy()
```

### Two errors that are deliberate

**The template must be a Script, not a ModuleScript.** A ModuleScript has no `Enabled`
property, and nothing would ever require the clones, so no handler would bind. The pool
checks and errors with that explanation rather than constructing something that will
silently never work.

**The parent must be somewhere Scripts run.** `Pool.new` defaults to `ServerScriptService`
and errors on `ReplicatedStorage`, `ServerStorage` and other inert containers. A pool
parented into one looks completely healthy — actors exist, clones are enabled, `destroy`
works — and every single job times out.

### Why waitReady exists

An Actor **drops** any message whose topic has no handler bound yet. There is no queue and
no error on either side.

Cloned Scripts do not run their top-level code until Roblox schedules them, so a pool
dispatched the instant it is constructed sends every job into a worker that is not
listening. They vanish, surfacing only as a timeout with nothing to show for it. `Pool.new`
calls `waitReady` for you.

### Seeding across the actor boundary

Every Actor is a separate Lua VM, so it loads its **own** copy of `Config` with its own
unseeded `Random`. `nano.seed` on the server cannot reach a worker — which meant a seeded
run was reproducible everywhere except inside the pool, and evolution and parallel rollouts
were the two places that mattered most.

`Pool.new` now propagates the seed automatically after `waitReady`, so seeding the server
before you build the pool is enough:

```lua
nano.seed(12345)
local pool = parallel.Pool.new(script.NanoWorker, 8)   -- workers seeded here
```

Worker `i` gets `base + i`, **not** `base`. Seeding every worker identically would make each
one draw the same exploration noise and the same environment layout, so a population of 8
would evaluate 8 copies of the same rollout — reproducible and useless.

The propagation happens after `waitReady` and never before, because an Actor drops any
message whose topic is not bound yet, and a seed dropped on the floor fails silently.
`parallel.serve` binds the reserved `__nanoSeed` topic for you; using that name for your own
handler errors.

## Worker side

| Member | Description |
| --- | --- |
| `parallel.serve(actor, handlers)` → `()` | Bind a table of `topic -> function` handlers on the worker's own Actor. |

```lua
-- inside NanoWorker, at the top level, SERIAL phase
local nano = require(game:GetService("ReplicatedStorage").Nano)
local nn, parallel = nano.nn, nano.parallel

local model = nn.Sequential(
    nn.Linear(8, 32), nn.Tanh(),
    nn.Linear(32, 2), nn.Tanh()
)

parallel.serve(script:GetActor(), {
    EvaluateGenome = function(genome)
        model:setFlat(genome)
        return runEpisode()
    end,
})
```

**`serve` takes the Actor as its first argument.** `script` is a per-script global; inside
`parallel.lua` it refers to that ModuleScript, so calling `GetActor()` there always returns
`nil`. Only the worker's own `script` can find the Actor it lives under.

**`require()` does not work in a desynchronized parallel phase.** Require everything and
build your models at the top level, in serial. The handlers run desynchronized and can only
use what already exists.

This extends further than it first appears: *everything* a handler touches must be resolved
in the serial phase. A module resolved lazily on first use works perfectly in every serial
test and then fails inside an Actor. `nn` used to resolve `functional` on demand, which put
a `require` inside `Linear.forward` and made every fused layer unusable in a worker.

Each worker should own one model and reuse it for every job. Building a fresh one per
genome allocates a full parameter set per evaluation.

Handlers return a payload; `serve` synchronizes before firing the completion event, because
touching the data model in parallel is unsafe.

## Weight transport

| Member | Description |
| --- | --- |
| `parallel.packWeights(model)` → `{shapes: {{number}}, data: string}` | Pack every parameter into one base64 string. |
| `parallel.unpackWeights(model, payload)` → `()` | Load a packed payload into a model. |

One string crosses the boundary instead of thousands of table writes. That is the
difference between a viable transfer and a 12 ms one.

## Evolution

Population-based training across a worker pool. Genomes are flat number arrays, so an
individual is one array copy and the result is one scalar — the workload Actors were built
for.

| Member | Description |
| --- | --- |
| `parallel.Evolution.new(pool, genomeSize, populationSize?)` → `Evolution` | `genomeSize` is `model:numParameters()`, `populationSize` defaults to 32. |
| `evo.sigma: number` | Mutation standard deviation. Defaults to `0.1`. |
| `evo.eliteFraction: number` | Fraction of the population kept as parents. Defaults to `0.25`. |
| `evo.generation: number` | Generations completed. |
| `evo.population: {{number}}` | The current genomes. |
| `evo.fitness: {number}` | Fitness per genome from the last `step`. |
| `evo.size: number` | Population size. |
| `evo:step(topic, timeout?)` → `(number, number)` | Evaluate the whole population in parallel, then produce the next generation. Returns best and mean fitness. |
| `evo:evolve()` → `()` | Produce the next generation from the current fitness. Called by `step`. |
| `evo:noise()` → `number` | One mutation sample, scaled by `sigma`. |
| `evo:best()` → `{number}` | The highest-fitness genome. |

```lua
local evo = parallel.Evolution.new(pool, model:numParameters(), 64)
evo.sigma = 0.1
evo.eliteFraction = 0.25

for gen = 1, 200 do
    local best, mean = evo:step("EvaluateGenome")
    print(gen, best, mean)
end

model:setFlat(evo:best())
```

Evolution needs no gradients at all, so wrap every fitness evaluation in `nano.noGrad` —
the worker is doing pure inference and the graph is pure waste.

Seed with `nano.seed` before constructing the population if you want a reproducible run;
mutation draws from the shared stream.
