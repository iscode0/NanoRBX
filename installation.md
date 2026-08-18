# Installation

## From the Toolbox (recommended)

Nano is published as a Roblox model. In Studio, open the Toolbox, search `NanoRBX` or by asset ID
`138581366222476`, and insert it.

Insertion drops the model wherever your current selection is, so **move the `Nano`
ModuleScript into `ReplicatedStorage`** afterwards. Everything else comes parented
correctly underneath it, including `NanoWorker` with `Enabled` already set to `false`.

Then:

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
```

Verify with the snippet at the bottom of this page before writing anything against it.

> Rojo and Wally support are planned. Until then the Toolbox model and the manual tree
> below are the two supported paths.

## Manual installation

Place the library in `ReplicatedStorage` with this exact structure:

```
ReplicatedStorage
    Nano                (ModuleScript, source = init.lua)
        Config          (ModuleScript)
        Tensor          (ModuleScript)
        functional      (ModuleScript)
        nn              (ModuleScript)
        optim           (ModuleScript)
        rnn             (ModuleScript)
        data            (ModuleScript)
        rl              (ModuleScript)
        algorithms      (ModuleScript)
        train           (ModuleScript)
        serialize       (ModuleScript)
        parallel        (ModuleScript)
        NanoWorker      (Script, Enabled = false)
```

Then:

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
```

## Three things that will break it

**Names must match exactly.** No `.lua` extension on the instances. Submodules find each
other with `script.Parent.X`, so a module named `Tensor.lua` or parented anywhere other
than under `Nano` will not resolve.

**`NanoWorker` must be a Script, not a ModuleScript, with `Enabled = false`.** It is only
needed if you use [`parallel`](reference/parallel.md). The pool clones it into Actors and
enables the clones. A ModuleScript has no `Enabled` property, and nothing would ever
require the clones, so the handlers would never bind and every job would silently time
out.

**Load order matters in exactly one place.** `functional` must be required before `nn`,
`rnn` and `rl`, because requiring it is what installs `cat`, `gather`, `layerNorm` and
`orthogonal` onto `Tensor`, and those three modules assume they exist. `init.lua` already
does this in the right order — you only need to care if you require submodules directly.

## Native code generation

`Tensor`, `functional`, `nn`, `optim`, `rnn`, `data` and `rl` are marked `--!native`.

Native codegen is **server-side only**. On the client the same code runs interpreted,
which is correct but slower. This matters if you are running inference on client-side
NPCs — see [Deployment](guides/deployment.md).

Verify it is working with the Script Profiler: `Adam.step` and the loss forwards should
show `<native>`. If they do not, a type misprediction is de-optimizing them.

## Placement, by use case

**Server-only training.** `Nano` in `ReplicatedStorage` (or `ServerStorage` if clients
never need it), training script in `ServerScriptService`.

**Client-side inference.** `Nano` must be in `ReplicatedStorage` so clients can require
it. Train on the server, save with [`serialize`](reference/serialize.md), and load on the
client.

**Parallel work.** Worker pools must be parented where Scripts actually run.
`parallel.Pool.new` defaults to `ServerScriptService` and errors if you pass
`ReplicatedStorage` or `ServerStorage` — a pool parented into an inert container looks
completely healthy and never completes a single job.

## Verify the install

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
print(nano.VERSION)                          --> 3.3.0

local t = nano.tensor({{1, 2}, {3, 4}})
print(t:numel(), t:dim())                    --> 4  2
print(nano.Tensor.sum(t):item())             --> 10
```

If that runs, you are installed. Continue to [Getting started](getting-started.md).
