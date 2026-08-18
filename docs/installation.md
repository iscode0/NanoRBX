# Installation

## The instance tree

Nano is one ModuleScript with twelve children. The names must be exact, with no `.lua`
extension on the instances — submodules find each other through `script.Parent.X`, so the
hierarchy is part of the API.

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

If you are installing from `Nano.rbxmx`, right-click `ReplicatedStorage` → **Insert from File**
and the whole tree arrives correctly parented.

## NanoWorker must be a Script

`NanoWorker` is only needed if you use [parallel](reference/parallel.md), but if you include it,
it must be a **Script** with `Enabled = false`.

The pool clones it into Actors and enables the clones. A ModuleScript cannot work here: it has no
`Enabled` property, and nothing would ever `require` the clones, so the message handlers would
never bind. The pool would look completely healthy — actors exist, clones are parented — and every
job would silently time out.

## Requiring it

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)

local T     = nano.Tensor
local nn    = nano.nn
local optim = nano.optim
```

`require` returns one table holding every module plus a set of shorthand aliases. See
[nano](reference/nano.md) for the complete list.

## Load order

`init.lua` requires the submodules in a fixed order, and one dependency is load-bearing:
**`functional` must load before `nn`, `rnn` and `rl`.** Requiring `functional` is what installs
its extra operations onto `Tensor`, so `T.cat` and `F.cat` become the same function. Modules that
call `T.cat` need that to have already happened.

You do not have to think about this unless you edit `init.lua`.

## Native code generation

`Tensor`, `functional`, `nn`, `optim`, `rnn`, `data` and `rl` carry `--!native` and `--!optimize 2`.

Native codegen is **server-side only**. The same code on a client runs interpreted — correct, just
slower. If you are training on the client, expect roughly interpreter-speed numbers rather than the
ones in [Performance](performance.md).

## Optional development scripts

These are not part of the library. Put them in `ServerScriptService` as Scripts if you intend to
modify Nano itself.

| script | purpose | when to run |
|---|---|---|
| `TestTensor` | Verifies the Tensor core alone, bypassing `init.lua` | First, and after any Tensor change |
| `TestGradients` | Verifies every backward pass against finite differences | After **any** change to a differentiable op |
| `TestAlgorithms` | Agent contracts, and whether they actually learn | After changing PPO / SAC / DQN |
| `TestAdditions` | Seeding, Trainer, diagnose, prioritized replay, compile | After changing those |
| `TestRecurrent` | RecurrentPPO on a provable memory task | After changing rnn or RecurrentPPO |
| `BenchPerf` | Measures the optimisations end to end | When you think you made something faster |
| `BenchBuffer` | Table vs buffer storage micro-benchmarks | Historical; the migration is done |

See [Testing](testing.md) for what each one proves.

## Verifying the install

```lua
local nano = require(game:GetService("ReplicatedStorage").Nano)
print(nano.VERSION)                     --> 3.3.0

local t = nano.tensor({{1, 2}, {3, 4}})
print(t:numel(), t:dim())               --> 4  2
print(nano.sum(t):item())               --> 10
```

If that runs, the tree is correct.
