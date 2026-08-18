# Extending Nano

Custom operations, layers, optimizers and environments. The theme throughout: a wrong
backward pass never errors, so verify everything.

## A custom differentiable op

```lua
local Tensor = nano.Tensor

local function softplus(x)
    return Tensor.unaryOp(x,
        function(v) return math.log(1 + math.exp(v)) end,       -- forward
        function(v, _) return 1 / (1 + math.exp(-v)) end        -- derivative
    )
end
```

| helper | signature |
|---|---|
| `Tensor.unaryOp(a, f, df)` | `df` receives `(input, output)` |
| `Tensor.binaryOp(a, b, f, da?, db?)` | derivatives receive `(x, y, out)` |

The derivative receives the output as well as the input because many derivatives are
cheaper to express that way — `sigmoid` is `out * (1 - out)`, no second exponential.

## Always gradcheck

```lua
local ok, err = Tensor.gradcheck(function(t)
    return Tensor.sum(softplus(t))
end, Tensor.randn({3, 4}))

assert(ok, "backward is wrong, worst relative error " .. err)
```

**A wrong backward pass never errors.** It trains slightly wrong, and you spend a week
blaming hyperparameters. `gradcheck` nudges each input element up and down, measures how
the output actually changed, and compares that against what your backward claims.

Two requirements:

**`fn` must return a scalar.** Wrap in `Tensor.sum` or `Tensor.mean`.

**`fn` must be deterministic.** It is called three times per element, so dropout or
unseeded sampling makes the finite difference compare two different networks. Build models
outside the closure.

If your op has a kink — a point where it is not differentiable, like `relu` at zero — offset
the test inputs away from it. Central differences are simply wrong *at* a kink, and a
failure there is measurement noise rather than a bug.

## Setting graph fields by hand

`Tensor` keeps `newResult` and `markTracked` local, so an op written in `functional.lua`
style has to set the four graph fields itself:

```lua
result.requiresGrad = a.requiresGrad and Config.isGradEnabled()
result._prev = { a }
result._isLeaf = false
result._backward = function()
    -- push gradient into a.grad
end
```

All four, every time. Miss `_prev` and the traversal stops early; miss `_isLeaf` and
in-place ops will happily corrupt your result.

Your `_backward` must write **exactly `numel` gradient entries**. `Tensor.accumulate`
checks this and errors — a short array from a hand-written backward used to be silently
*partially* applied, with no error and no visible symptom beyond a model that trained
slightly wrong.

## A custom layer

```lua
local nn = nano.nn

local MyLayer = setmetatable({}, {
    __index = nn.Module,
    __call = function(c, ...) return c.new(...) end,
})
MyLayer.__index = MyLayer
MyLayer.__call = function(self, ...) return self:forward(...) end

function MyLayer.new(size)
    local self = nn.Module.init(setmetatable({}, MyLayer))
    self:registerParameter("weight", nn.xavier(size, size, {size, size}))
    return self
end

function MyLayer.forward(self, x)
    return nano.Tensor.matmul(x, self.weight)
end
```

The two `__call` assignments do different jobs, and both are required:

- On the **metatable**: makes `MyLayer(8)` construct. A plain table is not callable, so
  without this the constructor call fails.
- On the **class**: makes `layer(x)` forward.

`nn.Module.init` sets up the fields the base methods need. `registerParameter` puts a
tensor into `parameters()`; `registerModule` does the same for a child module,
depth-first.

Verify it end to end:

```lua
local layer = MyLayer(4)
local ok, err = nano.Tensor.gradcheck(function(x)
    return nano.Tensor.sum(layer:forward(x))
end, nano.Tensor.randn({2, 4}))
assert(ok, err)

assert(#layer:parameters() == 1)
assert(layer:numParameters() == 16)
```

If a parameter is missing from `parameters()`, the optimizer never sees it and it never
trains — with no error, and a model that half-works.

### Layers with training-time behaviour

Read `self.training`, which `m:train()` and `m:eval()` set across the whole tree:

```lua
function MyLayer.forward(self, x)
    if self.training then
        return nn.dropoutFn(x, 0.2, true)
    end
    return x
end
```

Note that `nn.compile` returns `nil` for any layer it does not recognise, including yours.
That is correct — a compiled path that skipped your layer would be quietly wrong — but it
means deployment falls back to `model:forward`. See [Deployment](deployment.md).

## A custom optimizer

```lua
local optim = nano.optim

local MyOpt = setmetatable({}, { __index = optim.Optimizer })
MyOpt.__index = MyOpt

function MyOpt.new(params, lr)
    return optim.Optimizer.init(setmetatable({}, MyOpt), params, lr)
end

function MyOpt.step(self)
    self.stepCount += 1
    for _, p in ipairs(self.parameters) do
        if p.grad then
            -- read p.grad, write p.data, both buffers
        end
    end
end
```

`Optimizer.init` supplies `parameters`, `lr` and `stepCount`. The base class supplies
`zeroGrad`, `hardZeroGrad` and `clipGradNorm`, so you only write `step`.

Both `p.data` and `p.grad` are **buffers**. Use `buffer.readf64(buf, (i-1)*8)` and
`buffer.writef64`, not table indexing — `p.data[i]` returns `nil` rather than erroring.

Guard on `p.grad` existing. A parameter that has never received a gradient has none
allocated.

## A custom environment

Three methods, no base class:

```lua
function Env:reset() return obs end
function Env:step(action) return obs, reward, terminated, truncated end
function Env:seed(n) end   -- optional, strongly recommended
```

```lua
local ok, problems = nano.train.checkEnv(env, sampleAction)
for _, p in ipairs(problems) do warn(p) end
```

Two things worth repeating because they are the most common environment bugs:

**Keep `terminated` and `truncated` distinct.** Terminated means no future exists;
truncated means you stopped watching. Collapsing them teaches the critic that value is zero
at every episode boundary.

**Implement `seed`.** Without it, evaluation is not reproducible, so `keepBest` tracks the
luckiest evaluation rather than the best policy, and scores cannot be compared across runs
at all.

Full worked example in [Training an NPC with RL](rl-agent.md).

## Custom losses

A loss is just a differentiable function returning a scalar. It does not need to be a
module:

```lua
local function weightedMSE(prediction, target, weights)
    local diff = Tensor.sub(prediction, target)
    return Tensor.mean(Tensor.mul(Tensor.mul(diff, diff), weights))
end
```

Wrap it in a module only if you want `train()`/`eval()` or stored configuration.

If the composed version shows up in a profile, consider fusing it. That is where the
measured wins have been: Huber composed was seven graph nodes over 512 elements, and
fusing it into one measured **7.11x** faster forward+backward. Fusion pays where node count
dominates arithmetic — fusing the bias into `Linear`, where the bias is 1.6% of the work,
gained 1.06x. See [Performance](../performance.md).

## Checklist

- [ ] `gradcheck` passes, with kinked inputs offset away from the discontinuity
- [ ] Backward writes exactly `numel` gradient entries
- [ ] All four graph fields set: `requiresGrad`, `_prev`, `_isLeaf`, `_backward`
- [ ] Both `__call` metamethods present on a custom layer
- [ ] Every learnable tensor appears in `parameters()`
- [ ] `self.training` respected if behaviour differs at eval
- [ ] Nothing requires a module lazily if it will run inside an Actor

That last one is not obvious. `require` is unavailable in a desynchronized parallel phase,
so a module resolved on first use works perfectly in every serial test and then fails
inside a worker. `nn` used to resolve `functional` on demand, which put a `require` inside
`Linear.forward` and made every fused layer unusable in an Actor.

## Next

- [Tensors and autograd](../concepts/autograd.md) — how the graph works
- [`Tensor` reference](../reference/tensor.md) — `unaryOp`, `binaryOp`, `gradcheck`
- [Performance](../performance.md) — when fusion is worth it
