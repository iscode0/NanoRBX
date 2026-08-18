--!strict
--[[
	Nano — a self-contained ML / DL / RL library for Luau.

	    ReplicatedStorage
	        Nano                 (this ModuleScript, named "Nano")
	            Config
	            Tensor
	            functional
	            nn
	            optim
	            rnn
	            data
	            rl
	            serialize
	            parallel
	            NanoWorker       (Script, Enabled = false)

	    local nano = require(ReplicatedStorage.Nano)

	========================================================================
	WHAT EACH MODULE IS FOR
	========================================================================

	  Config      global grad switch, noGrad
	  Tensor      strided storage + reverse-mode autograd. The core.
	  functional  additional Tensor ops. Requiring it INSTALLS them onto
	              Tensor, so T.cat and F.cat are the same function.
	  nn          layers, containers, losses
	  optim       SGD / Adam / AdamW / RMSprop, clipping, LR schedules
	  rnn         RNN / LSTM / GRU cells, sequence runners, TBPTT
	  data        datasets, batching, scaling, metrics, early stopping
	  rl          buffers, GAE, distributions, normalizers, target networks
	  algorithms  PPO, SAC and DQN assembled from the above
	  train       Env protocol, Trainer, diagnostics
	  serialize   save and load weights and optimizer state
	  parallel    Actor worker pools, for coarse-grained work only

	LOAD ORDER MATTERS in exactly one place: functional must come before
	nn, rnn and rl, because requiring it is what puts cat, gather, layerNorm
	and orthogonal onto Tensor, and those three assume they exist.

	========================================================================
	DESIGN NOTES
	========================================================================

	FLAT STORAGE WITH STRIDES. A tensor is one flat array plus a shape and a
	stride description, not a table of tables. One multiply-add per access
	instead of a pointer chase per dimension, one allocation per op instead
	of one per row. Transpose and reshape become free.

	DIMENSION-AWARE OPS. sum, mean, max and softmax all take a `dim`. A
	softmax that normalises over the whole tensor instead of per row is
	silently wrong for any batch, and that bug is very hard to see.

	EXPLICIT STATE FOR RECURRENCE. Cells are forward(x, state) -> out, state.
	Hidden state on the module breaks parallel environments and turns
	truncated backprop into a mutation. See rnn.lua.

	GUARDED IN-PLACE OPS. add_/mul_ style mutation has no backward rule, so
	it refuses to run on a tensor produced by an operation rather than
	quietly corrupting the graph.

	GRADCHECK BUILT IN. A wrong backward pass never errors. It trains
	slightly wrong. Tensor.gradcheck exists so you catch that in seconds.

	ACTORS ARE FOR COARSE WORK ONLY. Shipping a tensor between actors costs
	far more than computing with it. parallel.lua is for populations,
	rollouts and sweeps — never for splitting one gradient update.

	========================================================================
	QUICK START
	========================================================================

	    local nano = require(ReplicatedStorage.Nano)
	    local T, nn, optim = nano.Tensor, nano.nn, nano.optim

	    local model = nn.Sequential(
	        nn.Linear(2, 8), nn.Tanh(),
	        nn.Linear(8, 1), nn.Sigmoid()
	    )
	    local opt = optim.Adam(model:parameters(), 0.05)
	    local criterion = nn.MSELoss()

	    for _ = 1, 2000 do
	        opt:zeroGrad()
	        local loss = criterion:forward(model:forward(inputs), targets)
	        loss:backward()
	        opt:step()
	    end

	Recurrent:

	    local cell = nano.rnn.LSTM(8, 32, 2)
	    local state = nano.rnn.zeroState(cell, batch)
	    local out; out, state = cell:forward(x, state)
	    state = nano.rnn.detachState(state)      -- at a TBPTT boundary

	Inference builds no graph:

	    nano.noGrad(function() return model:forward(x) end)
]]

local Config = require(script.Config)
local Tensor = require(script.Tensor)

-- Before nn/rnn/rl: this is what installs the extra ops onto Tensor.
local F = require(script.functional)

local nn = require(script.nn)
local optim = require(script.optim)
local rnn = require(script.rnn)
local data = require(script.data)
local rl = require(script.rl)
local algorithms = require(script.algorithms)
local train = require(script.train)
local serialize = require(script.serialize)
local parallel = require(script.parallel)

local nano = {}

nano.Config = Config
nano.Tensor = Tensor
nano.F = F
nano.functional = F
nano.nn = nn
nano.optim = optim
nano.rnn = rnn
nano.data = data
nano.rl = rl
nano.algorithms = algorithms
nano.train = train
nano.serialize = serialize
nano.parallel = parallel

-- ---------------------------------------------------------------- shorthand
nano.tensor = Tensor.new
nano.zeros = Tensor.zeros
nano.ones = Tensor.ones
nano.full = Tensor.full
nano.randn = Tensor.randn
nano.rand = Tensor.rand
nano.noGrad = Config.noGrad
nano.seed = Config.seed
nano.random = Config.random
nano.gaussian = Config.gaussian
nano.diagnose = train.diagnose
nano.compile = nn.compile
nano.gradcheck = Tensor.gradcheck

function nano.matmul(a, b) return Tensor.matmul(a, b) end
function nano.exp(a) return Tensor.exp(a) end
function nano.log(a) return Tensor.log(a) end
function nano.sqrt(a) return Tensor.sqrt(a) end
function nano.abs(a) return Tensor.abs(a) end
function nano.clamp(a, lo, hi) return Tensor.clamp(a, lo, hi) end
function nano.softmax(a, dim) return Tensor.softmax(a, dim) end
function nano.sum(a, dim, keepdim) return Tensor.sum(a, dim, keepdim) end
function nano.mean(a, dim, keepdim) return Tensor.mean(a, dim, keepdim) end
function nano.cat(tensors, dim) return F.cat(tensors, dim) end
function nano.stack(tensors, dim) return F.stack(tensors, dim) end

--- Elementwise minimum, via the identity (a+b-|a-b|)/2. Exact, not an
--- approximation. What PPO's clipped surrogate and SAC's twin-critic
--- minimum both need.
function nano.min(a, b)
	return Tensor.mul(Tensor.sub(Tensor.add(a, b), Tensor.abs(Tensor.sub(a, b))), 0.5)
end

function nano.max2(a, b)
	return Tensor.mul(Tensor.add(Tensor.add(a, b), Tensor.abs(Tensor.sub(a, b))), 0.5)
end

-- ---------------------------------------------------------------- save/load
function nano.save(model, metadata) return serialize.toTable(model, metadata) end
function nano.load(model, state) return serialize.fromTable(model, state) end

--[[
	3.4.0 — cleanup and optimisation pass.

	Bumped because the Toolbox model is the primary install path and VERSION is
	the only way a user can tell a pre-cleanup copy from this one. The
	differences are not cosmetic: optimizer serialization crashed outright,
	toString silently truncated every weight to float32, worker Actors drew
	from an unseeded RNG, and four optimizers applied stale gradients to
	parameters that received none.
]]
nano.VERSION = "3.4.0"

return nano
