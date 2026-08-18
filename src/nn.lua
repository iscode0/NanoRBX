--!strict
--!native
--!optimize 2
--[[
	Nano.nn — layers, containers and losses.

	A Module keeps a registry of its parameters and child modules so
	parameters() can walk the tree and hand an optimiser one flat list.
	The list is cached; registration invalidates it.
]]

local Tensor = require(script.Parent.Tensor)
local Config = require(script.Parent.Config)

local b_create = buffer.create
local b_read = buffer.readf64
local b_write = buffer.writef64
local b_copy = buffer.copy

type Tensor = Tensor.Tensor

local nn = {}

--[[
	Resolved eagerly, at load. This must never become lazy again.

	It was once resolved on first use, so that requiring nn would not silently
	install ops onto Tensor. That is a real concern, but deferring the require
	put it inside Linear.forward — and `require` is UNAVAILABLE during the
	parallel phase, so the first forward pass inside any Actor worker died with
	"require can not be used during the parallel phase". Every layer built on a
	fused kernel was unusable in parallel, and every serial test passed.

	Requiring here means the install happens when nn loads, which is
	serial-phase in every supported entry point, and a parallel phase only ever
	sees an already-resolved table.
]]
local F = require(script.Parent.functional)

local Module = {}
Module.__index = Module
nn.Module = Module

function Module.init(self: any)
	self._parameters = {}
	self._modules = {}
	self._paramCache = nil
	self._numParams = nil
	self.training = true
	return self
end

function Module.registerParameter(self: any, name: string, tensor: Tensor)
	self._parameters[name] = tensor
	self._paramCache = nil
	self._numParams = nil
end

function Module.registerModule(self: any, name: string, module: any)
	self._modules[name] = module
	self._paramCache = nil
	self._numParams = nil
end

--- Every parameter tensor in the tree, depth-first in a stable order.
--- Save/load and flat genome representations depend on that stability.
--- The result is cached — copy it before mutating.
function Module.parameters(self: any): {Tensor}
	local cached = self._paramCache
	if cached then return cached end

	local out: {Tensor} = {}

	local names = {}
	for name in self._parameters do
		table.insert(names, name)
	end
	table.sort(names)
	for _, name in ipairs(names) do
		table.insert(out, self._parameters[name])
	end

	local moduleNames = {}
	for name in self._modules do
		table.insert(moduleNames, name)
	end
	table.sort(moduleNames)
	for _, name in ipairs(moduleNames) do
		for _, p in ipairs(self._modules[name]:parameters()) do
			table.insert(out, p)
		end
	end

	self._paramCache = out
	return out
end

--- Drop cached parameter lists for this module and every descendant. Only
--- needed if a registry is mutated after parameters() was called higher up.
function Module.flushParameterCache(self: any)
	self._paramCache = nil
	self._numParams = nil
	for _, m in self._modules do
		m:flushParameterCache()
	end
end

function Module.zeroGrad(self: any)
	for _, p in ipairs(self:parameters()) do
		p:zeroGrad()
	end
end

--- Enable training behaviour (dropout active) throughout the tree.
function Module.train(self: any)
	self.training = true
	for _, m in self._modules do
		m:train()
	end
	return self
end

--- Disable training behaviour throughout the tree.
function Module.eval(self: any)
	self.training = false
	for _, m in self._modules do
		m:eval()
	end
	return self
end

function Module.numParameters(self: any): number
	local cached = self._numParams
	if cached then return cached end
	local total = 0
	for _, p in ipairs(self:parameters()) do
		total += p:numel()
	end
	self._numParams = total
	return total
end

--- All parameters as one flat Lua array. The genome interface for evolution,
--- and the boundary where buffer storage becomes a plain table.
function Module.getFlat(self: any): {number}
	local out = table.create(self:numParameters())
	local cursor = 1
	for _, p in ipairs(self:parameters()) do
		local data: buffer = p.data
		local n = p:numel()
		local o = 0
		for i = 0, n - 1 do
			out[cursor + i] = b_read(data, o)
			o += 8
		end
		cursor += n
	end
	return out
end

--- Write a flat array back into all parameters. Order must match getFlat.
function Module.setFlat(self: any, values: {number})
	--[[
		A wrong-sized genome used to fail deep inside writef64 with "number
		expected, got nil", naming neither the model nor the sizes. This is the
		mismatched-genome case in the flesh: setFlat has no architecture
		fingerprint, so it is the caller's only warning.
	]]
	local expected = Module.numParameters(self)
	if #values ~= expected then
		error(("setFlat: %d values for a model with %d parameters")
			:format(#values, expected), 2)
	end
	local cursor = 1
	for _, p in ipairs(self:parameters()) do
		local data: buffer = p.data
		local n = p:numel()
		local o = 0
		for i = 0, n - 1 do
			b_write(data, o, values[cursor + i])
			o += 8
		end
		cursor += n
	end
end

--- All parameters as one buffer. Cheaper than getFlat when the destination
--- is also buffer-backed — snapshots, target networks, checkpoints.
function Module.getFlatBuffer(self: any): buffer
	local total = self:numParameters()
	local out = b_create(total * 8)
	local byteCursor = 0
	for _, p in ipairs(self:parameters()) do
		local bytes = p:numel() * 8
		b_copy(out, byteCursor, p.data, p.offset * 8, bytes)
		byteCursor += bytes
	end
	return out
end

--- Inverse of getFlatBuffer.
function Module.setFlatBuffer(self: any, values: buffer)
	local byteCursor = 0
	for _, p in ipairs(self:parameters()) do
		local bytes = p:numel() * 8
		b_copy(p.data, p.offset * 8, values, byteCursor, bytes)
		byteCursor += bytes
	end
end

--[[
	Two __call hooks on two tables: one on the class metatable so
	nn.Linear(2, 8) constructs, one on the class table so model(x) forwards.
	Module methods are flattened in so instance lookup is a single hop.
]]
local function makeModule(class: any)
	for name, fn in Module do
		if name ~= "__index" and class[name] == nil then
			class[name] = fn
		end
	end
	setmetatable(class, {
		__index = Module,
		__call = function(cls, ...) return cls.new(...) end,
	})
	class.__index = class
	class.__call = function(self, ...) return self:forward(...) end
	return class
end

--- Xavier/Glorot initialisation. Keeps variance roughly constant through
--- tanh and sigmoid layers.
function nn.xavier(fanIn: number, fanOut: number, shape: {number}): Tensor
	local std = math.sqrt(2 / (fanIn + fanOut))
	local t = Tensor.randn(shape, true)
	local data: buffer = t.data
	local o = 0
	for _ = 1, t:numel() do
		b_write(data, o, b_read(data, o) * std)
		o += 8
	end
	return t
end

--- He initialisation. Doubles Xavier's variance to compensate for ReLU
--- discarding half its input.
function nn.he(fanIn: number, shape: {number}): Tensor
	local std = math.sqrt(2 / fanIn)
	local t = Tensor.randn(shape, true)
	local data: buffer = t.data
	local o = 0
	for _ = 1, t:numel() do
		b_write(data, o, b_read(data, o) * std)
		o += 8
	end
	return t
end

local Linear = makeModule({})
nn.Linear = Linear

--- Fully connected layer: y = x @ W + b, with b broadcasting across the batch.
--- @param bias boolean? -- pass false to omit
--- @param init string? -- "he" for He init, otherwise Xavier
function Linear.new(inFeatures: number, outFeatures: number, bias: boolean?, init: string?)
	local self = Module.init(setmetatable({}, Linear))

	self.inFeatures = inFeatures
	self.outFeatures = outFeatures

	local weight
	if init == "he" then
		weight = nn.he(inFeatures, { inFeatures, outFeatures })
	else
		weight = nn.xavier(inFeatures, outFeatures, { inFeatures, outFeatures })
	end
	self.weight = weight
	self:registerParameter("weight", weight)

	if bias ~= false then
		local b = Tensor.zeros({ outFeatures }, true)
		self.bias = b
		self:registerParameter("bias", b)
	end

	return self
end

function Linear.forward(self: any, x: Tensor): Tensor
	-- one fused node, not matmul + add
	return F.linear(x, self.weight, self.bias)
end

--- Scale weights and zero the bias after construction. scaleWeights(0.01) on
--- a policy head makes the policy start undecided rather than opinionated.
function Linear.scaleWeights(self: any, factor: number)
	local wData: buffer = self.weight.data
	local o = 0
	for _ = 1, self.weight:numel() do
		b_write(wData, o, b_read(wData, o) * factor)
		o += 8
	end
	local b = self.bias
	if b then
		buffer.fill(b.data, 0, 0, b:numel() * 8)
	end
	return self
end

local function activation(name: string, fn: (Tensor) -> Tensor)
	local class = makeModule({})
	class.new = function(...)
		local self = Module.init(setmetatable({}, class))
		self.args = { ... }
		return self
	end
	class.forward = function(self: any, x: Tensor): Tensor
		return fn(x)
	end
	nn[name] = class
	return class
end

activation("ReLU", function(x) return Tensor.relu(x) end)
activation("Tanh", function(x) return Tensor.tanh(x) end)
activation("Sigmoid", function(x) return Tensor.sigmoid(x) end)

local LeakyReLU = makeModule({})
nn.LeakyReLU = LeakyReLU

--- @param slope number? -- negative-side gradient, defaults to 0.01
function LeakyReLU.new(slope: number?)
	local self = Module.init(setmetatable({}, LeakyReLU))
	self.slope = slope or 0.01
	return self
end

function LeakyReLU.forward(self: any, x: Tensor): Tensor
	return Tensor.leakyRelu(x, self.slope)
end

local Softmax = makeModule({})
nn.Softmax = Softmax

--- @param dim number? -- defaults to the last dimension, which for
--- {batch, classes} is per-row
function Softmax.new(dim: number?)
	local self = Module.init(setmetatable({}, Softmax))
	self.dim = dim
	return self
end

function Softmax.forward(self: any, x: Tensor): Tensor
	return Tensor.softmax(x, self.dim)
end

local Sequential = makeModule({})
nn.Sequential = Sequential

--- Run layers in order. Each is registered as a child, so parameters() walks
--- all of them.
function Sequential.new(...)
	local self = Module.init(setmetatable({}, Sequential))
	self.layers = { ... }
	for i, layer in ipairs(self.layers) do
		self:registerModule(string.format("%03d", i), layer)
	end
	return self
end

function Sequential.forward(self: any, x: Tensor): Tensor
	for _, layer in self.layers do
		x = layer:forward(x)
	end
	return x
end

local LayerNorm = makeModule({})
nn.LayerNorm = LayerNorm

--- Normalise each row to zero mean and unit variance, then scale and shift.
--- Row-wise, not batch-wise: statistics come from within the sample, so
--- training and inference agree exactly at any batch size.
function LayerNorm.new(features: number, eps: number?)
	local self = Module.init(setmetatable({}, LayerNorm))
	self.features = features
	self.eps = eps or 1e-5
	self:registerParameter("gamma", Tensor.ones({ features }, true))
	self:registerParameter("beta", Tensor.zeros({ features }, true))
	self.gamma = self._parameters.gamma
	self.beta = self._parameters.beta
	return self
end

function LayerNorm.forward(self: any, x: Tensor): Tensor
	return F.layerNorm(x, self.gamma, self.beta, self.eps)
end

--- Inverted dropout. Survivors are scaled by 1/(1-p) so the expected
--- activation is unchanged, which is why inference needs no adjustment.
function nn.dropoutFn(x: Tensor, p: number, training: boolean): Tensor
	if not training or p <= 0 then
		return x
	end
	local n = x:numel()
	local scale = 1 / (1 - p)
	local mask = b_create(n * 8)
	local o = 0
	for _ = 1, n do
		b_write(mask, o, if Config.random() < p then 0 else scale)
		o += 8
	end
	return Tensor.mul(x, Tensor.new(mask, table.clone(x.shape), false))
end

local Dropout = makeModule({})
nn.Dropout = Dropout

--- @param p number? -- drop probability, defaults to 0.5
function Dropout.new(p: number?)
	local self = Module.init(setmetatable({}, Dropout))
	self.p = p or 0.5
	return self
end

function Dropout.forward(self: any, x: Tensor): Tensor
	return nn.dropoutFn(x, self.p, self.training)
end

local Embedding = makeModule({})
nn.Embedding = Embedding

--- Learned vector per integer index. Equivalent to one-hot @ W but computed
--- as a gather, so cost is independent of vocabulary size.
function Embedding.new(count: number, dim: number)
	local self = Module.init(setmetatable({}, Embedding))
	self.count = count
	self.dim = dim
	self:registerParameter("weight", F.uniform({ count, dim }, -0.1, 0.1, true))
	self.weight = self._parameters.weight
	return self
end

--- @param indices {number} -- 1-based, one per output row
function Embedding.forward(self: any, indices: {number}): Tensor
	local n = #indices
	local dim = self.dim
	local w = self.weight
	local src = w.data
	local rowBytes = dim * 8
	local out = b_create(n * rowBytes)

	for i = 1, n do
		local idx = indices[i]
		if idx < 1 or idx > self.count then
			error(("Embedding: index %d out of range 1..%d"):format(idx, self.count), 2)
		end
		b_copy(out, (i - 1) * rowBytes, src, (idx - 1) * rowBytes, rowBytes)
	end

	local res = Tensor.new(out, { n, dim }, false)
	if Config.gradEnabled and w.requiresGrad then
		res.requiresGrad = true
		res._prev = { w }
		res._isLeaf = false
		res._backward = function()
			local g = res.grad :: buffer
			local total = self.count * dim
			local gw = b_create(total * 8)
			for i = 1, n do
				local srcBase = (indices[i] - 1) * rowBytes
				local dstBase = (i - 1) * rowBytes
				for j = 0, dim - 1 do
					local s = srcBase + j * 8
					-- read-modify-write, so a repeated index accumulates
					b_write(gw, s, b_read(gw, s) + b_read(g, dstBase + j * 8))
				end
			end
			Tensor._accumulate(w, gw, total)
		end
	end
	return res
end

local Flatten = makeModule({})
nn.Flatten = Flatten

--- {B, ...} -> {B, prod(rest)}. Keeps the batch dimension.
function Flatten.new()
	return Module.init(setmetatable({}, Flatten))
end

function Flatten.forward(self: any, x: Tensor): Tensor
	local batch = x.shape[1]
	return Tensor.reshape(x, { batch, x:numel() // batch })
end

--[[
	Wrap a functional activation as a Module.

	The function is resolved once, here, and captured as an upvalue. Looking it
	up by string key inside forward cost a hash lookup on every layer of every
	forward pass, for a value that never changes.
]]
local function functionalActivation(name: string, key: string, defaultArg: number?)
	local fn = (F :: any)[key]
	if not fn then
		error(("nn: functional has no activation named %q"):format(key), 2)
	end

	local class = makeModule({})
	class.new = function(arg)
		local self = Module.init(setmetatable({}, class))
		self.arg = arg or defaultArg
		return self
	end
	class.forward = function(self: any, x: Tensor): Tensor
		local arg = self.arg
		if arg ~= nil then
			return fn(x, arg)
		end
		return fn(x)
	end
	nn[name] = class
	return class
end

functionalActivation("GELU", "gelu")
functionalActivation("SiLU", "silu")
functionalActivation("Swish", "silu")
functionalActivation("Mish", "mish")
functionalActivation("ELU", "elu", 1)
functionalActivation("Softplus", "softplus", 1)
functionalActivation("HardSigmoid", "hardSigmoid")

local MSELoss = makeModule({})
nn.MSELoss = MSELoss

--- @param reduction string? -- "mean" (default) or "sum"
function MSELoss.new(reduction: string?)
	local self = Module.init(setmetatable({}, MSELoss))
	self.reduction = reduction or "mean"
	return self
end

function MSELoss.forward(self: any, prediction: Tensor, target: Tensor): Tensor
	return F.mseLoss(prediction, target, self.reduction)
end

local CrossEntropyLoss = makeModule({})
nn.CrossEntropyLoss = CrossEntropyLoss

--- Cross entropy over LOGITS. Applying softmax then log separately is
--- numerically worse than logSoftmax, and taking logits means the caller
--- cannot accidentally double-softmax.
function CrossEntropyLoss.new()
	return Module.init(setmetatable({}, CrossEntropyLoss))
end

--- @param targets {number} -- 1-based class indices, one per batch row
function CrossEntropyLoss.forward(self: any, logits: Tensor, targets: {number}): Tensor
	local logProbs = Tensor.logSoftmax(logits, #logits.shape)
	return Tensor.nllLoss(logProbs, targets)
end

local HuberLoss = makeModule({})
nn.HuberLoss = HuberLoss

--- Quadratic near zero, linear beyond delta. Much less sensitive to outliers
--- than MSE, which is why value functions usually prefer it.
--- @param delta number? -- defaults to 1
function HuberLoss.new(delta: number?)
	local self = Module.init(setmetatable({}, HuberLoss))
	self.delta = delta or 1
	return self
end

function HuberLoss.forward(self: any, prediction: Tensor, target: Tensor): Tensor
	return F.huber(prediction, target, self.delta)
end

local BCELoss = makeModule({})
nn.BCELoss = BCELoss

--- Binary cross entropy over LOGITS, using the stable form
--- max(x,0) - x*y + log(1 + exp(-|x|)). Feeding sigmoid output into a naive
--- log-based BCE gives log(0) as soon as the network becomes confident.
--- @param reduction string? -- "mean" (default) or "sum"
function BCELoss.new(reduction: string?)
	local self = Module.init(setmetatable({}, BCELoss))
	self.reduction = reduction or "mean"
	return self
end

function BCELoss.forward(self: any, logits: Tensor, targets: Tensor): Tensor
	local term = Tensor.add(
		Tensor.sub(Tensor.relu(logits), Tensor.mul(logits, targets)),
		Tensor.log(Tensor.add(Tensor.exp(Tensor.neg(Tensor.abs(logits))), 1))
	)
	if self.reduction == "sum" then
		return Tensor.sum(term)
	end
	return Tensor.mean(term)
end

local L1Loss = makeModule({})
nn.L1Loss = L1Loss

--- @param reduction string? -- "mean" (default) or "sum"
function L1Loss.new(reduction: string?)
	local self = Module.init(setmetatable({}, L1Loss))
	self.reduction = reduction or "mean"
	return self
end

function L1Loss.forward(self: any, prediction: Tensor, target: Tensor): Tensor
	local term = Tensor.abs(Tensor.sub(prediction, target))
	if self.reduction == "sum" then
		return Tensor.sum(term)
	end
	return Tensor.mean(term)
end

local KLDivLoss = makeModule({})
nn.KLDivLoss = KLDivLoss

--- KL(target || prediction), both given as LOG-probabilities. Used for
--- policy distillation and trust-region penalties.
function KLDivLoss.new()
	return Module.init(setmetatable({}, KLDivLoss))
end

function KLDivLoss.forward(self: any, logPrediction: Tensor, logTarget: Tensor): Tensor
	local p = Tensor.exp(logTarget)
	return Tensor.mean(Tensor.sum(
		Tensor.mul(p, Tensor.sub(logTarget, logPrediction)),
		#logPrediction.shape
		))
end


--[[
	COMPILED INFERENCE

	A normal forward through a Sequential allocates a Tensor table and a
	buffer per layer, plus the graph checks each op performs before deciding
	it has nothing to track. None of that is used at inference.

	compile() walks a Sequential once and returns a plain function over Lua
	arrays that holds only weights and two reusable scratch buffers. No
	Tensor objects, no allocation per call, no graph.

	Worth it when many agents each run a small network every frame — 100 NPCs
	at 60fps spend about 15% of the frame budget on batch-1 forwards. Not
	worth it for training, which needs the graph anyway.

	Supports Sequential stacks of Linear plus the elementwise activations.
	Returns nil for anything else, so the caller can fall back rather than
	get a subtly wrong answer.
]]

local ACTIVATION_CODES = {
	ReLU = 1, Tanh = 2, Sigmoid = 3, LeakyReLU = 4,
	GELU = 5, SiLU = 6, Swish = 6, Softplus = 7,
}

local function activationCode(layer: any): number?
	for name, code in ACTIVATION_CODES do
		if getmetatable(layer) == nn[name] then
			return code
		end
	end
	return nil
end

local function applyActivation(code: number, buf: buffer, n: number, arg: number)
	local o = 0
	if code == 1 then
		for _ = 1, n do
			local v = b_read(buf, o)
			if v < 0 then b_write(buf, o, 0) end
			o += 8
		end
	elseif code == 2 then
		for _ = 1, n do b_write(buf, o, math.tanh(b_read(buf, o))); o += 8 end
	elseif code == 3 then
		for _ = 1, n do b_write(buf, o, 1 / (1 + math.exp(-b_read(buf, o)))); o += 8 end
	elseif code == 4 then
		for _ = 1, n do
			local v = b_read(buf, o)
			if v < 0 then b_write(buf, o, v * arg) end
			o += 8
		end
	elseif code == 5 then
		local C = 0.7978845608028654
		for _ = 1, n do
			local x = b_read(buf, o)
			b_write(buf, o, 0.5 * x * (1 + math.tanh(C * (x + 0.044715 * x * x * x))))
			o += 8
		end
	elseif code == 6 then
		for _ = 1, n do
			local x = b_read(buf, o)
			b_write(buf, o, x / (1 + math.exp(-x)))
			o += 8
		end
	else
		for _ = 1, n do
			local x = b_read(buf, o)
			b_write(buf, o, if x > 20 then x else math.log(1 + math.exp(x)))
			o += 8
		end
	end
end

--- Compile a Sequential into a weights-only forward for single observations.
--- @return ((obs: {number}) -> {number})? -- nil if the model is unsupported
--- Compile a Sequential into a weights-only forward for single observations.
--- @return ((obs: {number}) -> {number})? -- nil if the model is unsupported
function nn.compile(model: any): (({number}) -> {number})?
	local layers = model.layers
	if not layers then return nil end

	--[[
		Everything the forward needs is hoisted into PARALLEL ARRAYS, indexed by
		step number, rather than a list of tables.

		The old shape stored one table per step and read step.kind, step.w,
		step.inFeatures and step.outFeatures back out on every call — a string
		compare plus four hash lookups per layer, per forward, for values fixed
		at compile time. Array indexing is a different instruction entirely.
	]]
	local kind, srcB, dstB = {}, {}, {}
	local W, B, K, M, TAIL = {}, {}, {}, {}, {}
	local CODE, ARG, WIDTH = {}, {}, {}

	local count = 0
	local widest = 0
	local inSize: number?, outSize: number? = nil, nil

	-- first pass: validate and size the scratch buffers
	for _, layer in ipairs(layers) do
		if getmetatable(layer) == Linear then
			inSize = inSize or layer.inFeatures
			outSize = layer.outFeatures
			if layer.outFeatures > widest then widest = layer.outFeatures end
			if layer.inFeatures > widest then widest = layer.inFeatures end
		elseif not activationCode(layer) then
			return nil      -- unsupported layer; let the caller fall back
		end
	end

	if not inSize then return nil end

	-- two scratch buffers, ping-ponged between layers, allocated once
	local bufA = b_create(widest * 8)
	local bufB = b_create(widest * 8)

	--[[
		The ping-pong is DETERMINISTIC, so it is resolved here instead of at
		runtime. Each step records the exact buffers it reads and writes, which
		removes the swap, the width bookkeeping, and a branch from the inner
		loop of every forward pass.
	]]
	local cur = bufA
	local width = inSize

	for _, layer in ipairs(layers) do
		count += 1
		if getmetatable(layer) == Linear then
			local nxt = if cur == bufA then bufB else bufA
			kind[count] = 1
			srcB[count] = cur
			dstB[count] = nxt
			W[count] = layer.weight.data
			B[count] = if layer.bias then layer.bias.data else nil
			K[count] = layer.inFeatures
			M[count] = layer.outFeatures
			TAIL[count] = layer.outFeatures - 3      -- 4x unroll boundary
			cur = nxt
			width = layer.outFeatures
		else
			kind[count] = 2
			srcB[count] = cur
			CODE[count] = activationCode(layer)
			ARG[count] = layer.slope or layer.arg or 0.01
			WIDTH[count] = width
		end
	end

	local final = cur
	local outCount = outSize :: number
	local inCount = inSize :: number
	local result = table.create(outCount)

	return function(obs: {number}): {number}
		local o = 0
		for i = 1, inCount do
			b_write(bufA, o, obs[i])
			o += 8
		end

		for s = 1, count do
			if kind[s] == 1 then
				local src, dst, w = srcB[s], dstB[s], W[s]
				local k, m, tail = K[s], M[s], TAIL[s]
				local bias = B[s]

				if bias then
					b_copy(dst, 0, bias, 0, m * 8)
				else
					buffer.fill(dst, 0, 0, m * 8)
				end

				for p = 0, k - 1 do
					local xp = b_read(src, p * 8)
					-- post-activation rows are ~half zeros: one compare against
					-- m multiply-adds, the same trade F.linear makes
					if xp ~= 0 then
						local wRow = p * m * 8
						local j = 0
						-- 4x unrolled, matching F.linear. Without this the
						-- compiled path ran a naive loop while the ordinary
						-- forward ran the tuned one, and lost to it outright.
						while j < tail do
							local oj, wj = j * 8, wRow + j * 8
							b_write(dst, oj, b_read(dst, oj) + xp * b_read(w, wj))
							b_write(dst, oj + 8, b_read(dst, oj + 8) + xp * b_read(w, wj + 8))
							b_write(dst, oj + 16, b_read(dst, oj + 16) + xp * b_read(w, wj + 16))
							b_write(dst, oj + 24, b_read(dst, oj + 24) + xp * b_read(w, wj + 24))
							j += 4
						end
						while j < m do
							local oj = j * 8
							b_write(dst, oj, b_read(dst, oj) + xp * b_read(w, wRow + oj))
							j += 1
						end
					end
				end
			else
				applyActivation(CODE[s], srcB[s], WIDTH[s], ARG[s])
			end
		end

		for i = 1, outCount do
			result[i] = b_read(final, (i - 1) * 8)
		end
		return result
	end
end

nn.noGrad = Config.noGrad

return nn
