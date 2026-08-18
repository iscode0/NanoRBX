--!strict
--!native
--!optimize 2
--[[
	Nano.optim — parameter update rules.

	All arithmetic here is on raw numbers, never through the autograd graph:
	updating a weight is not part of the function being differentiated.
	Hot-loop locals are annotated {number} for native codegen; a wrong
	annotation is worse than none, so they appear only where the type is
	guaranteed by construction.
]]

local Tensor = require(script.Parent.Tensor)

type Tensor = Tensor.Tensor

local b_create = buffer.create
local b_read = buffer.readf64
local b_write = buffer.writef64

local optim = {}

-- Optimiser state lives in buffers alongside the parameters, so a step never
-- crosses between representations.
local function stateBuffer(n: number): buffer
	return b_create(n * 8)
end

local Optimizer = {}
Optimizer.__index = Optimizer
optim.Optimizer = Optimizer

function Optimizer.init(self: any, parameters: {Tensor}, lr: number)
	self.parameters = parameters
	self.lr = lr
	self.stepCount = 0
	return self
end

--- Ready every gradient for the next backward pass.
--- Marks buffers stale rather than zero-filling: the first accumulation of
--- the next step overwrites. Same result without an O(params) pass.
function Optimizer.zeroGrad(self: any)
	local params: {Tensor} = self.parameters
	for _, p in ipairs(params) do
		if p.grad then
			p._gradStale = true
		end
	end
end

--- Force an actual zero-fill. Only needed if you inspect .grad between
--- zeroGrad and backward, where a stale buffer still holds last step's values.
function Optimizer.hardZeroGrad(self: any)
	for _, p in ipairs(self.parameters) do
		p:zeroGrad()
	end
end

--- Rescale all gradients together if their combined magnitude exceeds
--- maxNorm. Uses the global norm across every parameter, not per-tensor:
--- clipping each separately would change the direction of the update.
--- @return number -- the norm before clipping
function Optimizer.clipGradNorm(self: any, maxNorm: number): number
	local params: {Tensor} = self.parameters

	local total = 0
	for _, p in ipairs(params) do
		local g: buffer? = p.grad
		if g then
			local o = 0
			for _ = 1, p:numel() do
				local gi = b_read(g, o)
				total += gi * gi
				o += 8
			end
		end
	end

	local norm = math.sqrt(total)
	if norm > maxNorm then
		local scale = maxNorm / (norm + 1e-6)
		for _, p in ipairs(params) do
			local g: buffer? = p.grad
			if g then
				local o = 0
				for _ = 1, p:numel() do
					b_write(g, o, b_read(g, o) * scale)
					o += 8
				end
			end
		end
	end
	return norm
end

-- Flatten shared methods into each class so instance lookup is one hop.
local function makeOptimizer(class: any)
	for name, fn in pairs(Optimizer) do
		if name ~= "__index" and class[name] == nil then
			class[name] = fn
		end
	end
	setmetatable(class, {
		__index = Optimizer,
		__call = function(cls, ...) return cls.new(...) end,
	})
	class.__index = class
	return class
end

local SGD = makeOptimizer({})
optim.SGD = SGD

--- Stochastic gradient descent, optionally with momentum.
--- @param lr number? -- defaults to 0.01
--- @param momentum number? -- defaults to 0
function SGD.new(parameters: {Tensor}, lr: number?, momentum: number?)
	local self = Optimizer.init(setmetatable({}, SGD), parameters, lr or 0.01)
	self.momentum = momentum or 0
	self.velocity = {}
	if self.momentum > 0 then
		for i, p in ipairs(parameters) do
			self.velocity[i] = stateBuffer(p:numel())
		end
	end
	return self
end

function SGD.step(self: any)
	self.stepCount += 1

	local params: {Tensor} = self.parameters
	local lr: number = self.lr
	local momentum: number = self.momentum
	local useMomentum = momentum > 0
	local velocity: {buffer} = self.velocity

	for i, p in ipairs(params) do
		local g: buffer? = p.grad
		-- A stale gradient belongs to the PREVIOUS step: this parameter received
		-- nothing from the last backward pass. Applying it again would keep
		-- updating an unused parameter forever, silently.
		if g and not p._gradStale then
			local data: buffer = p.data
			local n = p:numel()
			local o = 0
			if useMomentum then
				local v: buffer = velocity[i]
				for _ = 1, n do
					local vj = momentum * b_read(v, o) + b_read(g, o)
					b_write(v, o, vj)
					b_write(data, o, b_read(data, o) - lr * vj)
					o += 8
				end
			else
				for _ = 1, n do
					b_write(data, o, b_read(data, o) - lr * b_read(g, o))
					o += 8
				end
			end
		end
	end
end

local Adam = makeOptimizer({})
optim.Adam = Adam

--- Adaptive moment estimation. An EMA of the gradient (beta1) supplies
--- momentum; an EMA of the squared gradient (beta2) scales each parameter
--- individually, which is why Adam needs little learning-rate tuning.
--- @param lr number? -- defaults to 0.001
--- @param beta1 number? -- defaults to 0.9
--- @param beta2 number? -- defaults to 0.999
--- @param epsilon number? -- defaults to 1e-8
function Adam.new(parameters: {Tensor}, lr: number?, beta1: number?, beta2: number?, epsilon: number?)
	local self = Optimizer.init(setmetatable({}, Adam), parameters, lr or 0.001)
	self.beta1 = beta1 or 0.9
	self.beta2 = beta2 or 0.999
	self.epsilon = epsilon or 1e-8
	self.m = {}
	self.v = {}
	for i, p in ipairs(parameters) do
		self.m[i] = stateBuffer(p:numel())
		self.v[i] = stateBuffer(p:numel())
	end
	return self
end

function Adam.step(self: any)
	self.stepCount += 1
	local t: number = self.stepCount

	local params: {Tensor} = self.parameters
	local b1: number, b2: number = self.beta1, self.beta2
	local omb1, omb2 = 1 - b1, 1 - b2
	local eps: number = self.epsilon
	local ms: {buffer}, vs: {buffer} = self.m, self.v

	-- Both bias corrections fold into one per-step scalar, removing two
	-- divisions per weight. Epsilon ends up scaled by sqrt(1-b2^t), the same
	-- discrepancy PyTorch has against the paper; irrelevant at 1e-8.
	local stepSize = self.lr * math.sqrt(1 - b2 ^ t) / (1 - b1 ^ t)

	for i, p in ipairs(params) do
		local g: buffer? = p.grad
		-- A stale gradient belongs to the PREVIOUS step: this parameter received
		-- nothing from the last backward pass. Applying it again would keep
		-- updating an unused parameter forever, silently.
		if g and not p._gradStale then
			local m: buffer, v: buffer = ms[i], vs[i]
			local data: buffer = p.data
			local o = 0
			for _ = 1, p:numel() do
				local gj = b_read(g, o)
				local mj = b1 * b_read(m, o) + omb1 * gj
				local vj = b2 * b_read(v, o) + omb2 * gj * gj
				b_write(m, o, mj)
				b_write(v, o, vj)
				b_write(data, o, b_read(data, o) - stepSize * mj / (math.sqrt(vj) + eps))
				o += 8
			end
		end
	end
end

local AdamW = makeOptimizer({})
optim.AdamW = AdamW

--- Adam with decoupled weight decay. Classic L2-in-the-gradient decay gets
--- divided by sqrt(v) along with everything else, making regularisation
--- inversely proportional to gradient magnitude; this applies decay straight
--- to the weight instead.
--- @param weightDecay number? -- defaults to 0.01
function AdamW.new(parameters: {Tensor}, lr: number?, weightDecay: number?, beta1: number?, beta2: number?, epsilon: number?)
	local self = Optimizer.init(setmetatable({}, AdamW), parameters, lr or 0.001)
	self.beta1 = beta1 or 0.9
	self.beta2 = beta2 or 0.999
	self.epsilon = epsilon or 1e-8
	self.weightDecay = weightDecay or 0.01
	self.m = {}
	self.v = {}
	for i, p in ipairs(parameters) do
		self.m[i] = stateBuffer(p:numel())
		self.v[i] = stateBuffer(p:numel())
	end
	return self
end

function AdamW.step(self: any)
	self.stepCount += 1
	local t: number = self.stepCount

	local params: {Tensor} = self.parameters
	local b1: number, b2: number = self.beta1, self.beta2
	local omb1, omb2 = 1 - b1, 1 - b2
	local eps: number = self.epsilon
	local lr: number = self.lr
	local decay = lr * self.weightDecay
	local ms: {buffer}, vs: {buffer} = self.m, self.v

	local stepSize = lr * math.sqrt(1 - b2 ^ t) / (1 - b1 ^ t)

	for i, p in ipairs(params) do
		local g: buffer? = p.grad
		-- A stale gradient belongs to the PREVIOUS step: this parameter received
		-- nothing from the last backward pass. Applying it again would keep
		-- updating an unused parameter forever, silently.
		if g and not p._gradStale then
			local m: buffer, v: buffer = ms[i], vs[i]
			local data: buffer = p.data
			local o = 0
			for _ = 1, p:numel() do
				local gj = b_read(g, o)
				local mj = b1 * b_read(m, o) + omb1 * gj
				local vj = b2 * b_read(v, o) + omb2 * gj * gj
				b_write(m, o, mj)
				b_write(v, o, vj)
				local w = b_read(data, o)
				b_write(data, o, w - stepSize * mj / (math.sqrt(vj) + eps) - decay * w)
				o += 8
			end
		end
	end
end

local RMSprop = makeOptimizer({})
optim.RMSprop = RMSprop

--- Adam without the momentum term: per-parameter scaling only.
--- @param lr number? -- defaults to 0.01
--- @param alpha number? -- defaults to 0.99
function RMSprop.new(parameters: {Tensor}, lr: number?, alpha: number?, epsilon: number?)
	local self = Optimizer.init(setmetatable({}, RMSprop), parameters, lr or 0.01)
	self.alpha = alpha or 0.99
	self.epsilon = epsilon or 1e-8
	self.v = {}
	for i, p in ipairs(parameters) do
		self.v[i] = stateBuffer(p:numel())
	end
	return self
end

function RMSprop.step(self: any)
	self.stepCount += 1

	local params: {Tensor} = self.parameters
	local lr: number = self.lr
	local alpha: number = self.alpha
	local oma = 1 - alpha
	local eps: number = self.epsilon
	local vs: {buffer} = self.v

	for i, p in ipairs(params) do
		local g: buffer? = p.grad
		-- A stale gradient belongs to the PREVIOUS step: this parameter received
		-- nothing from the last backward pass. Applying it again would keep
		-- updating an unused parameter forever, silently.
		if g and not p._gradStale then
			local v: buffer = vs[i]
			local data: buffer = p.data
			local o = 0
			for _ = 1, p:numel() do
				local gj = b_read(g, o)
				local vj = alpha * b_read(v, o) + oma * gj * gj
				b_write(v, o, vj)
				b_write(data, o, b_read(data, o) - lr * gj / (math.sqrt(vj) + eps))
				o += 8
			end
		end
	end
end

local Scheduler = {}
Scheduler.__index = Scheduler
optim.Scheduler = Scheduler

--- Drive opt.lr from a function of step count. Call sched:step() after
--- opt:step().
--- @param fn (number) -> number -- returns a multiplier on the base lr
function optim.schedule(opt: any, fn: (number) -> number)
	return setmetatable({ opt = opt, fn = fn, baseLR = opt.lr, count = 0 }, Scheduler)
end

function Scheduler.step(self: any): number
	self.count += 1
	self.opt.lr = self.baseLR * self.fn(self.count)
	return self.opt.lr
end

function Scheduler.reset(self: any)
	self.count = 0
	self.opt.lr = self.baseLR
end

--- Linear decay to `finalFraction` over `total` steps. The PPO default.
function optim.linearDecay(total: number, finalFraction: number?)
	local final = finalFraction or 0
	return function(step: number): number
		return 1 - math.min(step / total, 1) * (1 - final)
	end
end

--- Cosine annealing: slow, fast, then a long low tail to settle in.
function optim.cosine(total: number, finalFraction: number?)
	local final = finalFraction or 0
	return function(step: number): number
		local progress = math.min(step / total, 1)
		return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress))
	end
end

--- Multiply by `factor` every `every` steps.
function optim.stepDecay(every: number, factor: number)
	return function(step: number): number
		return factor ^ (step // every)
	end
end

--- Linear warmup, then hand off to `after`. Adam's second-moment estimate is
--- unreliable for the first few dozen steps, so early updates can be huge;
--- ramping from zero makes them harmless.
function optim.warmup(steps: number, after: ((number) -> number)?)
	return function(step: number): number
		if step < steps then
			return step / steps
		end
		return if after then after(step - steps) else 1
	end
end

return optim
