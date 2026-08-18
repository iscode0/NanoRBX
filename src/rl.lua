--!strict
--!native
--!optimize 2
--[[
	Nano.rl — reinforcement learning substrate.

	Not algorithms: the pieces every algorithm rebuilds. Nothing here knows
	what PPO or SAC is. Compose them.
]]

local Tensor = require(script.Parent.Tensor)
local Config = require(script.Parent.Config)
local F = require(script.Parent.functional)

type Tensor = Tensor.Tensor

local rl = {}

local t_create = table.create
local m_exp = math.exp
local m_sqrt = math.sqrt
local LOG_2PI = math.log(2 * math.pi)

local RunningNorm = {}
RunningNorm.__index = RunningNorm
rl.RunningNorm = RunningNorm

--- Streaming per-feature mean and variance (Welford). Normalising inputs is
--- routinely the difference between learning and not: a raw feature in
--- [0,80] dominates every dot product against one in [-1,1].
--- Welford rather than summed squares, which loses precision over millions
--- of steps and can produce negative variance.
--- Set .frozen before evaluation so statistics stop moving.
function RunningNorm.new(features: number, clip: number?)
	return setmetatable({
		features = features,
		mean = t_create(features, 0),
		m2 = t_create(features, 0),
		count = 1e-4,           -- tiny nonzero start, avoids a divide by zero
		clip = clip or 10,
		frozen = false,
	}, RunningNorm)
end

function RunningNorm.observe(self: any, x: {number})
	if self.frozen then return end
	self.count += 1
	local n = self.count
	local mean, m2 = self.mean, self.m2
	for i = 1, self.features do
		local delta = x[i] - mean[i]
		mean[i] += delta / n
		m2[i] += delta * (x[i] - mean[i])   -- note: the UPDATED mean
	end
end

function RunningNorm.normalize(self: any, x: {number}): {number}
	local out = t_create(self.features)
	local mean, m2 = self.mean, self.m2
	local n = math.max(self.count, 2)
	local clip = self.clip
	for i = 1, self.features do
		local std = m_sqrt(m2[i] / (n - 1)) + 1e-8
		out[i] = math.clamp((x[i] - mean[i]) / std, -clip, clip)
	end
	return out
end

-- Observe and normalize in one call, which is what a rollout loop wants.
function RunningNorm.step(self: any, x: {number}): {number}
	self:observe(x)
	return self:normalize(x)
end

function RunningNorm.state(self: any)
	return { mean = table.clone(self.mean), m2 = table.clone(self.m2), count = self.count }
end

function RunningNorm.load(self: any, state: any)
	self.mean = table.clone(state.mean)
	self.m2 = table.clone(state.m2)
	self.count = state.count
end

local ReplayBuffer = {}
ReplayBuffer.__index = ReplayBuffer
rl.ReplayBuffer = ReplayBuffer

--- Circular transition buffer for SAC, DQN, TD3 and anything reusing data.
--- Preallocated and overwritten in place; growing a table to a million
--- entries will stall the server on collection.
function ReplayBuffer.new(capacity: number, obsSize: number, actionSize: number)
	return setmetatable({
		capacity = capacity,
		obsSize = obsSize,
		actionSize = actionSize,
		obs = t_create(capacity),
		action = t_create(capacity),
		reward = t_create(capacity, 0),
		nextObs = t_create(capacity),
		done = t_create(capacity, 0),
		count = 0,
		cursor = 0,
	}, ReplayBuffer)
end

--[[
	Store one transition.

	The last argument is `terminated`, NOT a general "done". A time limit is not
	the end of the world: the next state still has value, and storing a
	truncation here teaches the critic that value is zero at every episode
	boundary. The parameter used to be called `done`, which was the exact word
	the rule warns against.
]]
function ReplayBuffer.add(self: any, obs, action, reward: number, nextObs, terminated: boolean)
	self.cursor = (self.cursor % self.capacity) + 1
	local c = self.cursor
	self.obs[c] = obs
	self.action[c] = action
	self.reward[c] = reward
	self.nextObs[c] = nextObs
	-- the stored FIELD stays `done`: it is what gae and the critic read, and
	-- renaming it would churn every consumer for no gain
	self.done[c] = if terminated then 1 else 0
	self.count = math.min(self.count + 1, self.capacity)
end

function ReplayBuffer.ready(self: any, minimum: number): boolean
	return self.count >= minimum
end

-- Uniform sample. Returns tensors ready for a batched forward, plus the
-- reward and done arrays as plain Lua (they are used in scalar arithmetic
-- when building the target, not in the graph).
function ReplayBuffer.sample(self: any, batch: number)
	local obsRows = t_create(batch)
	local actionRows = t_create(batch)
	local nextRows = t_create(batch)
	local rewards = t_create(batch)
	local dones = t_create(batch)

	local n = self.count
	for i = 1, batch do
		local idx = Config.randomInt(n)
		obsRows[i] = self.obs[idx]
		actionRows[i] = self.action[idx]
		nextRows[i] = self.nextObs[idx]
		rewards[i] = self.reward[idx]
		dones[i] = self.done[idx]
	end

	return {
		obs = Tensor.new(obsRows),
		action = Tensor.new(actionRows),
		nextObs = Tensor.new(nextRows),
		reward = rewards,
		done = dones,
		obsRows = obsRows,
		actionRows = actionRows,
	}
end

local RolloutBuffer = {}
RolloutBuffer.__index = RolloutBuffer
rl.RolloutBuffer = RolloutBuffer

--- Fixed-horizon on-policy rollout, cleared after each update. Separate from
--- ReplayBuffer because it is written once, read a few times, then discarded,
--- and carries value estimates and log-probs.
function RolloutBuffer.new(horizon: number)
	return setmetatable({
		horizon = horizon,
		obs = t_create(horizon),
		action = t_create(horizon),
		logp = t_create(horizon, 0),
		reward = t_create(horizon, 0),
		value = t_create(horizon, 0),
		done = t_create(horizon, 0),
		bootstrap = t_create(horizon, 0),
		n = 0,
	}, RolloutBuffer)
end

function RolloutBuffer.add(self: any, obs, action, logp: number, reward: number, value: number, done: boolean, bootstrap: number?)
	local i = self.n + 1
	self.obs[i] = obs
	self.action[i] = action
	self.logp[i] = logp
	self.reward[i] = reward
	self.value[i] = value
	self.done[i] = if done then 1 else 0
	self.bootstrap[i] = bootstrap or 0
	self.n = i
end

function RolloutBuffer.clear(self: any)
	self.n = 0
end

function RolloutBuffer.full(self: any): boolean
	return self.n >= self.horizon
end

--- Generalized Advantage Estimation. lambda interpolates between one-step TD
--- (0: low variance, trusts the critic) and Monte Carlo (1: unbiased, noisy).
--- 0.95 is the near-universal default.
---
--- `dones` cuts the trace; `bootstraps` supplies gamma*V(s') at a time-limit
--- boundary. Both are needed — a cut without a bootstrap tells the critic the
--- future is worth zero, the most common silent bug in PPO implementations.
local Prioritized = {}
Prioritized.__index = Prioritized
rl.PrioritizedReplay = Prioritized

--[[
	Replay weighted by how surprising each transition was.

	Uniform sampling spends most of its effort on transitions the network has
	already fitted. Prioritised sampling draws in proportion to TD error, so
	the batch is made of the things still being got wrong — usually a large
	sample-efficiency gain for DQN and SAC.

	Two corrections make it correct rather than merely faster:

	  alpha  how strongly priority is followed. 0 is uniform, 1 is fully
	         greedy. 0.6 is the usual compromise; fully greedy overfits the
	         few highest-error transitions and starves everything else.

	  beta   importance-sampling weight. Sampling non-uniformly biases the
	         gradient, because high-error transitions now appear more often
	         than their true frequency. The IS weight cancels that. It is
	         annealed from ~0.4 to 1 over training: the bias matters least
	         early, when the value function is wrong anyway, and most later.

	Priorities live in a sum tree so sampling and updating are both O(log n)
	rather than O(n).
]]
function Prioritized.new(capacity: number, obsSize: number, actionSize: number, alpha: number?, beta: number?)
	-- the tree needs a power-of-two leaf count for the index arithmetic
	local leaves = 1
	while leaves < capacity do leaves *= 2 end

	return setmetatable({
		capacity = capacity,
		leaves = leaves,
		tree = t_create(2 * leaves, 0),
		obs = t_create(capacity),
		action = t_create(capacity),
		reward = t_create(capacity, 0),
		nextObs = t_create(capacity),
		done = t_create(capacity, 0),
		count = 0,
		cursor = 0,
		alpha = alpha or 0.6,
		beta = beta or 0.4,
		betaEnd = 1.0,
		maxPriority = 1.0,
	}, Prioritized)
end

local function treeSet(self: any, leaf: number, value: number)
	local i = self.leaves + leaf - 1
	local delta = value - self.tree[i]
	self.tree[i] = value
	i //= 2
	while i >= 1 do
		self.tree[i] += delta
		i //= 2
	end
end

--- Add a transition. New entries get the highest priority seen so far, so
--- everything is replayed at least once before its error is known.
--- Store one transition at maximum priority, so it is sampled at least once.
--- `terminated` means termination, never truncation — see ReplayBuffer.add.
function Prioritized.add(self: any, obs, action, reward: number, nextObs, terminated: boolean)
	self.cursor = (self.cursor % self.capacity) + 1
	local c = self.cursor
	self.obs[c] = obs
	self.action[c] = action
	self.reward[c] = reward
	self.nextObs[c] = nextObs
	self.done[c] = if terminated then 1 else 0
	self.count = math.min(self.count + 1, self.capacity)
	treeSet(self, c, self.maxPriority ^ self.alpha)
end

function Prioritized.ready(self: any, minimum: number): boolean
	return self.count >= minimum
end

--- Sample a batch. Returns the usual tensors plus `indices` and `weights`;
--- multiply your per-sample loss by the weights, then feed the resulting TD
--- errors back through updatePriorities.
function Prioritized.sample(self: any, batch: number)
	local total = self.tree[1]
	if total <= 0 then
		error("PrioritizedReplay: total priority is zero", 2)
	end

	local obsRows = t_create(batch)
	local actionRows = t_create(batch)
	local nextRows = t_create(batch)
	local rewards = t_create(batch)
	local dones = t_create(batch)
	local indices = t_create(batch)
	local weights = t_create(batch)

	-- stratified: one draw per equal slice of the priority mass, which gives
	-- lower variance than `batch` independent draws over the whole range
	local slice = total / batch
	local minProb = math.huge

	for i = 1, batch do
		local target = slice * (i - 1) + Config.random() * slice

		-- walk down the sum tree
		local node = 1
		while node < self.leaves do
			local left = node * 2
			if self.tree[left] >= target then
				node = left
			else
				target -= self.tree[left]
				node = left + 1
			end
		end
		local leaf = node - self.leaves + 1
		if leaf < 1 then leaf = 1 end
		if leaf > self.count then leaf = self.count end

		indices[i] = leaf
		obsRows[i] = self.obs[leaf]
		actionRows[i] = self.action[leaf]
		nextRows[i] = self.nextObs[leaf]
		rewards[i] = self.reward[leaf]
		dones[i] = self.done[leaf]

		local prob = self.tree[self.leaves + leaf - 1] / total
		weights[i] = prob
		if prob < minProb and prob > 0 then minProb = prob end
	end

	-- IS weights, normalised by the largest so they only ever scale DOWN;
	-- scaling a gradient up would defeat the point of clipping
	local beta = self.beta
	local maxWeight = (minProb * self.count) ^ -beta
	for i = 1, batch do
		local p = weights[i]
		weights[i] = if p > 0 then ((p * self.count) ^ -beta) / maxWeight else 1
	end

	return {
		obs = Tensor.new(obsRows),
		action = Tensor.new(actionRows),
		nextObs = Tensor.new(nextRows),
		reward = rewards,
		done = dones,
		obsRows = obsRows,
		actionRows = actionRows,
		indices = indices,
		weights = weights,
	}
end

--- Feed |TD error| back for the sampled transitions.
function Prioritized.updatePriorities(self: any, indices: {number}, errors: {number})
	for i, leaf in ipairs(indices) do
		local e = math.abs(errors[i]) + 1e-6
		if e > self.maxPriority then self.maxPriority = e end
		treeSet(self, leaf, e ^ self.alpha)
	end
end

--- Anneal beta toward 1 over the course of training.
function Prioritized.setProgress(self: any, fraction: number)
	local f = math.clamp(fraction, 0, 1)
	self.beta = 0.4 + (self.betaEnd - 0.4) * f
end

function rl.gae(rewards: {number}, values: {number}, dones: {number}, bootstraps: {number}?, lastValue: number, gamma: number, lambda: number)
	local n = #rewards
	local advantages = t_create(n)
	local returns = t_create(n)
	local lastGae = 0

	for t = n, 1, -1 do
		local nonTerminal = 1 - dones[t]
		local nextValue = if t == n then lastValue else values[t + 1]
		local bootstrap = if bootstraps then (bootstraps[t] or 0) else 0

		local delta = rewards[t] + bootstrap + gamma * nextValue * nonTerminal - values[t]
		lastGae = delta + gamma * lambda * nonTerminal * lastGae
		advantages[t] = lastGae
		returns[t] = lastGae + values[t]
	end

	return advantages, returns
end

--- Normalise advantages in place. Do this per BATCH, not per minibatch:
--- otherwise the same transition gets a different advantage depending on
--- which minibatch it landed in.
function rl.normalizeAdvantages(advantages: {number})
	local n = #advantages
	local mean = 0
	for i = 1, n do mean += advantages[i] end
	mean /= n

	local var = 0
	for i = 1, n do
		local d = advantages[i] - mean
		var += d * d
	end
	local std = m_sqrt(var / n) + 1e-8

	for i = 1, n do
		advantages[i] = (advantages[i] - mean) / std
	end
	return advantages
end

-- Discounted returns, for algorithms that skip the critic entirely
-- (REINFORCE, cross-entropy method, evolutionary fitness shaping).
function rl.discountedReturns(rewards: {number}, dones: {number}, gamma: number): {number}
	local n = #rewards
	local out = t_create(n)
	local running = 0
	for t = n, 1, -1 do
		running = rewards[t] + gamma * running * (1 - dones[t])
		out[t] = running
	end
	return out
end

-- Explained variance of the critic: 1 means perfect, 0 means no better than
-- predicting the mean, negative means actively worse than that. The fastest
-- single number for telling a broken critic from a broken policy.
function rl.explainedVariance(values: {number}, returns: {number}): number
	local n = #returns
	local mean = 0
	for i = 1, n do mean += returns[i] end
	mean /= n

	local varReturns, varResidual = 0, 0
	for i = 1, n do
		local d = returns[i] - mean
		varReturns += d * d
		local r = returns[i] - values[i]
		varResidual += r * r
	end
	if varReturns < 1e-8 then return 0 end
	return 1 - varResidual / varReturns
end

-- Distributions: sample, score, entropy. All log-prob functions are batched
-- and differentiable, so an algorithm swaps discrete for continuous by
-- changing one constructor.

local Normal = {}
Normal.__index = Normal
rl.Normal = Normal

-- `mean` is {N, D}; `logStd` is {D} (shared across the batch) or {N, D}.
function Normal.new(mean: Tensor, logStd: Tensor)
	return setmetatable({ mean = mean, logStd = logStd }, Normal)
end

function Normal.logProb(self: any, actions: Tensor): Tensor
	local std = Tensor.exp(self.logStd)
	local z = Tensor.div(Tensor.sub(actions, self.mean), std)
	local perDim = Tensor.sub(
		Tensor.mul(Tensor.mul(z, z), -0.5),
		Tensor.add(self.logStd, 0.5 * LOG_2PI)
	)
	return Tensor.sum(perDim, #self.mean.shape)
end

-- Entropy of a diagonal Gaussian is sum(logStd) plus a constant. Returned
-- with the constant included so the number is comparable across dimensions.
function Normal.entropy(self: any): Tensor
	local perDim = Tensor.add(self.logStd, 0.5 * (LOG_2PI + 1))
	return Tensor.sum(perDim)
end

-- Sample as plain Lua numbers, for the rollout path where no graph is wanted.
function Normal.sample(self: any): {{number}}
	local means = self.mean:toTable()
	local logStds = self.logStd:toTable()
	local rows = self.mean.shape[1]
	local dims = self.mean.shape[2]
	local shared = #logStds == dims

	local out = t_create(rows)
	for r = 1, rows do
		local row = t_create(dims)
		for d = 1, dims do
			local ls = if shared then logStds[d] else logStds[(r - 1) * dims + d]
			row[d] = means[(r - 1) * dims + d] + rl.gaussianNoise() * m_exp(ls)
		end
		out[r] = row
	end
	return out
end

local Categorical = {}
Categorical.__index = Categorical
rl.Categorical = Categorical

-- Built from LOGITS. logSoftmax is applied once at construction, so scoring
-- several action sets against the same distribution costs one softmax.
function Categorical.new(logits: Tensor)
	local logProbs = Tensor.logSoftmax(logits, #logits.shape)
	return setmetatable({ logits = logits, logProbs = logProbs }, Categorical)
end

function Categorical.logProb(self: any, actions: {number}): Tensor
	return F.gather(self.logProbs, #self.logProbs.shape, actions)
end

function Categorical.entropy(self: any): Tensor
	local p = Tensor.exp(self.logProbs)
	return Tensor.neg(Tensor.mean(
		Tensor.sum(Tensor.mul(p, self.logProbs), #self.logProbs.shape)
		))
end

function Categorical.sample(self: any): {number}
	local flat = Tensor.exp(self.logProbs):toTable()
	local rows = self.logProbs.shape[1]
	local classes = self.logProbs.shape[2]

	local out = t_create(rows)
	for r = 1, rows do
		local u = Config.random()
		local acc = 0
		local base = (r - 1) * classes
		local picked = classes
		for c = 1, classes do
			acc += flat[base + c]
			if u <= acc then
				picked = c
				break
			end
		end
		out[r] = picked
	end
	return out
end

--- Tanh-squashed Gaussian — the SAC policy. Sampling is reparameterised
--- (noise held constant) so the action is differentiable in mean and std and
--- the gradient of Q(s,a) reaches the actor.
---
--- The log(1 - tanh(u)^2) correction is not optional: squeezing an infinite
--- line into [-1,1] concentrates probability mass, and without it a learned
--- temperature tunes against an entropy it cannot measure.
local SquashedNormal = {}
SquashedNormal.__index = SquashedNormal
rl.SquashedNormal = SquashedNormal

function SquashedNormal.new(mean: Tensor, logStd: Tensor)
	return setmetatable({ mean = mean, logStd = logStd }, SquashedNormal)
end

function SquashedNormal.rsample(self: any, rows: number, dims: number): (Tensor, Tensor)
	local std = Tensor.exp(self.logStd)

	local noiseRows = t_create(rows)
	for i = 1, rows do
		local row = t_create(dims)
		for d = 1, dims do
			row[d] = rl.gaussianNoise()
		end
		noiseRows[i] = row
	end
	local noise = Tensor.new(noiseRows)

	local u = Tensor.add(self.mean, Tensor.mul(noise, std))
	local action = Tensor.tanh(u)

	-- z is exactly `noise`, since u = mean + std * noise
	local normalLogp = Tensor.sub(
		Tensor.mul(Tensor.mul(noise, noise), -0.5),
		Tensor.add(self.logStd, 0.5 * LOG_2PI)
	)
	local correction = Tensor.log(
		Tensor.add(Tensor.sub(Tensor.new(1), Tensor.mul(action, action)), 1e-6)
	)
	local logp = Tensor.sum(Tensor.sub(normalLogp, correction), 2)

	return action, logp
end

-- Box-Muller. One standard normal per call; the second value is discarded,
-- which costs one extra sin/cos and saves the state a cached pair would need.
function rl.gaussianNoise(): number
	return Config.gaussian()
end

--- Temporally correlated exploration noise. White noise averages out over
--- consecutive steps, so an agent with heading inertia barely moves under it;
--- OU noise drifts, producing sustained pushes. Worth trying whenever the
--- environment blends headings between steps.
local OUNoise = {}
OUNoise.__index = OUNoise
rl.OUNoise = OUNoise

function OUNoise.new(dims: number, theta: number?, sigma: number?)
	return setmetatable({
		dims = dims,
		theta = theta or 0.15,
		sigma = sigma or 0.2,
		state = t_create(dims, 0),
	}, OUNoise)
end

function OUNoise.sample(self: any): {number}
	local out = t_create(self.dims)
	for i = 1, self.dims do
		self.state[i] += self.theta * (0 - self.state[i]) + self.sigma * Config.gaussian()
		out[i] = self.state[i]
	end
	return out
end

function OUNoise.reset(self: any)
	for i = 1, self.dims do
		self.state[i] = 0
	end
end

-- target = tau*live + (1-tau)*target, in place. Direct .data access because
-- target networks are never differentiated through, so there is no graph to
-- respect and a tensor op would allocate for nothing.
function rl.polyak(targetParams: {Tensor}, liveParams: {Tensor}, tau: number)
	local keep = 1 - tau
	for i, p in ipairs(targetParams) do
		local l = liveParams[i].data
		local d = p.data
		local o = 0
		for _ = 1, p:numel() do
			buffer.writef64(d, o, tau * buffer.readf64(l, o) + keep * buffer.readf64(d, o))
			o += 8
		end
	end
end

function rl.hardCopy(targetParams: {Tensor}, liveParams: {Tensor})
	for i, p in ipairs(targetParams) do
		buffer.copy(p.data, p.offset * 8, liveParams[i].data,
			liveParams[i].offset * 8, p:numel() * 8)
	end
end

return rl
