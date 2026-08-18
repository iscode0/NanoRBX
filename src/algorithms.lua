--!strict
--!native
--!optimize 2
--[[
	Nano.algorithms — PPO, SAC and DQN, ready to use.

	rl.lua is substrate: buffers, GAE, distributions. This is the layer above,
	where the substrate is assembled correctly ONCE so every agent inherits it
	rather than re-deriving it.

	THE BUGS THIS MODULE EXISTS TO PREVENT. Every one of these was found the
	hard way, produces no error, and looks exactly like "the algorithm does
	not work on my task":

	  Truncation treated as termination. Hitting a step limit is not the end
	  of the world; the next state still has value. Feeding it to GAE as
	  terminal teaches the critic that value is zero at every episode
	  boundary. Handled here by taking `terminated` and `truncated` as
	  separate arguments and refusing to guess.

	  Clamped actions scored with an unclamped density. If you clamp a
	  sampled action but score it with a plain Gaussian log-pdf, every stored
	  log-prob is wrong, so every ratio is wrong, and the policy gradient
	  points somewhere arbitrary. Actions are never clamped here; the
	  environment clamps.

	  logStd clamped only after the step. A bad update drives it far
	  negative mid-update, std collapses, exp overflows to inf, and every
	  parameter becomes nan with no error. Clamped inside the graph.

	  No gradient clipping. One freak advantage takes the policy out and the
	  next rollout is garbage. Always on, tunable.

	  parameters() cache pollution. `table.insert(model:parameters(), extra)`
	  mutates nn's cached list permanently. Always copied first.

	THE INTERACTION PATTERN is act-then-observe, so an observation can never
	be paired with the wrong log-prob:

	    local action = agent:act(obs)
	    local nextObs, reward, terminated, truncated = env:step(action)
	    agent:observe(reward, terminated, truncated, nextObs)
]]

local Tensor = require(script.Parent.Tensor)
-- required for its side effect: this is what installs gather, cat and the
-- rest onto Tensor. DQN uses Tensor.gather.
local F = require(script.Parent.functional)
local nn = require(script.Parent.nn)
local optim = require(script.Parent.optim)
local rl = require(script.Parent.rl)
local rnn = require(script.Parent.rnn)
local Config = require(script.Parent.Config)
local serialize = require(script.Parent.serialize)

type Tensor = Tensor.Tensor

local algorithms = {}

local LOG_2PI = math.log(2 * math.pi)

-- ==========================================================================
-- SHARED
-- ==========================================================================

local function withDefaults(given: any, defaults: any): any
	local out = {}
	for k, v in defaults do out[k] = v end
	if given then
		for k, v in given do out[k] = v end
	end
	return out
end

--- Build an MLP. The head is scaled down so a policy starts undecided rather
--- than arbitrarily opinionated — small change, large effect on early
--- training.
local function mlp(inSize: number, hidden: number, outSize: number, depth: number, headScale: number?)
	local layers = {}
	local prev = inSize
	for _ = 1, depth do
		table.insert(layers, nn.Linear(prev, hidden))
		table.insert(layers, nn.Tanh())
		prev = hidden
	end
	local head = nn.Linear(prev, outSize)
	if headScale then
		head:scaleWeights(headScale)
	end
	table.insert(layers, head)
	return nn.Sequential(table.unpack(layers))
end

--- Copy a module's parameter list before adding to it. parameters() returns a
--- CACHED table; mutating it in place poisons every later call.
local function paramsPlus(module: any, ...): {Tensor}
	local out = table.clone(module:parameters())
	for _, extra in { ... } do
		table.insert(out, extra)
	end
	return out
end

local function collectParams(...): {Tensor}
	local out = {}
	for _, m in { ... } do
		for _, p in ipairs(m:parameters()) do
			table.insert(out, p)
		end
	end
	return out
end

-- ==========================================================================
-- PPO
-- ==========================================================================

local PPO = {}
PPO.__index = PPO
algorithms.PPO = PPO

PPO.defaults = {
	obsSize = 8,
	actionDim = 2,          -- continuous: dimensions; discrete: action count
	actionType = "continuous",   -- "continuous" or "discrete"
	hidden = 64,
	depth = 2,

	horizon = 512,
	epochs = 10,
	minibatch = 64,

	gamma = 0.99,
	lambda = 0.95,
	clip = 0.2,
	entropyCoef = 0.005,
	valueCoef = 0.5,
	maxGradNorm = 0.5,

	actorLR = 3e-4,
	criticLR = 1e-3,        -- the value function can and should learn faster

	initLogStd = -0.7,      -- exp(-0.7) ~= 0.5
	logStdMin = -2.5,
	logStdMax = 0.5,

	normalizeObs = true,
	normalizeAdvantages = true,
}

--- Create a PPO agent. See PPO.defaults for every option.
function PPO.new(config: any?)
	local c = withDefaults(config, PPO.defaults)
	local self = setmetatable({ config = c, kind = "ppo" }, PPO)

	self.continuous = c.actionType == "continuous"
	self.actor = mlp(c.obsSize, c.hidden, c.actionDim, c.depth, 0.01)
	self.critic = mlp(c.obsSize, c.hidden, 1, c.depth)

	local actorParams
	if self.continuous then
		-- one learnable log-std per dimension, held as a standalone
		-- parameter: the right noise level usually depends on training
		-- progress, not on the state
		local init = table.create(c.actionDim, c.initLogStd)
		self.logStd = Tensor.new(init, { c.actionDim }, true)
		actorParams = paramsPlus(self.actor, self.logStd)
	else
		actorParams = paramsPlus(self.actor)
	end

	self.actorOpt = optim.Adam(actorParams, c.actorLR)
	self.criticOpt = optim.Adam(self.critic:parameters(), c.criticLR)

	self.buffer = rl.RolloutBuffer.new(c.horizon)
	self.obsNorm = if c.normalizeObs then rl.RunningNorm.new(c.obsSize) else nil

	self.pending = nil
	self.stats = {}
	self.updates = 0
	return self
end

local function ppoPrepare(self: any, obs: {number}): {number}
	if self.obsNorm then
		return self.obsNorm:step(obs)
	end
	return obs
end

--[[
	Sample an action and remember what produced it.

	The action is NOT clamped. Clamping here while scoring with an unclamped
	Gaussian density makes every stored log-prob wrong; the environment is
	the right place to bound an action.
]]
function PPO.act(self: any, obs: {number}): any
	local c = self.config
	local normalized = ppoPrepare(self, obs)

	local action, logp, value
	Config.noGrad(function()
		local x = Tensor.new({ normalized })
		value = self.critic:forward(x):item()

		if self.continuous then
			local means = self.actor:forward(x):toTable()
			action = table.create(c.actionDim)
			logp = 0
			for d = 1, c.actionDim do
				local ls = math.clamp(self.logStd:at(d), c.logStdMin, c.logStdMax)
				local std = math.exp(ls)
				local a = means[d] + rl.gaussianNoise() * std
				action[d] = a
				local z = (a - means[d]) / std
				logp += -0.5 * z * z - ls - 0.5 * LOG_2PI
			end
		else
			local logits = self.actor:forward(x)
			local dist = rl.Categorical.new(logits)
			local picked = dist:sample()
			action = picked[1]
			logp = dist.logProbs:at(action)
		end
	end)

	self.pending = { obs = normalized, action = action, logp = logp, value = value }
	return action
end

--- Greedy action: the distribution mean, or the argmax. No exploration noise.
function PPO.actGreedy(self: any, obs: {number}): any
	local normalized = if self.obsNorm then self.obsNorm:normalize(obs) else obs
	local action
	Config.noGrad(function()
		local out = self.actor:forward(Tensor.new({ normalized }))
		action = if self.continuous then out:toTable() else Tensor.argmax(out)
	end)
	return action
end

function PPO.valueOf(self: any, obs: {number}): number
	local normalized = if self.obsNorm then self.obsNorm:normalize(obs) else obs
	local v
	Config.noGrad(function()
		v = self.critic:forward(Tensor.new({ normalized })):item()
	end)
	return v
end

--[[
	Complete the transition started by act().

	`terminated` and `truncated` are separate and neither is optional. A time
	limit is truncation: the trace is cut but the next state's value is
	bootstrapped in, because the world did not end. Conflating them is the
	single most common silent PPO bug.
]]
function PPO.observe(self: any, reward: number, terminated: boolean, truncated: boolean, nextObs: {number}?)
	local p = self.pending
	if not p then
		error("PPO.observe called before act", 2)
	end

	local bootstrap = 0
	if truncated and not terminated then
		if not nextObs then
			error("PPO.observe: a truncated step needs nextObs to bootstrap from", 2)
		end
		bootstrap = self.config.gamma * self:valueOf(nextObs)
	end

	self.buffer:add(p.obs, p.action, p.logp, reward, p.value,
		terminated or truncated, bootstrap)
	self.pending = nil
end

function PPO.ready(self: any): boolean
	return self.buffer.n >= self.config.horizon
end

--- Run the update. Call once the buffer is full, passing the observation the
--- next rollout will start from so the final value can be bootstrapped.
--- @return table -- diagnostics: return, explainedVariance, clipFraction, approxKL, entropy
function PPO.update(self: any, finalObs: {number}): any
	local c = self.config
	local b = self.buffer
	local n = b.n
	if n == 0 then return {} end

	local lastValue = self:valueOf(finalObs)
	local advantages, returns = rl.gae(
		b.reward, b.value, b.done, b.bootstrap, lastValue, c.gamma, c.lambda)

	local ev = rl.explainedVariance(b.value, returns)
	if c.normalizeAdvantages then
		-- per BATCH, not per minibatch: otherwise the same transition gets a
		-- different advantage depending on which minibatch it lands in
		rl.normalizeAdvantages(advantages)
	end

	local order = table.create(n)
	for i = 1, n do order[i] = i end

	local clipSum, klSum, entSum, count = 0, 0, 0, 0

	for _ = 1, c.epochs do
		for i = n, 2, -1 do
			local j = Config.randomInt(i)
			order[i], order[j] = order[j], order[i]
		end

		local cursor = 1
		while cursor <= n do
			local last = math.min(cursor + c.minibatch - 1, n)
			local size = last - cursor + 1

			local mObs = table.create(size)
			local mAct = table.create(size)
			local mLogp = table.create(size)
			local mAdv = table.create(size)
			local mRet = table.create(size)

			for k = cursor, last do
				local idx = order[k]
				local slot = k - cursor + 1
				mObs[slot] = b.obs[idx]
				mAct[slot] = b.action[idx]
				mLogp[slot] = b.logp[idx]
				mAdv[slot] = advantages[idx]
				mRet[slot] = returns[idx]
			end

			local clip, kl, ent = self:updateMinibatch(mObs, mAct, mLogp, mAdv, mRet)
			clipSum += clip; klSum += kl; entSum += ent; count += 1
			cursor = last + 1
		end
	end

	local episodeReturn = 0
	for i = 1, n do episodeReturn += b.reward[i] end

	b:clear()
	self.updates += count

	self.stats = {
		meanReward = episodeReturn / n,
		explainedVariance = ev,
		clipFraction = clipSum / count,
		approxKL = klSum / count,
		entropy = entSum / count,
		updates = self.updates,
	}
	return self.stats
end

function PPO.updateMinibatch(self: any, obsRows, actions, oldLogps, advantages, returns): (number, number, number)
	local c = self.config
	local n = #obsRows

	local X = Tensor.new(obsRows)
	local ADV = Tensor.new(advantages, { n })
	local R = Tensor.new(returns, { n })
	local OLD = Tensor.new(oldLogps, { n })

	self.actorOpt:zeroGrad()
	self.criticOpt:zeroGrad()

	local logp, entropy

	if self.continuous then
		local means = self.actor:forward(X)
		-- clamped INSIDE the graph: clamping only in .data afterwards lets a
		-- single bad update collapse std to zero and produce nan
		local logStd = Tensor.clamp(self.logStd, c.logStdMin, c.logStdMax)
		local std = Tensor.exp(logStd)
		local A = Tensor.new(actions)
		local z = Tensor.div(Tensor.sub(A, means), std)
		local perDim = Tensor.sub(
			Tensor.mul(Tensor.mul(z, z), -0.5),
			Tensor.add(logStd, 0.5 * LOG_2PI)
		)
		logp = Tensor.sum(perDim, 2)
		entropy = Tensor.sum(logStd)
	else
		local dist = rl.Categorical.new(self.actor:forward(X))
		logp = dist:logProb(actions)
		entropy = dist:entropy()
	end

	local logRatio = Tensor.sub(logp, OLD)
	local ratio = Tensor.exp(logRatio)
	local surr1 = Tensor.mul(ratio, ADV)
	local surr2 = Tensor.mul(Tensor.clamp(ratio, 1 - c.clip, 1 + c.clip), ADV)
	-- min via the abs identity: (a + b - |a - b|) / 2
	local objective = Tensor.mul(
		Tensor.sub(Tensor.add(surr1, surr2), Tensor.abs(Tensor.sub(surr1, surr2))), 0.5)
	local policyLoss = Tensor.neg(Tensor.mean(objective))

	local values = Tensor.reshape(self.critic:forward(X), { n })
	local diff = Tensor.sub(values, R)
	local valueLoss = Tensor.mean(Tensor.mul(diff, diff))

	local loss = Tensor.add(
		Tensor.sub(policyLoss, Tensor.mul(entropy, c.entropyCoef)),
		Tensor.mul(valueLoss, c.valueCoef)
	)
	loss:backward()

	-- always on: one freak advantage otherwise takes the policy out
	self.actorOpt:clipGradNorm(c.maxGradNorm)
	self.criticOpt:clipGradNorm(c.maxGradNorm)
	self.actorOpt:step()
	self.criticOpt:step()

	if self.continuous then
		for d = 1, c.actionDim do
			self.logStd:setAt(d, math.clamp(self.logStd:at(d), c.logStdMin, c.logStdMax))
		end
	end

	-- diagnostics: clipFraction healthy around 0.05-0.30, approxKL sustained
	-- above ~0.05 means the learning rate is too high
	local lr = logRatio:toTable()
	local clipped, klSum = 0, 0
	for i = 1, n do
		local r = math.exp(lr[i])
		if r < 1 - c.clip or r > 1 + c.clip then clipped += 1 end
		klSum += (r - 1) - lr[i]
	end
	return clipped / n, klSum / n, entropy:item()
end

-- ==========================================================================
-- SAC
-- ==========================================================================

local SAC = {}
SAC.__index = SAC
algorithms.SAC = SAC

SAC.defaults = {
	obsSize = 8,
	actionDim = 2,
	hidden = 64,
	depth = 2,

	gamma = 0.99,
	tau = 0.01,
	lr = 3e-4,
	batch = 64,
	bufferSize = 50000,
	warmup = 1000,
	updateEvery = 4,
	maxGradNorm = 0.5,

	initLogStd = -0.7,
	logStdMin = -2.5,
	logStdMax = 0.5,
	targetEntropy = nil,        -- defaults to -actionDim

	normalizeObs = true,
}

--[[
	Q(s, a) needs state AND action. Nano has no concatenate op and does not
	need one: a linear layer over [s; a] is exactly W_s.s + W_a.a + b, so two
	Linear layers summed is the same computation.
]]
local function buildCritic(obsSize: number, actionDim: number, hidden: number)
	local q = {}
	q.fromState = nn.Linear(obsSize, hidden)
	q.fromAction = nn.Linear(actionDim, hidden, false)   -- bias comes from fromState
	q.hidden = nn.Linear(hidden, hidden)
	q.out = nn.Linear(hidden, 1)

	function q.forward(state, action)
		local h = Tensor.tanh(Tensor.add(q.fromState:forward(state), q.fromAction:forward(action)))
		return q.out:forward(Tensor.tanh(q.hidden:forward(h)))
	end

	function q.parameters()
		return collectParams(q.fromState, q.fromAction, q.hidden, q.out)
	end

	return q
end

--- Create a SAC agent. Off-policy and continuous-only.
function SAC.new(config: any?)
	local c = withDefaults(config, SAC.defaults)
	c.targetEntropy = c.targetEntropy or -c.actionDim
	local self = setmetatable({ config = c, kind = "sac" }, SAC)

	-- no tanh on the head: the squash happens after sampling so the log-prob
	-- correction can be applied to it
	self.actor = mlp(c.obsSize, c.hidden, c.actionDim, c.depth, 0.01)

	self.q1 = buildCritic(c.obsSize, c.actionDim, c.hidden)
	self.q2 = buildCritic(c.obsSize, c.actionDim, c.hidden)
	self.q1Target = buildCritic(c.obsSize, c.actionDim, c.hidden)
	self.q2Target = buildCritic(c.obsSize, c.actionDim, c.hidden)

	self.q1Params = self.q1.parameters()
	self.q2Params = self.q2.parameters()
	self.q1TargetParams = self.q1Target.parameters()
	self.q2TargetParams = self.q2Target.parameters()
	rl.hardCopy(self.q1TargetParams, self.q1Params)
	rl.hardCopy(self.q2TargetParams, self.q2Params)

	local init = table.create(c.actionDim, c.initLogStd)
	self.logStd = Tensor.new(init, { c.actionDim }, true)
	-- temperature learned in log space so it can never go negative
	self.logAlpha = Tensor.new({ 0 }, { 1 }, true)

	self.actorOpt = optim.Adam(paramsPlus(self.actor, self.logStd), c.lr)
	self.q1Opt = optim.Adam(self.q1Params, c.lr)
	self.q2Opt = optim.Adam(self.q2Params, c.lr)
	self.alphaOpt = optim.Adam({ self.logAlpha }, c.lr)

	self.buffer = rl.ReplayBuffer.new(c.bufferSize, c.obsSize, c.actionDim)
	self.obsNorm = if c.normalizeObs then rl.RunningNorm.new(c.obsSize) else nil

	self.steps = 0
	self.updates = 0
	self.pending = nil
	self.stats = {}
	return self
end

function SAC.alpha(self: any): number
	return math.exp(self.logAlpha:at(1))
end

function SAC.act(self: any, obs: {number}): {number}
	local c = self.config
	self.steps += 1
	local normalized = if self.obsNorm then self.obsNorm:step(obs) else obs

	local action
	if self.steps <= c.warmup then
		-- uniform random fills the buffer with more variety than an
		-- untrained policy would
		action = table.create(c.actionDim)
		for d = 1, c.actionDim do
			action[d] = Config.random() * 2 - 1
		end
	else
		Config.noGrad(function()
			local means = self.actor:forward(Tensor.new({ normalized })):toTable()
			action = table.create(c.actionDim)
			for d = 1, c.actionDim do
				local ls = math.clamp(self.logStd:at(d), c.logStdMin, c.logStdMax)
				action[d] = math.tanh(means[d] + rl.gaussianNoise() * math.exp(ls))
			end
		end)
	end

	self.pending = { obs = normalized, action = action }
	return action
end

function SAC.actGreedy(self: any, obs: {number}): {number}
	local c = self.config
	local normalized = if self.obsNorm then self.obsNorm:normalize(obs) else obs
	local action
	Config.noGrad(function()
		local means = self.actor:forward(Tensor.new({ normalized })):toTable()
		action = table.create(c.actionDim)
		for d = 1, c.actionDim do
			action[d] = math.tanh(means[d])
		end
	end)
	return action
end

--- Complete the transition. Only `terminated` marks a state as having no
--- future; `truncated` is stored as non-terminal, because the next state's
--- value is still real.
function SAC.observe(self: any, reward: number, terminated: boolean, truncated: boolean, nextObs: {number})
	local p = self.pending
	if not p then
		error("SAC.observe called before act", 2)
	end
	local normalizedNext = if self.obsNorm then self.obsNorm:normalize(nextObs) else nextObs
	self.buffer:add(p.obs, p.action, reward, normalizedNext, terminated)
	self.pending = nil

	if self.steps % self.config.updateEvery == 0 then
		self:update()
	end
end

-- Sample an action with its log-probability, reparameterised so the gradient
-- of Q(s,a) reaches the actor. The tanh correction is not optional: without
-- it the temperature tunes against an entropy it cannot measure.
function SAC.sampleWithLogProb(self: any, states: Tensor, count: number): (Tensor, Tensor)
	local c = self.config
	local means = self.actor:forward(states)
	local logStd = Tensor.clamp(self.logStd, c.logStdMin, c.logStdMax)
	local std = Tensor.exp(logStd)

	local noiseRows = table.create(count)
	for i = 1, count do
		local row = table.create(c.actionDim)
		for d = 1, c.actionDim do row[d] = rl.gaussianNoise() end
		noiseRows[i] = row
	end
	local noise = Tensor.new(noiseRows)

	local u = Tensor.add(means, Tensor.mul(noise, std))
	local action = Tensor.tanh(u)

	local normalLogp = Tensor.sub(
		Tensor.mul(Tensor.mul(noise, noise), -0.5),
		Tensor.add(logStd, 0.5 * LOG_2PI)
	)
	local correction = Tensor.log(
		Tensor.add(Tensor.sub(Tensor.new(1), Tensor.mul(action, action)), 1e-6))

	return action, Tensor.sum(Tensor.sub(normalLogp, correction), 2)
end

function SAC.update(self: any)
	local c = self.config
	if not self.buffer:ready(math.max(c.batch, c.warmup)) then return end

	local batch = self.buffer:sample(c.batch)
	local S, A, S2 = batch.obs, batch.action, batch.nextObs
	local alpha = self:alpha()

	-- 1. critic target, no gradients
	local targets = table.create(c.batch)
	Config.noGrad(function()
		local nextAction, nextLogp = self:sampleWithLogProb(S2, c.batch)
		local q1n = self.q1Target.forward(S2, nextAction):toTable()
		local q2n = self.q2Target.forward(S2, nextAction):toTable()
		local lp = nextLogp:toTable()
		for i = 1, c.batch do
			local softValue = math.min(q1n[i], q2n[i]) - alpha * lp[i]
			targets[i] = batch.reward[i] + c.gamma * (1 - batch.done[i]) * softValue
		end
	end)
	local Y = Tensor.new(targets, { c.batch })

	-- 2. critics
	self.q1Opt:zeroGrad()
	self.q2Opt:zeroGrad()
	local q1 = Tensor.reshape(self.q1.forward(S, A), { c.batch })
	local q2 = Tensor.reshape(self.q2.forward(S, A), { c.batch })
	local d1 = Tensor.sub(q1, Y)
	local d2 = Tensor.sub(q2, Y)
	local criticLoss = Tensor.add(Tensor.mean(Tensor.mul(d1, d1)), Tensor.mean(Tensor.mul(d2, d2)))
	criticLoss:backward()
	self.q1Opt:clipGradNorm(c.maxGradNorm)
	self.q2Opt:clipGradNorm(c.maxGradNorm)
	self.q1Opt:step()
	self.q2Opt:step()

	-- 3. actor: gradient flows actor -> sampled action -> critic -> loss
	self.actorOpt:zeroGrad()
	local newAction, logp = self:sampleWithLogProb(S, c.batch)
	local qa = Tensor.reshape(self.q1.forward(S, newAction), { c.batch })
	local qb = Tensor.reshape(self.q2.forward(S, newAction), { c.batch })
	local minQ = Tensor.mul(
		Tensor.sub(Tensor.add(qa, qb), Tensor.abs(Tensor.sub(qa, qb))), 0.5)
	local actorLoss = Tensor.mean(Tensor.sub(Tensor.mul(logp, alpha), minQ))
	actorLoss:backward()
	self.actorOpt:clipGradNorm(c.maxGradNorm)
	self.actorOpt:step()
	-- the actor's backward also deposited gradients in the critics; they are
	-- cleared by q1Opt/q2Opt:zeroGrad at the top of the next update

	-- 4. temperature: if the policy is more certain than the target, alpha
	-- rises and pushes exploration back up. This is why SAC needs no
	-- exploration schedule.
	self.alphaOpt:zeroGrad()
	local lpFlat = logp:toTable()
	local meanLogp = 0
	for i = 1, c.batch do meanLogp += lpFlat[i] end
	meanLogp /= c.batch

	Tensor.sum(Tensor.mul(self.logAlpha, -(meanLogp + c.targetEntropy))):backward()
	self.alphaOpt:step()
	self.logAlpha:setAt(1, math.clamp(self.logAlpha:at(1), -6, 2))

	for d = 1, c.actionDim do
		self.logStd:setAt(d, math.clamp(self.logStd:at(d), c.logStdMin, c.logStdMax))
	end

	-- 5. drift the targets
	rl.polyak(self.q1TargetParams, self.q1Params, c.tau)
	rl.polyak(self.q2TargetParams, self.q2Params, c.tau)

	self.updates += 1
	self.stats = {
		alpha = alpha,
		criticLoss = criticLoss:item(),
		actorLoss = actorLoss:item(),
		meanLogp = meanLogp,
		updates = self.updates,
	}
end

-- ==========================================================================
-- DQN
-- ==========================================================================

local DQN = {}
DQN.__index = DQN
algorithms.DQN = DQN

DQN.defaults = {
	obsSize = 8,
	actionCount = 4,
	hidden = 64,
	depth = 2,

	gamma = 0.99,
	lr = 1e-3,
	batch = 64,
	bufferSize = 50000,
	warmup = 1000,
	updateEvery = 4,
	targetSync = 500,       -- hard target copy every N updates
	maxGradNorm = 10,

	epsilonStart = 1.0,
	epsilonEnd = 0.05,
	epsilonDecay = 20000,   -- steps to reach epsilonEnd

	doubleDQN = true,
	normalizeObs = true,
}

--- Create a DQN agent. Discrete actions only.
function DQN.new(config: any?)
	local c = withDefaults(config, DQN.defaults)
	local self = setmetatable({ config = c, kind = "dqn" }, DQN)

	self.q = mlp(c.obsSize, c.hidden, c.actionCount, c.depth)
	self.qTarget = mlp(c.obsSize, c.hidden, c.actionCount, c.depth)
	rl.hardCopy(self.qTarget:parameters(), self.q:parameters())

	self.opt = optim.Adam(self.q:parameters(), c.lr)
	self.criterion = nn.HuberLoss(1)   -- far less sensitive to outliers than MSE

	self.buffer = rl.ReplayBuffer.new(c.bufferSize, c.obsSize, 1)
	self.obsNorm = if c.normalizeObs then rl.RunningNorm.new(c.obsSize) else nil

	self.steps = 0
	self.updates = 0
	self.pending = nil
	self.stats = {}
	return self
end

--- Current exploration rate, decayed linearly from epsilonStart to epsilonEnd.
function DQN.epsilon(self: any): number
	local c = self.config
	local progress = math.min(self.steps / c.epsilonDecay, 1)
	return c.epsilonStart + (c.epsilonEnd - c.epsilonStart) * progress
end

--- Epsilon-greedy action, as a 1-based action index.
function DQN.act(self: any, obs: {number}): number
	local c = self.config
	self.steps += 1
	local normalized = if self.obsNorm then self.obsNorm:step(obs) else obs

	local action
	if Config.random() < self:epsilon() then
		action = Config.randomInt(c.actionCount)
	else
		Config.noGrad(function()
			action = Tensor.argmax(self.q:forward(Tensor.new({ normalized })))
		end)
	end

	self.pending = { obs = normalized, action = action }
	return action
end

function DQN.actGreedy(self: any, obs: {number}): number
	local normalized = if self.obsNorm then self.obsNorm:normalize(obs) else obs
	local action
	Config.noGrad(function()
		action = Tensor.argmax(self.q:forward(Tensor.new({ normalized })))
	end)
	return action
end

function DQN.observe(self: any, reward: number, terminated: boolean, truncated: boolean, nextObs: {number})
	local p = self.pending
	if not p then
		error("DQN.observe called before act", 2)
	end
	local normalizedNext = if self.obsNorm then self.obsNorm:normalize(nextObs) else nextObs
	-- only `terminated` means the state has no future
	self.buffer:add(p.obs, { p.action }, reward, normalizedNext, terminated)
	self.pending = nil

	if self.steps % self.config.updateEvery == 0 then
		self:update()
	end
end

--[[
	One gradient step.

	DOUBLE DQN by default. Plain DQN uses the target network both to pick the
	next action and to value it, so any overestimation in the target is
	selected FOR — max over noisy estimates skews high, and the bias
	compounds. Double DQN picks with the online network and values with the
	target, which decorrelates the two and removes most of the bias.
]]
function DQN.update(self: any)
	local c = self.config
	if not self.buffer:ready(math.max(c.batch, c.warmup)) then return end

	local batch = self.buffer:sample(c.batch)
	local S, S2 = batch.obs, batch.nextObs

	local targets = table.create(c.batch)
	Config.noGrad(function()
		local nextQTarget = self.qTarget:forward(S2):toTable()
		local nextQOnline = if c.doubleDQN then self.q:forward(S2):toTable() else nextQTarget

		for i = 1, c.batch do
			local base = (i - 1) * c.actionCount
			-- pick with the online network
			local bestIndex, best = 1, nextQOnline[base + 1]
			for a = 2, c.actionCount do
				if nextQOnline[base + a] > best then
					best, bestIndex = nextQOnline[base + a], a
				end
			end
			-- value with the target network
			local nextValue = nextQTarget[base + bestIndex]
			targets[i] = batch.reward[i] + c.gamma * (1 - batch.done[i]) * nextValue
		end
	end)

	self.opt:zeroGrad()

	local qAll = self.q:forward(S)
	-- gather the Q of the action actually taken; one graph node regardless
	-- of batch size
	local takenIndices = table.create(c.batch)
	for i = 1, c.batch do
		takenIndices[i] = batch.actionRows[i][1]
	end
	local taken = F.gather(qAll, 2, takenIndices)

	local loss = self.criterion:forward(taken, Tensor.new(targets, { c.batch }))
	loss:backward()
	self.opt:clipGradNorm(c.maxGradNorm)
	self.opt:step()

	self.updates += 1
	if self.updates % c.targetSync == 0 then
		rl.hardCopy(self.qTarget:parameters(), self.q:parameters())
	end

	self.stats = {
		loss = loss:item(),
		epsilon = self:epsilon(),
		updates = self.updates,
	}
end


-- ==========================================================================
-- RECURRENT PPO
-- ==========================================================================

local RecurrentPPO = {}
RecurrentPPO.__index = RecurrentPPO
algorithms.RecurrentPPO = RecurrentPPO

--[[
	PPO with memory, for partially observable tasks — anything where the
	observation does not contain everything the agent needs, so it has to
	remember. Occlusion, a target that goes out of view, a signal shown once
	at the start of an episode.

	THREE THINGS MAKE THIS DIFFERENT FROM FEEDFORWARD PPO, and getting any
	of them wrong produces plausible gradients on a broken objective:

	1. MINIBATCHES ARE SEQUENCES, NOT TIMESTEPS. Feedforward PPO shuffles
	   individual transitions freely because each is independent. Here a
	   timestep only means anything given the hidden state that preceded it,
	   so the rollout is chunked into fixed-length sequences and whole
	   sequences are shuffled. Shuffling timesteps would destroy the very
	   thing being learned.

	2. THE HIDDEN STATE AT EACH SEQUENCE START IS STORED AND REPLAYED. The
	   update cannot recompute it — the parameters have changed since. It is
	   recorded during the rollout and fed back in as a constant.

	3. STATE IS MASKED AT EPISODE BOUNDARIES. A sequence may straddle a
	   reset, and memory must not leak from one episode into the next. The
	   mask multiplies the state by zero on those steps, which is
	   differentiable and so does not break the graph — resetting in place
	   would.

	The trunk is shared between actor and critic. Memory is expensive to
	learn and duplicating it doubles that cost for no benefit; the value
	gradient flowing into the trunk is standard and helps it.
]]

RecurrentPPO.defaults = {
	obsSize = 8,
	actionDim = 2,
	actionType = "continuous",
	hidden = 64,
	cell = "gru",             -- "gru", "lstm" or "rnn"
	layers = 1,

	horizon = 512,
	seqLen = 16,              -- BPTT window; must divide horizon
	epochs = 4,
	seqPerBatch = 8,          -- sequences per minibatch

	gamma = 0.99,
	lambda = 0.95,
	clip = 0.2,
	entropyCoef = 0.005,
	valueCoef = 0.5,
	maxGradNorm = 0.5,

	lr = 3e-4,

	initLogStd = -0.7,
	logStdMin = -2.5,
	logStdMax = 0.5,

	normalizeObs = true,
	normalizeAdvantages = true,
}

function RecurrentPPO.new(config: any?)
	local c = withDefaults(config, RecurrentPPO.defaults)
	if c.horizon % c.seqLen ~= 0 then
		error(("RecurrentPPO: horizon %d must be divisible by seqLen %d")
			:format(c.horizon, c.seqLen), 2)
	end

	local self = setmetatable({ config = c, kind = "recurrentPPO" }, RecurrentPPO)
	self.continuous = c.actionType == "continuous"

	local builder = if c.cell == "lstm" then rnn.LSTM
		elseif c.cell == "rnn" then rnn.RNN
		else rnn.GRU
	self.trunk = builder(c.obsSize, c.hidden, c.layers)

	self.actorHead = nn.Linear(c.hidden, c.actionDim)
	self.actorHead:scaleWeights(0.01)
	self.criticHead = nn.Linear(c.hidden, 1)

	local params = collectParams(self.trunk, self.actorHead, self.criticHead)
	if self.continuous then
		local init = table.create(c.actionDim, c.initLogStd)
		self.logStd = Tensor.new(init, { c.actionDim }, true)
		table.insert(params, self.logStd)
	end
	self.opt = optim.Adam(params, c.lr)

	self.obsNorm = if c.normalizeObs then rl.RunningNorm.new(c.obsSize) else nil
	self.state = rnn.zeroState(self.trunk, 1)

	self:clearBuffer()
	self.pending = nil
	self.stats = {}
	self.updates = 0
	return self
end

function RecurrentPPO.clearBuffer(self: any)
	self.buf = {
		obs = {}, action = {}, logp = {}, reward = {},
		value = {}, done = {}, bootstrap = {}, state = {},
		n = 0,
	}
end

--- Reset memory. Call between unrelated episodes if you drive the agent
--- manually; observe() does it automatically at episode boundaries.
function RecurrentPPO.resetState(self: any)
	self.state = rnn.zeroState(self.trunk, 1)
end

-- Hidden state as plain Lua arrays, so it can be stored per step and rebuilt
-- into a batch later without holding graph references alive.
local function stateToTables(state: any): {{number}}
	local out = table.create(#state)
	for i, t in state do
		out[i] = t:toTable()
	end
	return out
end

function RecurrentPPO.act(self: any, obs: {number}): any
	local c = self.config
	local normalized = if self.obsNorm then self.obsNorm:step(obs) else obs

	-- the state ENTERING this step is what the update will need to replay it
	local stateIn = stateToTables(self.state)

	local action, logp, value
	Config.noGrad(function()
		local x = Tensor.new({ normalized })
		local h, newState = self.trunk:forward(x, self.state)
		self.state = newState

		value = self.criticHead:forward(h):item()

		if self.continuous then
			local means = self.actorHead:forward(h):toTable()
			action = table.create(c.actionDim)
			logp = 0
			for d = 1, c.actionDim do
				local ls = math.clamp(self.logStd:at(d), c.logStdMin, c.logStdMax)
				local std = math.exp(ls)
				local a = means[d] + rl.gaussianNoise() * std
				action[d] = a
				local z = (a - means[d]) / std
				logp += -0.5 * z * z - ls - 0.5 * LOG_2PI
			end
		else
			local dist = rl.Categorical.new(self.actorHead:forward(h))
			action = dist:sample()[1]
			logp = dist.logProbs:at(action)
		end
	end)

	self.pending = {
		obs = normalized, action = action, logp = logp,
		value = value, state = stateIn,
	}
	return action
end

--- Greedy action. Advances memory, so use a separate agent or resetState()
--- if you interleave evaluation with training on one instance.
function RecurrentPPO.actGreedy(self: any, obs: {number}): any
	local normalized = if self.obsNorm then self.obsNorm:normalize(obs) else obs
	local action
	Config.noGrad(function()
		local h, newState = self.trunk:forward(Tensor.new({ normalized }), self.state)
		self.state = newState
		local out = self.actorHead:forward(h)
		action = if self.continuous then out:toTable() else Tensor.argmax(out)
	end)
	return action
end

function RecurrentPPO.valueOf(self: any, obs: {number}): number
	local normalized = if self.obsNorm then self.obsNorm:normalize(obs) else obs
	local v
	Config.noGrad(function()
		local h = self.trunk:forward(Tensor.new({ normalized }), self.state)
		v = self.criticHead:forward(h):item()
	end)
	return v
end

function RecurrentPPO.observe(self: any, reward: number, terminated: boolean, truncated: boolean, nextObs: {number}?)
	local p = self.pending
	if not p then
		error("RecurrentPPO.observe called before act", 2)
	end

	local bootstrap = 0
	if truncated and not terminated then
		if not nextObs then
			error("RecurrentPPO.observe: a truncated step needs nextObs to bootstrap from", 2)
		end
		bootstrap = self.config.gamma * self:valueOf(nextObs)
	end

	local b = self.buf
	local i = b.n + 1
	b.obs[i] = p.obs
	b.action[i] = p.action
	b.logp[i] = p.logp
	b.reward[i] = reward
	b.value[i] = p.value
	b.done[i] = if (terminated or truncated) then 1 else 0
	b.bootstrap[i] = bootstrap
	b.state[i] = p.state
	b.n = i

	self.pending = nil

	-- memory must not leak across an episode boundary
	if terminated or truncated then
		self:resetState()
	end
end

function RecurrentPPO.ready(self: any): boolean
	return self.buf.n >= self.config.horizon
end

--- Rebuild a batched state tensor from the per-step Lua arrays at the given
--- sequence starts.
local function batchState(buf: any, starts: {number}, componentCount: number, hidden: number): {any}
	local out = table.create(componentCount)
	for comp = 1, componentCount do
		local rows = table.create(#starts)
		for k, start in ipairs(starts) do
			rows[k] = buf.state[start][comp]
		end
		out[comp] = Tensor.new(rows)
	end
	return out
end

function RecurrentPPO.update(self: any, finalObs: {number}): any
	local c = self.config
	local b = self.buf
	local n = b.n
	if n == 0 then return {} end

	local lastValue = self:valueOf(finalObs)
	local advantages, returns = rl.gae(
		b.reward, b.value, b.done, b.bootstrap, lastValue, c.gamma, c.lambda)
	local ev = rl.explainedVariance(b.value, returns)
	if c.normalizeAdvantages then
		rl.normalizeAdvantages(advantages)
	end

	local L = c.seqLen
	local seqCount = n // L
	local componentCount = self.trunk.stateCount

	local order = table.create(seqCount)
	for i = 1, seqCount do order[i] = i end

	local clipSum, klSum, entSum, count = 0, 0, 0, 0

	for _ = 1, c.epochs do
		-- shuffle SEQUENCES, never timesteps: a timestep is meaningless
		-- without the hidden state that preceded it
		for i = seqCount, 2, -1 do
			local j = Config.randomInt(i)
			order[i], order[j] = order[j], order[i]
		end

		local cursor = 1
		while cursor <= seqCount do
			local last = math.min(cursor + c.seqPerBatch - 1, seqCount)
			local starts = {}
			for k = cursor, last do
				table.insert(starts, (order[k] - 1) * L + 1)
			end

			local clip, kl, ent = self:updateSequences(starts, L, componentCount, advantages, returns)
			clipSum += clip; klSum += kl; entSum += ent; count += 1
			cursor = last + 1
		end
	end

	local totalReward = 0
	for i = 1, n do totalReward += b.reward[i] end

	self:clearBuffer()
	self:resetState()
	self.updates += count

	self.stats = {
		meanReward = totalReward / n,
		explainedVariance = ev,
		clipFraction = clipSum / count,
		approxKL = klSum / count,
		entropy = entSum / count,
		updates = self.updates,
	}
	return self.stats
end

function RecurrentPPO.updateSequences(self: any, starts: {number}, L: number, componentCount: number, advantages, returns): (number, number, number)
	local c = self.config
	local b = self.buf
	local M = #starts

	self.opt:zeroGrad()

	local state = batchState(b, starts, componentCount, c.hidden)

	local logStd, std
	if self.continuous then
		-- clamped inside the graph, so a bad update cannot collapse std
		logStd = Tensor.clamp(self.logStd, c.logStdMin, c.logStdMax)
		std = Tensor.exp(logStd)
	end

	local logpParts = table.create(L)
	local valueParts = table.create(L)
	local entropyTotal = nil

	-- targets gathered in the same (t, sequence) order the parts are built
	local oldLogps = table.create(M * L)
	local advList = table.create(M * L)
	local retList = table.create(M * L)
	local cursor = 0

	for t = 1, L do
		local obsRows = table.create(M)
		local actionRows = table.create(M)
		local discreteActions = table.create(M)
		local resetMask = table.create(M)

		for k, start in ipairs(starts) do
			local idx = start + t - 1
			obsRows[k] = b.obs[idx]
			if self.continuous then
				actionRows[k] = b.action[idx]
			else
				discreteActions[k] = b.action[idx]
			end
			-- a sequence may straddle an episode reset; zero the state
			-- carried in from the previous step when it was terminal
			resetMask[k] = { if t > 1 and b.done[idx - 1] == 1 then 0 else 1 }

			cursor += 1
			oldLogps[cursor] = b.logp[idx]
			advList[cursor] = advantages[idx]
			retList[cursor] = returns[idx]
		end

		if t > 1 then
			local mask = Tensor.new(resetMask)
			for comp = 1, componentCount do
				state[comp] = Tensor.mul(state[comp], mask)
			end
		end

		local h, newState = self.trunk:forward(Tensor.new(obsRows), state)
		state = newState

		valueParts[t] = Tensor.reshape(self.criticHead:forward(h), { M })

		if self.continuous then
			local means = self.actorHead:forward(h)
			local z = Tensor.div(Tensor.sub(Tensor.new(actionRows), means), std)
			local perDim = Tensor.sub(
				Tensor.mul(Tensor.mul(z, z), -0.5),
				Tensor.add(logStd, 0.5 * LOG_2PI)
			)
			logpParts[t] = Tensor.sum(perDim, 2)
		else
			local dist = rl.Categorical.new(self.actorHead:forward(h))
			logpParts[t] = dist:logProb(discreteActions)
			local e = dist:entropy()
			entropyTotal = if entropyTotal then Tensor.add(entropyTotal, e) else e
		end
	end

	local logp = F.cat(logpParts, 1)
	local values = F.cat(valueParts, 1)
	local total = M * L

	local OLD = Tensor.new(oldLogps, { total })
	local ADV = Tensor.new(advList, { total })
	local RET = Tensor.new(retList, { total })

	local logRatio = Tensor.sub(logp, OLD)
	local ratio = Tensor.exp(logRatio)
	local surr1 = Tensor.mul(ratio, ADV)
	local surr2 = Tensor.mul(Tensor.clamp(ratio, 1 - c.clip, 1 + c.clip), ADV)
	local objective = Tensor.mul(
		Tensor.sub(Tensor.add(surr1, surr2), Tensor.abs(Tensor.sub(surr1, surr2))), 0.5)
	local policyLoss = Tensor.neg(Tensor.mean(objective))

	local diff = Tensor.sub(values, RET)
	local valueLoss = Tensor.mean(Tensor.mul(diff, diff))

	local entropy = if self.continuous
		then Tensor.sum(logStd)
		else Tensor.div(entropyTotal :: any, L)

	local loss = Tensor.add(
		Tensor.sub(policyLoss, Tensor.mul(entropy, c.entropyCoef)),
		Tensor.mul(valueLoss, c.valueCoef)
	)
	loss:backward()

	self.opt:clipGradNorm(c.maxGradNorm)
	self.opt:step()

	if self.continuous then
		for d = 1, c.actionDim do
			self.logStd:setAt(d, math.clamp(self.logStd:at(d), c.logStdMin, c.logStdMax))
		end
	end

	local lr = logRatio:toTable()
	local clipped, klSum = 0, 0
	for i = 1, total do
		local r = math.exp(lr[i])
		if r < 1 - c.clip or r > 1 + c.clip then clipped += 1 end
		klSum += (r - 1) - lr[i]
	end
	return clipped / total, klSum / total, entropy:item()
end

-- ==========================================================================
-- SAVE / LOAD
-- ==========================================================================

--- Save every network in an agent. Optimiser state is not included; add it
--- with serialize.optimizerToTable if you intend to resume mid-run.
function algorithms.save(agent: any): any
	local out = { kind = agent.kind }

	if out.kind == "ppo" then
		out.actor = serialize.toTable(agent.actor)
		out.critic = serialize.toTable(agent.critic)
		if agent.logStd then out.logStd = agent.logStd:toTable() end
	elseif out.kind == "recurrentPPO" then
		out.trunk = serialize.toTable(agent.trunk)
		out.actorHead = serialize.toTable(agent.actorHead)
		out.criticHead = serialize.toTable(agent.criticHead)
		if agent.logStd then out.logStd = agent.logStd:toTable() end
	elseif out.kind == "sac" then
		out.actor = serialize.toTable(agent.actor)
		out.logStd = agent.logStd:toTable()
		out.logAlpha = agent.logAlpha:toTable()
	elseif out.kind == "dqn" then
		out.q = serialize.toTable(agent.q)
	else
		error(("algorithms.save: unknown agent kind %q"):format(tostring(out.kind)), 2)
	end

	if agent.obsNorm then
		out.obsNorm = agent.obsNorm:state()
	end
	return out
end

function algorithms.load(agent: any, state: any)
	if state.kind and agent.kind and state.kind ~= agent.kind then
		error(("algorithms.load: saved a %s agent, loading into a %s")
			:format(tostring(state.kind), tostring(agent.kind)), 2)
	end
	if state.actor then serialize.fromTable(agent.actor, state.actor) end
	if state.critic then serialize.fromTable(agent.critic, state.critic) end
	if state.trunk then serialize.fromTable(agent.trunk, state.trunk) end
	if state.actorHead then serialize.fromTable(agent.actorHead, state.actorHead) end
	if state.criticHead then serialize.fromTable(agent.criticHead, state.criticHead) end
	if state.q then
		serialize.fromTable(agent.q, state.q)
		rl.hardCopy(agent.qTarget:parameters(), agent.q:parameters())
	end
	if state.logStd and agent.logStd then
		for i, v in ipairs(state.logStd) do agent.logStd:setAt(i, v) end
	end
	if state.logAlpha and agent.logAlpha then
		agent.logAlpha:setAt(1, state.logAlpha[1])
	end
	if state.obsNorm and agent.obsNorm then
		agent.obsNorm:load(state.obsNorm)
	end
end

return algorithms
