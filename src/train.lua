--!strict
--!native
--!optimize 2
--[[
	Nano.train — the training loop, the environment protocol, and diagnostics.

	THE ENVIRONMENT PROTOCOL

	    env:reset()          -> obs
	    env:step(action)     -> obs, reward, terminated, truncated
	    env:seed(n)          -> ()        OPTIONAL but strongly recommended

	`seed` is optional only because not every task can support it. Without
	it, evaluation is not reproducible: each evaluation faces a different
	random draw, so scores cannot be compared across time and "keep the best
	weights" degenerates into "keep the luckiest evaluation".

	`terminated` and `truncated` are SEPARATE and both required.

	    terminated  the episode genuinely ended. The state has no future and
	                its value is zero. Death, goal reached, invalid state.
	    truncated   you stopped watching. A step limit, a timeout, an
	                epoch boundary. The state still has value.

	Collapsing them into one `done` is the single most common silent bug in
	RL. It teaches the critic that value is zero at every episode boundary,
	which corrupts the advantages of every step leading up to it — and it
	looks exactly like "the algorithm does not work on my task". The protocol
	makes the mistake unrepresentable rather than merely documented.

	WHY A TRAINER

	The loop itself carries rules that are not obvious and are easy to get
	wrong: yield on a TIME budget rather than a step count, update at the
	right moment for the algorithm in hand, evaluate greedily rather than
	from the exploring policy, and keep the best weights rather than the
	last. Writing it once here means every agent inherits all of that.
]]

local Config = require(script.Parent.Config)
local Tensor = require(script.Parent.Tensor)

local train = {}

-- ==========================================================================
-- ENVIRONMENT PROTOCOL
-- ==========================================================================

--- Verify an environment implements the protocol, and that one step behaves.
--- Call it once when wiring up a new task; it catches the mistakes that
--- otherwise show up as a mysteriously untrainable agent.
--- @return boolean, {string} -- ok, list of problems
function train.checkEnv(env: any, sampleAction: any): (boolean, {string})
	local problems = {}

	if type(env.reset) ~= "function" then
		table.insert(problems, "env:reset is missing")
	end
	if type(env.step) ~= "function" then
		table.insert(problems, "env:step is missing")
	end
	if #problems > 0 then
		return false, problems
	end

	local obs = env:reset()
	if type(obs) ~= "table" then
		table.insert(problems, "reset() must return an observation array")
		return false, problems
	end
	local obsSize = #obs
	for i = 1, obsSize do
		if type(obs[i]) ~= "number" then
			table.insert(problems, ("reset() observation element %d is not a number"):format(i))
			break
		end
	end

	local nextObs, reward, terminated, truncated = env:step(sampleAction)

	if type(nextObs) ~= "table" or #nextObs ~= obsSize then
		table.insert(problems, ("step() must return an observation of the same size as reset() (%d)"):format(obsSize))
	end
	if type(reward) ~= "number" then
		table.insert(problems, "step() must return a numeric reward as its second value")
	end
	if type(terminated) ~= "boolean" then
		table.insert(problems,
			"step() must return `terminated` as its third value — a boolean, not a number or nil")
	end
	if type(truncated) ~= "boolean" then
		table.insert(problems,
			"step() must return `truncated` as its fourth value. If you only have one `done` flag, "
				.. "decide which it is: a step limit is truncated, not terminated")
	end
	if terminated == true and truncated == true then
		table.insert(problems,
			"a step reported both terminated and truncated; they are mutually exclusive")
	end

	if type(env.seed) ~= "function" then
		table.insert(problems,
			"WARNING env:seed(n) is missing — evaluation scores will not be "
				.. "comparable across time, so Trainer keepBest cannot work reliably")
	end

	-- the warning is advisory; only hard protocol breaks fail the check
	local hard = {}
	for _, p in ipairs(problems) do
		if not string.find(p, "^WARNING") then
			table.insert(hard, p)
		end
	end
	return #hard == 0, problems
end

-- ==========================================================================
-- TRAINER
-- ==========================================================================

--[[
	How does this agent learn?

	Detected from the methods it exposes rather than read off a `kind` tag.
	A tag can go missing — a stale file, a hand-rolled agent — and the
	comparison then silently yields false, so the trainer collects thousands
	of transitions and never updates. Nothing errors; the policy just never
	moves. Capability detection cannot fail that way, and an agent that
	exposes no update path at all is rejected outright.

	  "rollout"   act -> observe -> ready() -> update(obs)   (PPO)
	  "internal"  act -> observe, which updates from replay  (SAC, DQN)
]]
function train.updateMode(agent: any): string
	for _, required in { "act", "observe", "actGreedy" } do
		if type(agent[required]) ~= "function" then
			error(("Trainer: the agent has no %s method — it does not implement the agent interface")
				:format(required), 3)
		end
	end

	local hasReady = type(agent.ready) == "function"
	local hasUpdate = type(agent.update) == "function"

	if hasReady and hasUpdate then
		return "rollout"
	end
	if hasUpdate then
		return "internal"
	end
	error("Trainer: the agent exposes neither ready()+update() nor update(); "
		.. "there is no way for it to learn", 3)
end

local Trainer = {}
Trainer.__index = Trainer
train.Trainer = Trainer

Trainer.defaults = {
	steps = 50000,           -- total environment steps
	timeBudget = 0.008,      -- seconds of work between yields
	logEvery = 5000,
	evalEvery = 0,           -- 0 disables periodic evaluation
	evalEpisodes = 20,
	evalMaxSteps = 500,
	evalEnv = nil,           -- strongly recommended; see Trainer.new
	evalSeed = 20240607,     -- fixed, so scores are comparable across time
	keepBest = true,         -- snapshot weights whenever evaluation improves
	verbose = true,
	onLog = nil,             -- optional (stats) -> ()
}

--[[
	Create a trainer.

	Pass `evalEnv` — a SECOND environment instance. Evaluating on the
	training environment resets it mid-rollout, so the loop resumes with an
	observation that no longer describes the environment's state, and every
	transition immediately after an evaluation is garbage. With no evalEnv
	the trainer resynchronises afterwards, which is correct but throws away
	the episode in progress.

	Evaluation reseeds via `env:seed(evalSeed)` when the environment supports
	it, so every evaluation faces the same episodes and scores are actually
	comparable. Without that, keepBest tracks luck rather than skill.
]]
function Trainer.new(agent: any, env: any, config: any?)
	local c = {}
	for k, v in Trainer.defaults do c[k] = v end
	if config then
		for k, v in config do c[k] = v end
	end

	return setmetatable({
		agent = agent,
		env = env,
		config = c,
		steps = 0,
		episodes = 0,
		episodeReturn = 0,
		returns = {},
		bestScore = -math.huge,
		bestWeights = nil,
		history = {},
	}, Trainer)
end

--[[
	Yield on a TIME budget, never a step count.

	A step that triggers a gradient update costs roughly a hundred times a
	plain environment step, so any fixed step count either stalls the server
	during update-heavy stretches or wastes most of the frame during cheap
	ones. Measuring elapsed time adapts automatically.
]]
local function budgetKeeper(seconds: number)
	local deadline = os.clock() + seconds
	return function()
		if os.clock() >= deadline then
			task.wait()
			deadline = os.clock() + seconds
		end
	end
end

--[[
	Greedy evaluation, with exploration off. A policy that only looks
	competent while adding noise has not learned much.

	Runs on evalEnv when one was given, and reseeds first when the
	environment supports it, so repeated evaluations are directly
	comparable. Returns the score and whether it was deterministic — an
	incomparable score should not drive keepBest.
]]
function Trainer.evaluate(self: any, episodes: number?, maxSteps: number?): (number, boolean)
	local c = self.config
	local count = episodes or c.evalEpisodes
	local limit = maxSteps or c.evalMaxSteps
	local breathe = budgetKeeper(c.timeBudget)

	local env = c.evalEnv or self.env
	local deterministic = type(env.seed) == "function"
	if deterministic then
		env:seed(c.evalSeed)
	end

	local total = 0
	for _ = 1, count do
		local obs = env:reset()
		for _ = 1, limit do
			local action = self.agent:actGreedy(obs)
			local nextObs, reward, terminated, truncated = env:step(action)
			total += reward
			obs = nextObs
			breathe()
			if terminated or truncated then break end
		end
	end
	return total / count, deterministic
end

--- Run training to completion. Returns a summary table.
function Trainer.run(self: any): any
	local c = self.config
	local agent = self.agent
	local env = self.env
	local breathe = budgetKeeper(c.timeBudget)

	local mode = train.updateMode(agent)
	local isOnPolicy = mode == "rollout"
	self.updatesFired = 0
	local obs = env:reset()
	local startClock = os.clock()
	local windowReturn, windowEpisodes = 0, 0

	while self.steps < c.steps do
		local action = agent:act(obs)
		local nextObs, reward, terminated, truncated = env:step(action)
		agent:observe(reward, terminated, truncated, nextObs)

		self.steps += 1
		self.episodeReturn += reward
		obs = nextObs

		if terminated or truncated then
			self.episodes += 1
			windowReturn += self.episodeReturn
			windowEpisodes += 1
			table.insert(self.returns, self.episodeReturn)
			self.episodeReturn = 0
			obs = env:reset()
		end

		-- PPO accumulates a rollout and updates when full; SAC and DQN
		-- update from their replay buffer inside observe()
		if isOnPolicy and agent:ready() then
			agent:update(obs)
			self.updatesFired += 1
		end

		if c.logEvery > 0 and self.steps % c.logEvery == 0 then
			local mean = if windowEpisodes > 0 then windowReturn / windowEpisodes else self.episodeReturn
			local entry = {
				steps = self.steps,
				episodes = self.episodes,
				meanReturn = mean,
				elapsed = os.clock() - startClock,
				stats = agent.stats,
			}
			table.insert(self.history, entry)
			windowReturn, windowEpisodes = 0, 0

			if c.verbose then
				self:printLog(entry)
			end
			if c.onLog then
				c.onLog(entry)
			end
		end

		if c.evalEvery > 0 and self.steps % c.evalEvery == 0 then
			local score, deterministic = self:evaluate()

			if not deterministic and not self.warnedNondeterministic then
				self.warnedNondeterministic = true
				warn("[nano.train] evaluation is not reproducible: give the "
					.. "environment a seed(n) method, or pass evalEnv. Until "
					.. "then keepBest tracks the luckiest evaluation, not the "
					.. "best policy.")
			end

			if score > self.bestScore then
				self.bestScore = score
				if c.keepBest then
					self.bestWeights = self:snapshot()
				end
			end
			if c.verbose then
				print(("    eval @ %d steps: %.4f  (best %.4f)"):format(
					self.steps, score, self.bestScore))
			end

			-- evaluating on the training env reset it mid-rollout; the loop's
			-- `obs` is now stale, so start a fresh episode rather than feed a
			-- mismatched transition into the buffer
			if not c.evalEnv then
				obs = env:reset()
			end
		end

		breathe()
	end

	local total = os.clock() - startClock

	--[[
		A rollout agent that never updated has been fed thousands of
		transitions and learned from none of them. Nothing errors when this
		happens — the loop runs, the returns log, the policy simply never
		moves — so it is checked explicitly.
	]]
	if isOnPolicy and self.updatesFired == 0 then
		warn(("[nano.train] %d steps collected and NOT ONE update ran. The agent's "
			.. "rollout buffer never reported ready — check that horizon (%s) is "
			.. "not larger than the number of steps, and that agent:ready() exists.")
			:format(self.steps, tostring(agent.config and agent.config.horizon)))
	end

	--[[
		An internal-update agent (SAC, DQN) runs its updates inside observe(),
		so the trainer never sees them and updatesFired stays at zero.
		Reporting that as "0 gradient updates" is simply wrong — SAC does
		thousands. Prefer the agent's own counter when it keeps one.
	]]
	local updates = self.updatesFired
	if not isOnPolicy and type(agent.updates) == "number" then
		updates = agent.updates
	end

	return {
		steps = self.steps,
		updates = updates,
		episodes = self.episodes,
		elapsed = total,
		stepsPerSecond = self.steps / total,
		bestScore = self.bestScore,
		history = self.history,
	}
end

function Trainer.printLog(self: any, entry: any)
	local s = entry.stats or {}
	local extra = ""
	if s.explainedVariance then
		extra = (" | EV %5.2f clip %.2f KL %.4f"):format(
			s.explainedVariance, s.clipFraction or 0, s.approxKL or 0)
	elseif s.alpha then
		extra = (" | alpha %.3f"):format(s.alpha)
	elseif s.epsilon then
		extra = (" | eps %.3f loss %.4f"):format(s.epsilon, s.loss or 0)
	end
	print(("    %7d steps | %5d eps | return %8.3f%s | %6.1fs"):format(
		entry.steps, entry.episodes, entry.meanReturn, extra, entry.elapsed))
end

--[[
	Flat weights of every network the agent owns.

	The field list must cover every agent shape. A name missing from it does
	not error — snapshot simply returns fewer entries, restoreBest reports
	success, and nothing is restored. That is the silent-no-op failure this
	library keeps trying to eliminate, so an agent that yields no networks at
	all is treated as a bug rather than an empty snapshot.
]]
local SNAPSHOT_FIELDS = {
	"actor", "critic", "q",              -- PPO, SAC, DQN
	"trunk", "actorHead", "criticHead",  -- RecurrentPPO
}

function Trainer.snapshot(self: any): any
	local agent = self.agent
	local out = {}
	local found = 0
	for _, field in SNAPSHOT_FIELDS do
		local module = agent[field]
		if module and type(module.getFlat) == "function" then
			out[field] = module:getFlat()
			found += 1
		end
	end
	if found == 0 then
		error("Trainer.snapshot: the agent exposes no known network fields, so "
			.. "there is nothing to snapshot and keepBest would silently do "
			.. "nothing. Add its field names to SNAPSHOT_FIELDS in train.lua.", 2)
	end
	return out
end

--- Restore a snapshot taken by snapshot().
function Trainer.restore(self: any, snap: any)
	local agent = self.agent
	local restored = 0
	for field, values in snap do
		local module = agent[field]
		if module and type(module.setFlat) == "function" then
			module:setFlat(values)
			restored += 1
		end
	end
	if restored == 0 then
		error("Trainer.restore: nothing was restored — the snapshot does not "
			.. "match this agent", 2)
	end
end

--- Put the best-evaluated weights back into the agent. Training past the
--- optimum degrades a model, so finishing with the final weights is usually
--- worse than finishing with the best ones.
function Trainer.restoreBest(self: any): boolean
	if not self.bestWeights then return false end
	self:restore(self.bestWeights)
	return true
end

-- ==========================================================================
-- DIAGNOSTICS
-- ==========================================================================

--[[
	Almost every failure in this library has been silent: a wrong log-prob, a
	stale gradient, an off-by-one bias, a collapsed std. None of them error.
	This looks for the fingerprints they leave in the parameters.
]]

--- Inspect a model's parameters and gradients for the usual silent failures.
--- @return table -- report; `report.problems` is empty when healthy
function train.diagnose(model: any, options: any?): any
	local opts = options or {}
	local params = model:parameters()

	local report = {
		parameters = 0,
		tensors = #params,
		layers = {},
		problems = {},
		weightNorm = 0,
		gradNorm = 0,
		nonFinite = 0,
		zeroGradTensors = 0,
		noGradTensors = 0,
	}

	local weightSq, gradSq = 0, 0

	for i, p in ipairs(params) do
		local n = p:numel()
		report.parameters += n

		local wSq, maxAbs, nonFinite = 0, 0, 0
		for j = 1, n do
			local v = p:at(j)
			if v ~= v or v == math.huge or v == -math.huge then
				nonFinite += 1
			else
				wSq += v * v
				local a = if v < 0 then -v else v
				if a > maxAbs then maxAbs = a end
			end
		end

		local gSq, gradMax = 0, 0
		local hasGrad = p.grad ~= nil
		if hasGrad then
			for j = 1, n do
				local g = buffer.readf64(p.grad :: buffer, (j - 1) * 8)
				if g ~= g or g == math.huge or g == -math.huge then
					nonFinite += 1
				else
					gSq += g * g
					local a = if g < 0 then -g else g
					if a > gradMax then gradMax = a end
				end
			end
			if gSq == 0 then
				report.zeroGradTensors += 1
			end
		else
			report.noGradTensors += 1
		end

		weightSq += wSq
		gradSq += gSq
		report.nonFinite += nonFinite

		table.insert(report.layers, {
			index = i,
			shape = table.clone(p.shape),
			count = n,
			weightNorm = math.sqrt(wSq),
			weightMax = maxAbs,
			gradNorm = math.sqrt(gSq),
			gradMax = gradMax,
			nonFinite = nonFinite,
		})

		if nonFinite > 0 then
			table.insert(report.problems, ("tensor %d (%s) holds %d non-finite values — training has diverged")
				:format(i, table.concat(p.shape, "x"), nonFinite))
		end
		if maxAbs > (opts.weightLimit or 1e3) then
			table.insert(report.problems, ("tensor %d has a weight of magnitude %.3g — exploding")
				:format(i, maxAbs))
		end
		if hasGrad and gradMax > (opts.gradLimit or 1e3) then
			table.insert(report.problems, ("tensor %d has a gradient of magnitude %.3g — clip harder")
				:format(i, gradMax))
		end
	end

	report.weightNorm = math.sqrt(weightSq)
	report.gradNorm = math.sqrt(gradSq)

	if report.gradNorm == 0 and report.noGradTensors < #params then
		table.insert(report.problems,
			"every gradient is zero — backward may not have run, or zeroGrad ran after it")
	end
	if report.noGradTensors == #params then
		table.insert(report.problems,
			"no tensor has a gradient buffer — nothing has been backpropagated yet")
	end

	return report
end

--[[
	Fraction of hidden units that never activate on a sample of inputs.

	A dead ReLU outputs zero for every input it will ever see, so its
	gradient is zero forever and the unit is permanently wasted. A large dead
	fraction usually means the learning rate was too high early on. Tanh
	networks cannot die this way but can SATURATE, which is reported the same
	way: a unit pinned at +/-1 has a vanishing gradient.
]]
function train.activationReport(model: any, samples: {{number}}): any
	local layers = model.layers
	if not layers then
		return { supported = false }
	end

	local counts = {}
	local totals = {}

	Config.noGrad(function()
		local x = Tensor.new(samples)
		for i, layer in ipairs(layers) do
			x = layer:forward(x)
			-- only look after an activation, where saturation is meaningful
			local isActivation = layer.forward and not layer.weight
			if isActivation then
				local n = x:numel()
				local dead, saturated = 0, 0
				for j = 1, n do
					local v = x:at(j)
					if v == 0 then dead += 1 end
					local a = if v < 0 then -v else v
					if a > 0.99 then saturated += 1 end
				end
				counts[i] = { dead = dead, saturated = saturated }
				totals[i] = n
			end
		end
	end)

	local out = {}
	for i, c in counts do
		table.insert(out, {
			layer = i,
			deadFraction = c.dead / totals[i],
			saturatedFraction = c.saturated / totals[i],
		})
	end
	return { supported = true, layers = out }
end

--- Print a diagnose() report.
function train.printDiagnosis(report: any)
	print(("  %d parameters across %d tensors"):format(report.parameters, report.tensors))
	print(("  weight norm %.4f | gradient norm %.4f"):format(report.weightNorm, report.gradNorm))

	if report.nonFinite > 0 then
		print(("  %d NON-FINITE values"):format(report.nonFinite))
	end
	if report.zeroGradTensors > 0 then
		print(("  %d tensors have an all-zero gradient"):format(report.zeroGradTensors))
	end

	print("  per tensor:")
	for _, l in ipairs(report.layers) do
		print(("    %-12s n=%-6d |w| %8.4f (max %7.4f)  |g| %8.4f (max %7.4f)"):format(
			table.concat(l.shape, "x"), l.count, l.weightNorm, l.weightMax, l.gradNorm, l.gradMax))
	end

	if #report.problems > 0 then
		print("  PROBLEMS")
		for _, p in ipairs(report.problems) do
			print("    " .. p)
		end
	else
		print("  no problems detected")
	end
end

return train
