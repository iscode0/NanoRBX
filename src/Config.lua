--!strict
--!native
--!optimize 2
--[[
	Nano.Config — global switches and the shared random source.

	NATIVE because a Script Profiler export showed both noGrad and gaussian
	running interpreted while every other numeric module compiled. Config was
	the one hot module without the annotation: noGrad wraps every inference
	call in the library, and gaussian is sqrt + log + cos per sample, called
	once per element by randn and once per dimension by every exploration
	step.

	Kept dependency-free so every other module can require it without a cycle.

	WHY THE RNG LIVES HERE. Everything used to call math.random directly,
	which meant no run could be reproduced: a change that looked like an
	improvement might just have been a lucky seed, and a bug that appeared
	once might never appear again. One shared, seedable source fixes both.
	Seed it and a whole training run replays exactly.
]]

local Config = {}

--[[
	When false, operations skip building the autograd graph entirely: no
	_prev, no _backward closure, no gradient buffers. Inference and
	evolutionary methods never need gradients, and skipping the graph is a
	large speedup, not a micro-optimisation.
]]
Config.gradEnabled = true

function Config.isGradEnabled(): boolean
	return Config.gradEnabled
end

--[[
	Run fn with the graph disabled, then restore the PREVIOUS state rather
	than setting true. Nested noGrad calls are common once helper functions
	use it internally, and restoring on error means a throw inside fn cannot
	leave gradients globally disabled.
]]
function Config.noGrad<T>(fn: () -> T): T
	local previous = Config.gradEnabled
	Config.gradEnabled = false
	local ok, result = pcall(fn)
	Config.gradEnabled = previous
	if not ok then
		error(result, 2)
	end
	return result
end

-- ==========================================================================
-- RANDOM
-- ==========================================================================

local rng = Random.new()
local seeded = false
local currentSeed: number? = nil

--- Seed every random source in Nano. Pass nothing to reseed unpredictably.
--- Reproducibility is what lets you tell a real improvement from a lucky
--- seed — without it, two runs of the same code are not comparable.
function Config.seed(value: number?)
	if value then
		rng = Random.new(value)
		seeded = true
		currentSeed = value
	else
		rng = Random.new()
		seeded = false
		currentSeed = nil
	end
end

--- The seed in force, or nil if unseeded.
function Config.getSeed(): number?
	return currentSeed
end

function Config.isSeeded(): boolean
	return seeded
end

--- Uniform in [0, 1).
function Config.random(): number
	return rng:NextNumber()
end

--- Uniform in [lo, hi).
function Config.uniform(lo: number, hi: number): number
	return rng:NextNumber(lo, hi)
end

--- Uniform integer in [lo, hi], or [1, lo] when hi is omitted.
function Config.randomInt(lo: number, hi: number?): number
	if hi then
		return rng:NextInteger(lo, hi)
	end
	return rng:NextInteger(1, lo)
end

--[[
	One standard normal, via Box-Muller.

	The second value of the pair is discarded rather than cached. Caching it
	would halve the trig cost but makes the stream stateful in a way that
	breaks reproducibility across differently-shaped calls: the same seed
	would give different numbers depending on how many samples were drawn
	before. Determinism is worth more here than the saved cosine.
]]
function Config.gaussian(): number
	local u1, u2 = rng:NextNumber(), rng:NextNumber()
	return math.sqrt(-2 * math.log(u1 + 1e-12)) * math.cos(2 * math.pi * u2)
end

--- A private Random, for anything that needs its own reproducible stream
--- without disturbing the global one — an environment, a dataset split.
function Config.stream(seed: number): Random
	return Random.new(seed)
end

return Config
