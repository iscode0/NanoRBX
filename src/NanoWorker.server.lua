--!native
--!optimize 2
--[[
	NanoWorker — template Script, cloned into every Actor by parallel.Pool.

	Place it as a child of the Nano ModuleScript with Enabled = false. The pool
	clones it, enables the clone, and parents it to an Actor.

	Everything above parallel.serve() runs in the serial phase, which is the
	only place require() works. Build your models here; the handlers below run
	desynchronized and can only use what already exists.

	Edit the handlers for your task. The two below are working examples.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")

local nano = require(ReplicatedStorage.Nano)
local nn, parallel = nano.nn, nano.parallel

-- Each worker owns one model and reuses it for every job. Building a fresh
-- one per genome would allocate a full parameter set per evaluation.
local OBS_SIZE = 8
local ACTION_DIM = 2
local HIDDEN = 32

local model = nn.Sequential(
	nn.Linear(OBS_SIZE, HIDDEN), nn.Tanh(),
	nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
	nn.Linear(HIDDEN, ACTION_DIM), nn.Tanh()
)

local function simulate(steps: number): number
	local total = 0
	local obs = table.create(OBS_SIZE, 0)

	nano.noGrad(function()
		for _ = 1, steps do
			-- at() reads one element directly. toFlat() returns a BUFFER,
			-- not an array, so action[1] would error; toTable() would work
			-- but allocates a fresh table every step.
			local action = model:forward(nano.tensor({ obs })):at(1)
			-- replace with a real environment step
			total += action * 0.01
			for i = 1, OBS_SIZE do
				obs[i] = math.clamp(obs[i] + action * 0.1, -1, 1)
			end
		end
	end)

	return total
end

--[[
	The actor is passed explicitly because `script` inside parallel.lua would
	refer to that ModuleScript, not to this worker. Only this script's own
	`script` global can find the Actor it lives under.
]]
parallel.serve(script:GetActor(), {
	--- Load a genome, run an episode, return a fitness scalar.
	EvaluateGenome = function(genome: {number}): number
		model:setFlat(genome)
		return simulate(200)
	end,

	--- Load packed weights, collect a rollout, return trajectory arrays.
	CollectRollout = function(job: any): any
		parallel.unpackWeights(model, job.weights)

		local observations = table.create(job.steps)
		local rewards = table.create(job.steps)
		local obs = table.create(OBS_SIZE, 0)

		nano.noGrad(function()
			for t = 1, job.steps do
				local action = model:forward(nano.tensor({ obs })):at(1)
				observations[t] = table.clone(obs)
				rewards[t] = action * 0.01
				for i = 1, OBS_SIZE do
					obs[i] = math.clamp(obs[i] + action * 0.1, -1, 1)
				end
			end
		end)

		return { observations = observations, rewards = rewards }
	end,
})
