--!strict
--[[
	Nano.parallel — Actor-based worker pools.

	WHAT THIS IS AND IS NOT FOR

	Shipping a tensor between actors costs roughly 0.003 ms per element write.
	A {64,64} tensor is ~12 ms to transfer; a batch-32 training step is ~1.7 ms
	to compute. Parallelising a single gradient update is therefore strictly
	slower than not doing it.

	Actors pay off only when the work per message is large and the payload is
	small. That means:
	    yes  evolutionary populations (genome in, fitness scalar out)
	    yes  rollout collection across many environments
	    yes  hyperparameter and seed sweeps
	    yes  many independent NPCs each running their own small network
	    no   splitting one matmul, one minibatch, or one backward pass

	Servers expose roughly 2 worker threads; clients up to 8. Use more actors
	than threads so the scheduler can balance them.
]]

local Tensor = require(script.Parent.Tensor)
local Config = require(script.Parent.Config)
local serialize = require(script.Parent.serialize)

local parallel = {}

-- Reserved message topic. Workers bind it automatically in parallel.serve.
local SEED_TOPIC = "__nanoSeed"

local Pool = {}
Pool.__index = Pool
parallel.Pool = Pool

--- Create a pool of Actors, each running a clone of `workerScript`.
--- `workerScript` must be a Script whose top-level code calls parallel.serve().
--- @param workerScript Script -- template cloned into every actor
--- @param workerCount number? -- defaults to 8
--- @param parent Instance? -- where actors are parented, defaults to workerScript.Parent
--[[
	Containers where a server Script will never execute. Parenting a worker
	pool into one of these produces a pool that looks entirely healthy —
	the actors exist, the clones are Enabled, destroy works — while nothing
	ever runs and every job silently times out.

	It is an easy mistake because the template naturally lives beside the
	library in ReplicatedStorage, which is exactly where scripts do not run.
]]
local INERT_CONTAINERS = {
	ReplicatedStorage = true,
	ReplicatedFirst = true,
	ServerStorage = true,
	StarterGui = true,
	StarterPack = true,
	StarterPlayer = true,
	Lighting = true,
}

local function isInert(instance: Instance): (boolean, string?)
	local node: Instance? = instance
	while node and node ~= game do
		if INERT_CONTAINERS[node.Name] and node.Parent == game then
			return true, node.Name
		end
		node = node.Parent
	end
	return false
end

function Pool.new(workerScript: Script, workerCount: number?, parent: Instance?)
	local count = workerCount or 8

	--[[
		Default to ServerScriptService rather than the template's own parent.
		The template lives with the library, and inheriting its location is
		how a pool ends up somewhere its scripts cannot run.
	]]
	local home = parent or game:GetService("ServerScriptService")

	if not workerScript:IsA("Script") then
		error(("parallel.Pool: the worker template is a %s; it must be a Script. "
			.. "A ModuleScript has no Enabled property and would never be run by "
			.. "anything, so no handler would ever bind.")
			:format(workerScript.ClassName), 2)
	end

	local inert, where = isInert(home)
	if inert then
		error(("parallel.Pool: cannot parent workers into %s — server Scripts do "
			.. "not run there, so every job would time out. Pass an explicit "
			.. "parent such as ServerScriptService.")
			:format(where :: string), 2)
	end

	local self = setmetatable({
		actors = table.create(count),
		count = count,
		pending = 0,
		results = {},
		done = Instance.new("BindableEvent"),
		container = Instance.new("Folder"),
	}, Pool)

	self.container.Name = "NanoPool"
	self.container.Parent = home
	self.done.Parent = self.container

	for i = 1, count do
		local actor = Instance.new("Actor")
		actor.Name = "NanoWorker" .. i
		local clone = workerScript:Clone()
		clone.Enabled = true
		clone.Parent = actor
		actor.Parent = self.container
		self.actors[i] = actor
	end

	self.done.Event:Connect(function(jobId: number, payload: any)
		self.results[jobId] = payload
		self.pending -= 1
	end)

	-- cloned Scripts do not run their top-level code until Roblox schedules
	-- them, so the pool is not usable the instant it is constructed
	self:waitReady()

	-- After waitReady, never before: an Actor drops any message whose topic is
	-- not bound yet, and a seed dropped on the floor fails silently.
	local seed = Config.getSeed()
	if seed then
		self:seedWorkers(seed)
	end

	return self
end

--[[
	Block until every worker has bound its handlers, or the timeout expires.

	An Actor DROPS any message whose topic has no handler bound yet — there
	is no queue and no error on either side. A pool dispatched immediately
	after construction therefore sends every job into a worker that is not
	listening, and they vanish silently, surfacing only as a timeout with
	nothing to show for it.

	Called automatically by Pool.new.

	@return boolean -- true if every worker reported ready
]]
function Pool.waitReady(self: any, timeout: number?): boolean
	local deadline = os.clock() + (timeout or 5)

	local function readyCount(): number
		local n = 0
		for _, actor in ipairs(self.actors) do
			if actor:GetAttribute("NanoReady") then
				n += 1
			end
		end
		return n
	end

	while os.clock() < deadline do
		if readyCount() >= self.count then
			return true
		end
		task.wait()
	end

	warn(("[nano.parallel] only %d of %d workers bound handlers within %.0fs. "
		.. "The worker script must call parallel.serve(script:GetActor(), handlers) "
		.. "at its top level — check the output window for errors inside it.")
		:format(readyCount(), self.count, timeout or 5))
	return false
end

--- Send one job to a specific worker. Fire and forget.
function Pool.send(self: any, workerIndex: number, topic: string, ...)
	self.actors[((workerIndex - 1) % self.count) + 1]:SendMessage(topic, ...)
end

--[[
	Give each worker its own reproducible stream.

	Worker i is seeded with base + i, NOT with base. Seeding every worker
	identically would make each one draw the same exploration noise and the
	same environment layout, so a population of 8 would evaluate 8 copies of
	the same rollout — reproducible and useless.

	Called automatically by Pool.new when a seed is in force, so seeding the
	server before building the pool is enough.
]]
function Pool.seedWorkers(self: any, base: number)
	for i = 1, self.count do
		self.actors[i]:SendMessage(SEED_TOPIC, base + i)
	end
end

--- Send the same message to every worker.
function Pool.broadcast(self: any, topic: string, ...)
	for _, actor in self.actors do
		actor:SendMessage(topic, ...)
	end
end

--- Dispatch `jobs` across all workers and yield until every result is back.
--- Each job is a table of arguments; the worker receives (jobId, done, unpack(job)).
--- Returns results indexed by job number.
--- @param topic string -- message topic the worker is bound to
--- @param jobs {any} -- one entry per unit of work
--- @param timeout number? -- seconds before giving up, defaults to 30
function Pool.map(self: any, topic: string, jobs: {any}, timeout: number?): {any}
	local limit = timeout or 30
	self.results = {}
	self.pending = #jobs

	for i, job in ipairs(jobs) do
		local actor = self.actors[((i - 1) % self.count) + 1]
		actor:SendMessage(topic, i, self.done, job)
	end

	local deadline = os.clock() + limit
	while self.pending > 0 do
		if os.clock() > deadline then
			warn(("[nano.parallel] %d of %d jobs did not return within %ds"):format(
				self.pending, #jobs, limit))
			break
		end
		task.wait()
	end

	return self.results
end

--- Destroy every actor and the pool's container.
function Pool.destroy(self: any)
	self.container:Destroy()
	table.clear(self.actors)
end

--[[
	Bind topic handlers inside a worker script.

	THE ACTOR MUST BE PASSED IN. `script` is a per-script global: inside this
	ModuleScript it refers to the module itself, not to the worker that
	called in, so script:GetActor() here would always resolve against
	ReplicatedStorage and return nil. Only the caller's `script` refers to
	the worker, so only the caller can find its actor.

	Call at the top level, in the serial phase, after requiring everything
	the handlers need — require() is unavailable once desynchronized.

	    parallel.serve(script:GetActor(), {
	        EvaluateGenome = function(genome) ... end,
	    })

	@param actor Actor -- pass script:GetActor() from the worker
	@param handlers {[string]: (any) -> any}
]]
function parallel.serve(actor: Actor, handlers: {[string]: (any) -> any})
	-- catch the old single-argument call rather than failing obscurely
	if type(actor) == "table" and handlers == nil then
		error("parallel.serve now takes the actor first: "
			.. "parallel.serve(script:GetActor(), handlers)", 2)
	end

	if typeof(actor) ~= "Instance" or not actor:IsA("Actor") then
		error("parallel.serve: first argument must be an Actor. Pass "
			.. "script:GetActor() from the worker script — calling GetActor "
			.. "inside this module would resolve against the module itself.", 2)
	end

	if type(handlers) ~= "table" then
		error("parallel.serve: second argument must be a table of handlers", 2)
	end

	--[[
		Seeding across the actor boundary.

		Every Actor is a separate Lua VM, so it loads its OWN copy of Config
		with its own `rng = Random.new()` — unseeded. Config.seed on the server
		cannot reach a worker, which meant a "seeded" run was reproducible
		everywhere except inside the pool, and evolution and parallel rollouts
		were the two places that mattered most.

		Bound before the user handlers and under a reserved name, so a worker
		gets this for free without knowing it exists.
	]]
	actor:BindToMessageParallel(SEED_TOPIC, function(seed: number)
		Config.seed(seed)
	end)

	for topic, handler in handlers do
		if topic == SEED_TOPIC then
			error(("parallel.serve: %q is reserved for seed propagation"):format(SEED_TOPIC), 2)
		end
		actor:BindToMessageParallel(topic, function(jobId: number, done: BindableEvent, job: any)
			local ok, payload = pcall(handler, job)
			-- firing a BindableEvent touches the data model, so return to serial
			task.synchronize()
			if ok then
				done:Fire(jobId, payload)
			else
				warn(("[nano.parallel] job %d failed: %s"):format(jobId, tostring(payload)))
				done:Fire(jobId, nil)
			end
		end)
	end

	--[[
		Announce that handlers are bound.

		An Actor DROPS any message whose topic has no handler yet — there is
		no queue and no error on either side. Since a cloned Script does not
		run its top-level code until Roblox schedules it, a pool that
		dispatches immediately after construction sends every job into a
		worker that is not listening. Pool.new waits on this attribute.
	]]
	actor:SetAttribute("NanoReady", true)
end

--- Pack a model's weights into one base64 string.
--- One string crosses the actor boundary instead of thousands of table writes,
--- which is the difference between a viable transfer and a 12 ms one.
function parallel.packWeights(model: any): { shapes: {{number}}, data: string }
	return serialize.toBase64(model)
end

--- Load weights packed by packWeights into a structurally identical model.
function parallel.unpackWeights(model: any, payload: { shapes: {{number}}, data: string })
	return serialize.fromBase64(model, payload)
end

local Evolution = {}
Evolution.__index = Evolution
parallel.Evolution = Evolution

--- Population-based training across a worker pool.
--- Genomes are flat number arrays, so a whole individual is one array copy
--- rather than a structured transfer. This is the workload actors were built
--- for: seconds of work per message, a single scalar back.
--- @param pool Pool
--- @param genomeSize number -- model:numParameters()
--- @param populationSize number?
function Evolution.new(pool: any, genomeSize: number, populationSize: number?)
	local size = populationSize or 32
	local self = setmetatable({
		pool = pool,
		genomeSize = genomeSize,
		size = size,
		population = table.create(size),
		fitness = table.create(size, 0),
		generation = 0,
		sigma = 0.1,
		eliteFraction = 0.25,
	}, Evolution)

	for i = 1, size do
		local genome = table.create(genomeSize)
		for j = 1, genomeSize do
			genome[j] = (Config.random() * 2 - 1) * 0.5
		end
		self.population[i] = genome
	end

	return self
end

--- Evaluate the whole population in parallel, then produce the next generation.
--- The worker bound to `topic` receives a genome and returns a fitness number.
function Evolution.step(self: any, topic: string, timeout: number?): (number, number)
	local results = self.pool:map(topic, self.population, timeout)

	local best, total = -math.huge, 0
	for i = 1, self.size do
		local f = results[i] or -math.huge
		self.fitness[i] = f
		if f > best then best = f end
		total += if f > -math.huge then f else 0
	end

	self:evolve()
	self.generation += 1
	return best, total / self.size
end

--- Truncation selection with Gaussian mutation.
function Evolution.evolve(self: any)
	local order = table.create(self.size)
	for i = 1, self.size do order[i] = i end
	table.sort(order, function(a, b)
		return self.fitness[a] > self.fitness[b]
	end)

	local eliteCount = math.max(1, math.floor(self.size * self.eliteFraction))
	local next_ = table.create(self.size)

	for i = 1, eliteCount do
		next_[i] = table.clone(self.population[order[i]])
	end

	for i = eliteCount + 1, self.size do
		local parent = self.population[order[Config.randomInt(eliteCount)]]
		local child = table.create(self.genomeSize)
		for j = 1, self.genomeSize do
			child[j] = parent[j] + self:noise() * self.sigma
		end
		next_[i] = child
	end

	self.population = next_
end

function Evolution.noise(self: any): number
	return Config.gaussian()
end

--- The fittest genome of the last evaluated generation.
function Evolution.best(self: any): {number}
	local bestIndex, bestFitness = 1, -math.huge
	for i = 1, self.size do
		if self.fitness[i] > bestFitness then
			bestIndex, bestFitness = i, self.fitness[i]
		end
	end
	return self.population[bestIndex]
end

--- Number of parallel worker threads likely available. Servers expose about
--- two; clients scale with the device. Advisory only — use more actors than
--- threads regardless, so the scheduler can balance them.
function parallel.suggestedWorkers(): number
	return if game:GetService("RunService"):IsServer() then 8 else 16
end

return parallel
