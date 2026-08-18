--!strict
--!native
--!optimize 2
--[[
	Nano.data — datasets, batching and metrics.
]]

local Tensor = require(script.Parent.Tensor)
local Config = require(script.Parent.Config)

type Tensor = Tensor.Tensor

local data = {}

local t_create = table.create
local m_sqrt = math.sqrt

local Dataset = {}
Dataset.__index = Dataset
data.Dataset = Dataset

--- Rows of features with matching targets. Targets may be rows (regression)
--- or plain numbers (class indices for CrossEntropyLoss).
function Dataset.new(inputs: {{number}}, targets: {any}?)
	if targets and #inputs ~= #targets then
		error(("Dataset: %d inputs but %d targets"):format(#inputs, #targets), 2)
	end
	return setmetatable({
		inputs = inputs,
		targets = targets,
		count = #inputs,
	}, Dataset)
end

function Dataset.size(self: any): number
	return self.count
end

--- Shuffled train/validation split. The shuffle matters: data arrives sorted
--- far more often than expected, and a naive tail split on sorted data gives
--- a validation set of classes the model never trained on.
--- @param seed number? -- pass to make the split reproducible
function Dataset.split(self: any, trainFraction: number, seed: number?)
	local order = t_create(self.count)
	for i = 1, self.count do order[i] = i end

	-- An explicit seed gets a private stream, so a split stays identical even
	-- if the amount of global randomness drawn before it changes. With no
	-- seed, fall through to the shared stream so nano.seed still covers it.
	local rng = if seed then Config.stream(seed) else nil
	for i = self.count, 2, -1 do
		local j = if rng then rng:NextInteger(1, i) else Config.randomInt(i)
		order[i], order[j] = order[j], order[i]
	end

	local trainCount = math.floor(self.count * trainFraction)

	local function build(from: number, to: number)
		local inputs = t_create(to - from + 1)
		local targets = if self.targets then t_create(to - from + 1) else nil
		for k = from, to do
			local idx = order[k]
			inputs[k - from + 1] = self.inputs[idx]
			if targets then
				targets[k - from + 1] = self.targets[idx]
			end
		end
		return Dataset.new(inputs, targets)
	end

	return build(1, trainCount), build(trainCount + 1, self.count)
end

local Loader = {}
Loader.__index = Loader
data.Loader = Loader

--- Iterate a dataset in shuffled minibatches:
---     for X, Y in data.Loader(dataset, 32):iter() do
--- Reshuffles every epoch. Y comes back as a Tensor for row targets and a
--- plain array for class indices, matching what each loss expects.
function Loader.new(dataset: any, batchSize: number, shuffle: boolean?)
	return setmetatable({
		dataset = dataset,
		batchSize = batchSize,
		shuffle = if shuffle == nil then true else shuffle,
		dropLast = false,
	}, Loader)
end

function Loader.iter(self: any)
	local count = self.dataset.count
	local order = t_create(count)
	for i = 1, count do order[i] = i end

	-- Through Config, not math.random, so nano.seed actually covers minibatch
	-- order. It did not, which made a "reproducible" run reproducible in its
	-- weights and not in the order they were trained on.
	if self.shuffle then
		for i = count, 2, -1 do
			local j = Config.randomInt(i)
			order[i], order[j] = order[j], order[i]
		end
	end

	local cursor = 1
	local targets = self.dataset.targets
	local targetsAreRows = targets ~= nil and type(targets[1]) == "table"

	return function()
		if cursor > count then return nil end
		local last = math.min(cursor + self.batchSize - 1, count)
		if self.dropLast and last - cursor + 1 < self.batchSize then
			return nil
		end

		local size = last - cursor + 1
		local xRows = t_create(size)
		local yRows = if targets then t_create(size) else nil

		for k = cursor, last do
			local idx = order[k]
			xRows[k - cursor + 1] = self.dataset.inputs[idx]
			if yRows then
				yRows[k - cursor + 1] = targets[idx]
			end
		end

		cursor = last + 1

		local X = Tensor.new(xRows)
		if not yRows then
			return X, nil
		end
		if targetsAreRows then
			return X, Tensor.new(yRows)
		end
		return X, yRows      -- class indices, for CrossEntropyLoss
	end
end

function Loader.batchCount(self: any): number
	return math.ceil(self.dataset.count / self.batchSize)
end

local Scaler = {}
Scaler.__index = Scaler
data.Scaler = Scaler

--- Per-feature standardisation. Fit on TRAINING data only, then transform
--- both sets: fitting on everything leaks validation statistics into
--- training and makes the resulting score optimistic by an unknown amount.
--- @param mode string? -- "standard" (default) or "minmax"
function Scaler.new(mode: string?)
	return setmetatable({
		mode = mode or "standard",   -- "standard" or "minmax"
		mean = {},
		std = {},
		lo = {},
		hi = {},
		fitted = false,
	}, Scaler)
end

function Scaler.fit(self: any, rows: {{number}})
	local n = #rows
	if n == 0 then
		error("Scaler:fit needs at least one row", 2)
	end
	if type(rows[1]) ~= "table" then
		error("Scaler:fit expects rows of features, e.g. {{1, 2}, {3, 4}}", 2)
	end
	local features = #rows[1]

	local mean = t_create(features, 0)
	local lo = t_create(features, math.huge)
	local hi = t_create(features, -math.huge)

	for i = 1, n do
		local row = rows[i]
		for j = 1, features do
			local v = row[j]
			mean[j] += v
			if v < lo[j] then lo[j] = v end
			if v > hi[j] then hi[j] = v end
		end
	end
	for j = 1, features do mean[j] /= n end

	local std = t_create(features, 0)
	for i = 1, n do
		local row = rows[i]
		for j = 1, features do
			local d = row[j] - mean[j]
			std[j] += d * d
		end
	end
	for j = 1, features do
		-- the +1e-8 is not cosmetic: a constant feature has zero variance,
		-- and dividing by it turns every row into nan
		std[j] = m_sqrt(std[j] / n) + 1e-8
	end

	self.mean, self.std, self.lo, self.hi = mean, std, lo, hi
	self.features = features
	self.fitted = true
	return self
end

function Scaler.transform(self: any, rows: {{number}}): {{number}}
	if not self.fitted then
		error("Scaler: fit() before transform()", 2)
	end
	local out = t_create(#rows)
	for i = 1, #rows do
		local row = rows[i]
		local scaled = t_create(self.features)
		if self.mode == "minmax" then
			for j = 1, self.features do
				local span = self.hi[j] - self.lo[j]
				scaled[j] = if span > 1e-12 then (row[j] - self.lo[j]) / span else 0
			end
		else
			for j = 1, self.features do
				scaled[j] = (row[j] - self.mean[j]) / self.std[j]
			end
		end
		out[i] = scaled
	end
	return out
end

function Scaler.fitTransform(self: any, rows: {{number}}): {{number}}
	return self:fit(rows):transform(rows)
end

data.metrics = {}

function data.metrics.accuracy(logits: Tensor, targets: {number}): number
	local flat = logits:toTable()
	local rows = logits.shape[1]
	local classes = logits.shape[2]
	local correct = 0

	for r = 1, rows do
		local base = (r - 1) * classes
		local best, bestIndex = flat[base + 1], 1
		for c = 2, classes do
			if flat[base + c] > best then
				best, bestIndex = flat[base + c], c
			end
		end
		if bestIndex == targets[r] then
			correct += 1
		end
	end
	return correct / rows
end

function data.metrics.rmse(prediction: Tensor, target: Tensor): number
	local p, t = prediction:toTable(), target:toTable()
	local total = 0
	for i = 1, #p do
		local d = p[i] - t[i]
		total += d * d
	end
	return m_sqrt(total / #p)
end

function data.metrics.mae(prediction: Tensor, target: Tensor): number
	local p, t = prediction:toTable(), target:toTable()
	local total = 0
	for i = 1, #p do
		total += math.abs(p[i] - t[i])
	end
	return total / #p
end

--- Coefficient of determination. 1 is perfect, 0 matches predicting the mean,
--- negative is worse than that — common early in training and more
--- informative than an RMSE with no scale to compare against.
function data.metrics.r2(prediction: Tensor, target: Tensor): number
	local p, t = prediction:toTable(), target:toTable()
	local n = #t

	local mean = 0
	for i = 1, n do mean += t[i] end
	mean /= n

	local ssTot, ssRes = 0, 0
	for i = 1, n do
		local d = t[i] - mean
		ssTot += d * d
		local r = t[i] - p[i]
		ssRes += r * r
	end
	if ssTot < 1e-12 then return 0 end
	return 1 - ssRes / ssTot
end

--- Confusion matrix, actual (row) by predicted (column). Print it whenever
--- accuracy looks suspiciously good: 95% on data that is 95% one class means
--- the model learned to answer with that class.
function data.metrics.confusion(logits: Tensor, targets: {number}, classes: number): {{number}}
	local matrix = t_create(classes)
	for i = 1, classes do
		matrix[i] = t_create(classes, 0)
	end

	local flat = logits:toTable()
	local rows = logits.shape[1]

	for r = 1, rows do
		local base = (r - 1) * classes
		local best, bestIndex = flat[base + 1], 1
		for c = 2, classes do
			if flat[base + c] > best then
				best, bestIndex = flat[base + c], c
			end
		end
		matrix[targets[r]][bestIndex] += 1
	end
	return matrix
end

local EarlyStopping = {}
EarlyStopping.__index = EarlyStopping
data.EarlyStopping = EarlyStopping

--- Stop when validation stops improving, keeping the best weights seen.
--- The snapshot is the important half: training past the optimum actively
--- degrades the model, so restore() matters more than the stop signal.
function EarlyStopping.new(patience: number?, minDelta: number?)
	return setmetatable({
		patience = patience or 10,
		minDelta = minDelta or 0,
		best = math.huge,
		waiting = 0,
		bestWeights = nil,
		stopped = false,
	}, EarlyStopping)
end

function EarlyStopping.update(self: any, loss: number, model: any?): boolean
	if loss < self.best - self.minDelta then
		self.best = loss
		self.waiting = 0
		if model then
			self.bestWeights = model:getFlat()
		end
	else
		self.waiting += 1
		if self.waiting >= self.patience then
			self.stopped = true
		end
	end
	return self.stopped
end

function EarlyStopping.restore(self: any, model: any)
	if self.bestWeights then
		model:setFlat(self.bestWeights)
	end
end

return data
