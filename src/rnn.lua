--!strict
--!native
--!optimize 2
--[[
	Nano.rnn — recurrent layers.

	Cells are forward(x, state) -> (output, newState). State is never stored
	on the module: RL runs N environments through one network each with its
	own state, and truncated backprop needs to detach at a chosen boundary.
	Module-owned state makes both of those a mutation.
]]

local Tensor = require(script.Parent.Tensor)
local Config = require(script.Parent.Config)
local F = require(script.Parent.functional)
local nn = require(script.Parent.nn)

type Tensor = Tensor.Tensor

local rnn = {}

local Module = nn.Module

-- Two __call hooks: class metatable constructs, class table forwards.
local function makeCell(class: any)
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

--- State is a plain array of tensors: {h} for RNN and GRU, {h, c} for LSTM.
--- Array rather than named fields so the runners work for any cell type.
export type State = {Tensor}

function rnn.zeroState(cell: any, batch: number): State
	local out = {}
	for i = 1, cell.stateCount do
		out[i] = Tensor.zeros({ batch, cell.hiddenSize }, false)
	end
	return out
end

--- Truncated backprop through time: keep the values, cut the graph. Without
--- this the graph grows unbounded across a long rollout. Call every k steps,
--- where k is how far back credit should travel (16-64 is typical).
function rnn.detachState(state: State): State
	local out = table.create(#state)
	for i, t in state do
		out[i] = Tensor.detach(t)
	end
	return out
end

-- Pull one environment's state out of a batched state, and write one back.
-- Needed when N environments share a network but reset independently.
function rnn.sliceState(state: State, index: number): State
	local out = table.create(#state)
	for i, t in state do
		out[i] = Tensor.select(t, 1, index)
	end
	return out
end

--- Zero the state of batch rows whose environment just reset.
--- @param doneMask {number} -- 1 to reset that row
function rnn.resetRows(state: State, doneMask: {number})
	for _, t in state do
		local hidden = t.shape[2]
		local rowBytes = hidden * 8
		for row = 1, t.shape[1] do
			if doneMask[row] == 1 then
				buffer.fill(t.data, (row - 1) * rowBytes, 0, rowBytes)
			end
		end
	end
end

local RNNCell = makeCell({})
rnn.RNNCell = RNNCell

--- Simple recurrent cell: h' = tanh(x @ W_ih + h @ W_hh + b).
--- Gradient decays or explodes geometrically with sequence length — reach
--- for GRU or LSTM unless sequences are very short.
function RNNCell.new(inputSize: number, hiddenSize: number, activation: string?)
	local self = Module.init(setmetatable({}, RNNCell))

	self.inputSize = inputSize
	self.hiddenSize = hiddenSize
	self.stateCount = 1
	self.activation = activation or "tanh"

	self:registerParameter("weightIH", nn.xavier(inputSize, hiddenSize, { inputSize, hiddenSize }))
	self:registerParameter("weightHH", F.orthogonal({ hiddenSize, hiddenSize }, 1))
	self:registerParameter("bias", Tensor.zeros({ hiddenSize }, true))

	self.weightIH = self._parameters.weightIH
	self.weightHH = self._parameters.weightHH
	self.bias = self._parameters.bias

	-- orthogonal() returns an untracked tensor; the recurrent matrix is a
	-- learned parameter like any other, so it has to opt in
	self.weightHH.requiresGrad = true
	self.weightHH.grad = buffer.create(hiddenSize * hiddenSize * 8)

	return self
end

function RNNCell.forward(self: any, x: Tensor, state: State?): (Tensor, State)
	local s = state or rnn.zeroState(self, x.shape[1])
	local h = s[1]

	local pre = Tensor.add(
		Tensor.add(Tensor.matmul(x, self.weightIH), Tensor.matmul(h, self.weightHH)),
		self.bias
	)
	local out = if self.activation == "relu" then Tensor.relu(pre) else Tensor.tanh(pre)
	return out, { out }
end

local LSTMCell = makeCell({})
rnn.LSTMCell = LSTMCell

--- LSTM cell. Gates are i, f, g, o in that order (matching PyTorch, so
--- weights transfer).
---
--- The cell line c' = f*c + i*g has no repeated matrix multiply, so
--- gradient flowing back along c is scaled only by the forget gates. That
--- additive path is what a plain RNN lacks.
function LSTMCell.new(inputSize: number, hiddenSize: number)
	local self = Module.init(setmetatable({}, LSTMCell))

	self.inputSize = inputSize
	self.hiddenSize = hiddenSize
	self.stateCount = 2      -- h and c

	local gates = 4 * hiddenSize

	self:registerParameter("weightIH", nn.xavier(inputSize, gates, { inputSize, gates }))

	-- Four stacked H x H blocks, each orthogonal on its own. One orthogonal
	-- {H, 4H} is not equivalent: no individual block would be norm-preserving.
	local blocks = table.create(4)
	for i = 1, 4 do
		blocks[i] = F.orthogonal({ hiddenSize, hiddenSize }, 1)
	end
	local weightHH = F.cat(blocks, 2)
	weightHH.requiresGrad = true
	weightHH.grad = buffer.create(hiddenSize * gates * 8)
	weightHH._isLeaf = true
	weightHH._prev = {}
	weightHH._backward = nil
	self:registerParameter("weightHH", weightHH)

	-- Forget bias starts at 1: at 0 the gate is sigmoid(0)=0.5, so cell state
	-- halves every step and the net must learn to remember before it can learn
	-- anything requiring memory. Open by default; forgetting gets learned.
	local bias = Tensor.zeros({ gates }, true)
	for j = hiddenSize, 2 * hiddenSize - 1 do    -- the f slice, 0-based
		buffer.writef64(bias.data, j * 8, 1)
	end
	self:registerParameter("bias", bias)

	self.weightIH = self._parameters.weightIH
	self.weightHH = self._parameters.weightHH
	self.bias = self._parameters.bias

	return self
end

function LSTMCell.forward(self: any, x: Tensor, state: State?): (Tensor, State)
	local s = state or rnn.zeroState(self, x.shape[1])
	local h, c = s[1], s[2]

	-- two matmuls, not eight
	local gates = Tensor.add(
		Tensor.add(Tensor.matmul(x, self.weightIH), Tensor.matmul(h, self.weightHH)),
		self.bias
	)

	local parts = F.chunk(gates, 4, 2)
	local i = Tensor.sigmoid(parts[1])
	local f = Tensor.sigmoid(parts[2])
	local g = Tensor.tanh(parts[3])
	local o = Tensor.sigmoid(parts[4])

	local newC = Tensor.add(Tensor.mul(f, c), Tensor.mul(i, g))
	local newH = Tensor.mul(o, Tensor.tanh(newC))

	return newH, { newH, newC }
end

local GRUCell = makeCell({})
rnn.GRUCell = GRUCell

--- GRU cell. One update gate replaces the LSTM's separate input and forget
--- gates, so they cannot disagree: ~25% fewer parameters and compute for
--- comparable quality. Start here.
---
--- The reset gate applies to the hidden projection AFTER its matmul, not to
--- h before it — that is why the hidden side cannot be one fused matmul.
function GRUCell.new(inputSize: number, hiddenSize: number)
	local self = Module.init(setmetatable({}, GRUCell))

	self.inputSize = inputSize
	self.hiddenSize = hiddenSize
	self.stateCount = 1

	local gates = 3 * hiddenSize

	self:registerParameter("weightIH", nn.xavier(inputSize, gates, { inputSize, gates }))

	local blocks = table.create(3)
	for i = 1, 3 do
		blocks[i] = F.orthogonal({ hiddenSize, hiddenSize }, 1)
	end
	local weightHH = F.cat(blocks, 2)
	weightHH.requiresGrad = true
	weightHH.grad = buffer.create(hiddenSize * gates * 8)
	weightHH._isLeaf = true
	weightHH._prev = {}
	weightHH._backward = nil
	self:registerParameter("weightHH", weightHH)

	self:registerParameter("biasIH", Tensor.zeros({ gates }, true))
	self:registerParameter("biasHH", Tensor.zeros({ gates }, true))

	self.weightIH = self._parameters.weightIH
	self.weightHH = self._parameters.weightHH
	self.biasIH = self._parameters.biasIH
	self.biasHH = self._parameters.biasHH

	return self
end

function GRUCell.forward(self: any, x: Tensor, state: State?): (Tensor, State)
	local s = state or rnn.zeroState(self, x.shape[1])
	local h = s[1]
	local H = self.hiddenSize

	-- Two separate biases, unlike the LSTM. The reset gate multiplies the
	-- hidden contribution to n but not the input contribution, so the two
	-- halves cannot share a bias without changing what r gates.
	local xAll = Tensor.add(Tensor.matmul(x, self.weightIH), self.biasIH)
	local hAll = Tensor.add(Tensor.matmul(h, self.weightHH), self.biasHH)

	local xr = Tensor.narrow(xAll, 2, 1, H)
	local xz = Tensor.narrow(xAll, 2, H + 1, H)
	local xn = Tensor.narrow(xAll, 2, 2 * H + 1, H)

	local hr = Tensor.narrow(hAll, 2, 1, H)
	local hz = Tensor.narrow(hAll, 2, H + 1, H)
	local hn = Tensor.narrow(hAll, 2, 2 * H + 1, H)

	local r = Tensor.sigmoid(Tensor.add(xr, hr))
	local z = Tensor.sigmoid(Tensor.add(xz, hz))
	local n = Tensor.tanh(Tensor.add(xn, Tensor.mul(r, hn)))

	-- h' = (1-z)*n + z*h, written as n + z*(h - n) to save one op
	local newH = Tensor.add(n, Tensor.mul(z, Tensor.sub(h, n)))
	return newH, { newH }
end

local Recurrent = makeCell({})
rnn.Recurrent = Recurrent

--- Stack cells with optional inter-layer dropout. State is one flat array
--- holding every layer's state, so the stack still looks like a single cell.
--- @param cellType string -- "lstm", "gru" or "rnn"
function Recurrent.new(cellType: string, inputSize: number, hiddenSize: number, layers: number?, dropout: number?)
	local self = Module.init(setmetatable({}, Recurrent))

	local count = layers or 1
	self.hiddenSize = hiddenSize
	self.layerCount = count
	self.dropout = dropout or 0

	local constructor
	if cellType == "lstm" then
		constructor = LSTMCell.new
	elseif cellType == "gru" then
		constructor = GRUCell.new
	elseif cellType == "rnn" then
		constructor = RNNCell.new
	else
		error("Recurrent: cellType must be 'lstm', 'gru' or 'rnn'", 2)
	end

	self.cells = table.create(count)
	self.perCellState = 0
	for i = 1, count do
		local cell = constructor(if i == 1 then inputSize else hiddenSize, hiddenSize)
		self.cells[i] = cell
		self:registerModule(string.format("%03d", i), cell)
		self.perCellState = cell.stateCount
	end
	self.stateCount = self.perCellState * count

	return self
end

function Recurrent.forward(self: any, x: Tensor, state: State?): (Tensor, State)
	local s = state or rnn.zeroState(self, x.shape[1])
	local per = self.perCellState
	local newState = table.create(self.stateCount)

	local out = x
	for i = 1, self.layerCount do
		local base = (i - 1) * per
		local cellState = table.create(per)
		for j = 1, per do
			cellState[j] = s[base + j]
		end

		local cellOut, cellNew = self.cells[i]:forward(out, cellState)
		for j = 1, per do
			newState[base + j] = cellNew[j]
		end

		out = cellOut
		if self.dropout > 0 and self.training and i < self.layerCount then
			out = nn.dropoutFn(out, self.dropout, true)
		end
	end

	return out, newState
end

--- Run a cell over a sequence of {batch, features} tensors.
--- Outputs stay a Lua array; use F.stack if you want one {T,B,H} tensor.
function rnn.runSequence(cell: any, inputs: {Tensor}, state: State?): ({Tensor}, State)
	local s = state or rnn.zeroState(cell, inputs[1].shape[1])
	local outputs = table.create(#inputs)

	for t = 1, #inputs do
		local out, newState = cell:forward(inputs[t], s)
		outputs[t] = out
		s = newState
	end

	return outputs, s
end

--- Run a long sequence in windows, calling `onWindow` after each so you can
--- compute a loss and step, then detaching state before continuing.
--- Bounds graph size and backward cost while state still carries across.
function rnn.runTruncated(
	cell: any,
	inputs: {Tensor},
	window: number,
	onWindow: ({Tensor}, number, number) -> (),
	state: State?
): State
	local s = state or rnn.zeroState(cell, inputs[1].shape[1])
	local total = #inputs
	local cursor = 1

	while cursor <= total do
		local last = math.min(cursor + window - 1, total)
		local outputs = table.create(last - cursor + 1)

		for t = cursor, last do
			local out, newState = cell:forward(inputs[t], s)
			outputs[t - cursor + 1] = out
			s = newState
		end

		onWindow(outputs, cursor, last)

		-- the cut: values survive, the graph does not
		s = rnn.detachState(s)
		cursor = last + 1
	end

	return s
end

function rnn.LSTM(inputSize: number, hiddenSize: number, layers: number?, dropout: number?)
	return Recurrent.new("lstm", inputSize, hiddenSize, layers, dropout)
end

function rnn.GRU(inputSize: number, hiddenSize: number, layers: number?, dropout: number?)
	return Recurrent.new("gru", inputSize, hiddenSize, layers, dropout)
end

function rnn.RNN(inputSize: number, hiddenSize: number, layers: number?, dropout: number?)
	return Recurrent.new("rnn", inputSize, hiddenSize, layers, dropout)
end

return rnn
