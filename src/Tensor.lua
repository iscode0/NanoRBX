--!strict
--!native
--!optimize 2
--[[
	Nano.Tensor — strided buffer-storage tensor with reverse-mode autograd.

	STORAGE. A {2,3} tensor is one buffer of six f64 values plus shape,
	stride and offset. Element (i,j) lives at
	    offset + (i-1)*stride[1] + (j-1)*stride[2]
	measured in ELEMENTS; every access multiplies by 8 to get bytes. Hot
	loops carry byte offsets directly so that multiply disappears.

	WHY BUFFER, AND WHY f64. Measured on this workload: a table array read
	is one VM instruction and buffer.readf64 is a builtin call, so pure
	sequential access is a wash (1.02x). But the inner op is read-modify-
	write, which measures 1.44-1.53x, and a full matmul measures 1.82x —
	and matmul is ~100% of a training step. f32 is not an option: it puts
	0.14% error on a central difference against a 2e-4 gradcheck tolerance.
	f64 is bit-identical to table arithmetic, which is what makes this
	migration verifiable — all 110 gradchecks must still pass unchanged.
	Memory halves as a side effect: 8 bytes against a 16-byte TValue.

	AUTOGRAD. Every tensor from an operation records _prev (its inputs),
	_backward (a closure pushing gradient into them) and grad. Gradients
	ACCUMULATE — that is what makes multi-branch graphs work.

	INTEROP. data is a buffer, not a table. Use t:toTable() to get a Lua
	array and Tensor.fromTable() to build one. t:toFlat() returns a
	contiguous buffer.
]]

local Config = require(script.Parent.Config)

local Tensor = {}
Tensor.__index = Tensor

local m_exp = math.exp
local m_log = math.log
local m_sqrt = math.sqrt
local m_abs = math.abs
local m_tanh = math.tanh
local m_clamp = math.clamp
local m_min = math.min
local m_max = math.max
local m_huge = math.huge
-- randomness routed through Config so Config.seed covers tensor init
local t_create = table.create
local t_clone = table.clone

local b_create = buffer.create
local b_read = buffer.readf64
local b_write = buffer.writef64
local b_copy = buffer.copy
local b_fill = buffer.fill
local b_len = buffer.len

export type Tensor = typeof(setmetatable({} :: {
	data: buffer,
	shape: {number},
	stride: {number},
	offset: number,
	requiresGrad: boolean,
	grad: buffer?,
	_prev: {any},
	_backward: (() -> ())?,
	_isLeaf: boolean,
	_contig: boolean,
	_gradStale: boolean,
}, Tensor))

local EMPTY_PREV: {any} = table.freeze({})

-- ==========================================================================
-- SHAPE HELPERS
-- ==========================================================================

local function numelOf(shape: {number}): number
	local n = 1
	for i = 1, #shape do
		n *= shape[i]
	end
	return n
end

local function stridesFor(shape: {number}): {number}
	local stride = t_create(#shape)
	local acc = 1
	for i = #shape, 1, -1 do
		stride[i] = acc
		acc *= shape[i]
	end
	return stride
end

local function shapesEqual(a: {number}, b: {number}): boolean
	local n = #a
	if n ~= #b then return false end
	for i = 1, n do
		if a[i] ~= b[i] then return false end
	end
	return true
end

local function shapeToString(shape: {number}): string
	local parts = t_create(#shape)
	for i = 1, #shape do
		parts[i] = tostring(shape[i])
	end
	return "{" .. table.concat(parts, ", ") .. "}"
end

-- ==========================================================================
-- STORAGE HELPERS
-- ==========================================================================

--- Allocate storage for n f64 elements. buffer.create always zero-fills, so
--- this is both the "zeroed" and the "raw" allocator; there is no cheaper
--- uninitialised variant to reach for.
local function alloc(n: number): buffer
	return b_create(n * 8)
end

local function fillBuffer(buf: buffer, value: number, n: number)
	if value == 0 then
		b_fill(buf, 0, 0, n * 8)
		return
	end
	local o = 0
	for _ = 1, n do
		b_write(buf, o, value)
		o += 8
	end
end

local function bufferFromTable(values: {number}, n: number): buffer
	local buf = alloc(n)
	local o = 0
	for i = 1, n do
		b_write(buf, o, values[i])
		o += 8
	end
	return buf
end

local function tableFromBuffer(buf: buffer, n: number, from: number?): {number}
	local out = t_create(n)
	local o = (from or 0) * 8
	for i = 1, n do
		out[i] = b_read(buf, o)
		o += 8
	end
	return out
end

-- ==========================================================================
-- CONSTRUCTION
-- ==========================================================================

local function inferShape(nested: any): {number}
	local shape = {}
	local node = nested
	while type(node) == "table" do
		table.insert(shape, #node)
		node = node[1]
	end
	return shape
end

local function flattenNested(nested: any, out: {number})
	if type(nested) == "table" then
		for i = 1, #nested do
			flattenNested(nested[i], out)
		end
	else
		out[#out + 1] = nested
	end
end

--- Construct a tensor from a nested table, a flat table plus shape, a buffer
--- plus shape, or a single number.
function Tensor.new(data: any, shape: {number}?, requiresGrad: boolean?): Tensor
	local storage: buffer
	local finalShape: {number}

	if type(data) == "buffer" then
		if not shape then
			error("Tensor.new: a buffer needs an explicit shape", 2)
		end
		storage = data
		finalShape = shape
		local n = numelOf(finalShape)
		if b_len(storage) < n * 8 then
			error(("Tensor.new: buffer holds %d elements, shape %s needs %d")
				:format(b_len(storage) // 8, shapeToString(finalShape), n), 2)
		end
	elseif shape then
		finalShape = shape
		local n = numelOf(finalShape)
		if #data ~= n then
			error(("Tensor.new: %d values do not fill shape %s")
				:format(#data, shapeToString(finalShape)), 2)
		end
		storage = bufferFromTable(data, n)
	elseif type(data) == "number" then
		finalShape = { 1 }
		storage = alloc(1)
		b_write(storage, 0, data)
	else
		finalShape = inferShape(data)
		local flat = {}
		flattenNested(data, flat)
		local n = numelOf(finalShape)
		if #flat ~= n then
			error(("Tensor.new: %d values do not fill shape %s")
				:format(#flat, shapeToString(finalShape)), 2)
		end
		storage = bufferFromTable(flat, n)
	end

	local self = setmetatable({
		data = storage,
		shape = finalShape,
		stride = stridesFor(finalShape),
		offset = 0,
		requiresGrad = requiresGrad or false,
		grad = nil,
		_prev = EMPTY_PREV,
		_backward = nil,
		_isLeaf = true,
		_contig = true,
		_gradStale = false,
	}, Tensor)

	if self.requiresGrad then
		self.grad = alloc(numelOf(finalShape))
	end

	return self
end

--- Build from a plain Lua array. Equivalent to Tensor.new(values, shape).
function Tensor.fromTable(values: {number}, shape: {number}?, requiresGrad: boolean?): Tensor
	return Tensor.new(values, shape or { #values }, requiresGrad)
end

--[[
	Result of an op: skips validation, never eagerly allocates grad.

	_isLeaf is FALSE here even when nothing is tracked. It used to be true and
	only markTracked cleared it, which meant the in-place guard fired on
	"tracked" rather than on "produced by an operation" — so every result
	computed under noGrad, or from inputs that did not require grad, accepted
	silent mutation despite the docs promising leaves only. Nothing broke,
	because an untracked result has no graph to corrupt, but the guard did not
	mean what it said and a later change could have made that matter.
]]
local function newResult(storage: buffer, shape: {number}): Tensor
	return setmetatable({
		data = storage,
		shape = shape,
		stride = stridesFor(shape),
		offset = 0,
		requiresGrad = false,
		grad = nil,
		_prev = EMPTY_PREV,
		_backward = nil,
		_isLeaf = false,
		_contig = true,
		_gradStale = false,
	}, Tensor)
end

local function markTracked(t: Tensor, prev: {any})
	t.requiresGrad = true
	t._prev = prev
	t._isLeaf = false
end

function Tensor.zeros(shape: {number}, requiresGrad: boolean?): Tensor
	return Tensor.new(alloc(numelOf(shape)), shape, requiresGrad)
end

function Tensor.ones(shape: {number}, requiresGrad: boolean?): Tensor
	local n = numelOf(shape)
	local buf = alloc(n)
	fillBuffer(buf, 1, n)
	return Tensor.new(buf, shape, requiresGrad)
end

function Tensor.full(shape: {number}, value: number, requiresGrad: boolean?): Tensor
	local n = numelOf(shape)
	local buf = alloc(n)
	fillBuffer(buf, value, n)
	return Tensor.new(buf, shape, requiresGrad)
end

function Tensor.randn(shape: {number}, requiresGrad: boolean?): Tensor
	local n = numelOf(shape)
	local buf = alloc(n)
	local o = 0
	for _ = 1, n do
		b_write(buf, o, Config.gaussian())
		o += 8
	end
	return Tensor.new(buf, shape, requiresGrad)
end

function Tensor.rand(shape: {number}, requiresGrad: boolean?): Tensor
	local n = numelOf(shape)
	local buf = alloc(n)
	local o = 0
	for _ = 1, n do
		b_write(buf, o, Config.random())
		o += 8
	end
	return Tensor.new(buf, shape, requiresGrad)
end

function Tensor.isTensor(x: any): boolean
	return type(x) == "table" and getmetatable(x) == Tensor
end

-- ==========================================================================
-- READING
-- ==========================================================================

function Tensor.numel(self: Tensor): number
	return numelOf(self.shape)
end

function Tensor.dim(self: Tensor): number
	return #self.shape
end

local function unravel(shape: {number}, linear: number, out: {number})
	local rem = linear
	for i = #shape, 1, -1 do
		local s = shape[i]
		out[i] = rem % s
		rem //= s
	end
end

local function isContiguous(t: Tensor): boolean
	return t._contig == true
end

Tensor.isContiguous = isContiguous

--- A contiguous buffer of this tensor's values in row-major order. Returns
--- the underlying storage directly when it is already contiguous at offset 0.
function Tensor.toFlat(self: Tensor): buffer
	local n = numelOf(self.shape)
	if self._contig and self.offset == 0 then
		return self.data
	end

	local out = alloc(n)
	if self._contig then
		b_copy(out, 0, self.data, self.offset * 8, n * 8)
		return out
	end

	local shape, stride, data = self.shape, self.stride, self.data
	local ndim = #shape
	local idx = t_create(ndim, 0)
	local o = 0
	for linear = 0, n - 1 do
		unravel(shape, linear, idx)
		local off = self.offset
		for i = 1, ndim do
			off += idx[i] * stride[i]
		end
		b_write(out, o, b_read(data, off * 8))
		o += 8
	end
	return out
end

--- Values as a plain Lua array. The interop path for code that has not been
--- migrated to buffers.
function Tensor.toTable(self: Tensor): {number}
	local n = numelOf(self.shape)
	if self._contig then
		return tableFromBuffer(self.data, n, self.offset)
	end
	return tableFromBuffer(self:toFlat(), n, 0)
end

function Tensor.item(self: Tensor): number
	if numelOf(self.shape) ~= 1 then
		error("item() requires a single-element tensor, got " .. shapeToString(self.shape), 2)
	end
	return b_read(self.data, self.offset * 8)
end

function Tensor.get(self: Tensor, ...: number): number
	local off = self.offset
	local stride = self.stride
	for i = 1, select("#", ...) do
		off += ((select(i, ...)) - 1) * stride[i]
	end
	return b_read(self.data, off * 8)
end

--[[
	Read element i (1-based) of the flat row-major layout.

	Contiguous only, and asserted rather than assumed. The offset arithmetic
	here ignores strides entirely, so on a non-contiguous tensor it would read
	the wrong element and report nothing. Every operation in Nano materializes,
	so _contig is always true today — but that is an invariant held by
	convention, and one view implementation away from making this silently
	wrong. The branch is perfectly predicted and effectively free.
]]
function Tensor.at(self: Tensor, i: number): number
	if not self._contig then
		error("at() is contiguous-only; call contiguous() first", 2)
	end
	return b_read(self.data, (self.offset + i - 1) * 8)
end

--- Write element i (1-based) of the flat row-major layout. No autograd.
--- Contiguous only, for the same reason as at().
function Tensor.setAt(self: Tensor, i: number, value: number)
	if not self._contig then
		error("setAt() is contiguous-only; call contiguous() first", 2)
	end
	b_write(self.data, (self.offset + i - 1) * 8, value)
end

function Tensor.contiguous(self: Tensor): Tensor
	local n = numelOf(self.shape)
	local out = alloc(n)
	b_copy(out, 0, self:toFlat(), 0, n * 8)
	return Tensor.new(out, t_clone(self.shape), false)
end

function Tensor.__tostring(self: Tensor): string
	local n = numelOf(self.shape)
	local flat = self:toFlat()
	local shown = m_min(n, 12)
	local parts = t_create(shown)
	for i = 1, shown do
		parts[i] = string.format("%.4f", b_read(flat, (i - 1) * 8))
	end
	local body = table.concat(parts, ", ")
	if n > shown then
		body ..= ", ..."
	end
	return ("Tensor(%s, [%s])"):format(shapeToString(self.shape), body)
end

-- ==========================================================================
-- AUTOGRAD CORE
-- ==========================================================================

--[[
	Gradient accumulation.

	SET TO NONE. _gradStale means "what is in grad belongs to a previous
	step; overwrite rather than add". The optimiser sets it after step(),
	which lets it skip zero-filling every buffer — an O(params) pass removed
	from every training step. The first accumulation of a step overwrites;
	every later one adds, so multi-branch graphs still work.

	LENGTH GUARD. A backward returning the wrong element count used to be
	silently partially applied: no error, just a subtly wrong model. That is
	the failure mode of a hand-written backward, so it errors here.
]]
local function accumulate(t: Tensor, values: buffer, count: number?)
	if not t.requiresGrad then return end

	local n = numelOf(t.shape)
	local supplied = count or (b_len(values) // 8)
	if supplied ~= n then
		error(("gradient has %d elements but tensor %s has %d — a backward pass is returning the wrong shape")
			:format(supplied, shapeToString(t.shape), n), 2)
	end

	local g = t.grad
	if not g then
		g = alloc(n)
		t.grad = g
		t._gradStale = false
	end

	if t._gradStale then
		t._gradStale = false
		b_copy(g, 0, values, 0, n * 8)
	else
		local o = 0
		for _ = 1, n do
			b_write(g, o, b_read(g, o) + b_read(values, o))
			o += 8
		end
	end
end

Tensor._accumulate = accumulate

--- Accumulate from a plain Lua array. Interop for modules not yet migrated.
function Tensor.accumulateTable(t: Tensor, values: {number})
	accumulate(t, bufferFromTable(values, #values), #values)
end

--- Mark a gradient buffer stale so the next accumulation overwrites instead
--- of adding. Call after step(), never mid-backward.
function Tensor.markGradStale(self: Tensor)
	if self.grad then
		self._gradStale = true
	end
end

local function tracking1(a: Tensor): boolean
	return Config.gradEnabled and a.requiresGrad
end

local function tracking2(a: Tensor, b: Tensor): boolean
	return Config.gradEnabled and (a.requiresGrad or b.requiresGrad)
end

function Tensor.zeroGrad(self: Tensor)
	local g = self.grad
	if g then
		b_fill(g, 0, 0, b_len(g))
	end
	self._gradStale = false
end

function Tensor.detach(self: Tensor): Tensor
	return Tensor.contiguous(self)
end

--- Reverse-mode backward from a scalar. The topological sort is iterative so
--- deep graphs from long rollouts cannot hit a recursion limit.
function Tensor.backward(self: Tensor)
	if numelOf(self.shape) ~= 1 then
		error("backward() must start from a scalar; got " .. shapeToString(self.shape), 2)
	end

	local order: {Tensor} = {}
	local visited: {[Tensor]: boolean} = {}

	local stackNode: {Tensor} = { self }
	local stackChild: {number} = { 1 }
	visited[self] = true
	local top = 1

	while top > 0 do
		local node = stackNode[top]
		local ci = stackChild[top]
		local parent = node._prev[ci]
		if parent then
			stackChild[top] = ci + 1
			if not visited[parent] then
				visited[parent] = true
				top += 1
				stackNode[top] = parent
				stackChild[top] = 1
			end
		else
			order[#order + 1] = node
			stackNode[top] = nil
			stackChild[top] = nil
			top -= 1
		end
	end

	local seed = alloc(1)
	b_write(seed, 0, 1)
	self.grad = seed
	self._gradStale = false

	for i = #order, 1, -1 do
		local fn = order[i]._backward
		if fn then fn() end
	end
end

-- ==========================================================================
-- BROADCASTING
-- ==========================================================================

local function broadcastShape(a: {number}, b: {number}): {number}
	local n = m_max(#a, #b)
	local out = t_create(n)
	for i = 1, n do
		local x = a[#a - n + i] or 1
		local y = b[#b - n + i] or 1
		if x == y then
			out[i] = x
		elseif x == 1 then
			out[i] = y
		elseif y == 1 then
			out[i] = x
		else
			error(("cannot broadcast %s with %s"):format(shapeToString(a), shapeToString(b)))
		end
	end
	return out
end

local function broadcastStride(t: Tensor, target: {number}): {number}
	local n = #target
	local out = t_create(n)
	local pad = n - #t.shape
	for i = 1, n do
		local si = i - pad
		local size = if si >= 1 then t.shape[si] else 1
		local st = if si >= 1 then t.stride[si] else 0
		out[i] = if size == 1 and target[i] ~= 1 then 0 else st
	end
	return out
end

local function reduceGrad(grad: buffer, gradCount: number, gradShape: {number}, targetShape: {number}): (buffer, number)
	if shapesEqual(gradShape, targetShape) then
		return grad, gradCount
	end

	local targetCount = numelOf(targetShape)

	if targetCount == 1 then
		local total = 0
		local o = 0
		for _ = 1, gradCount do
			total += b_read(grad, o)
			o += 8
		end
		local out = alloc(1)
		b_write(out, 0, total)
		return out, 1
	end

	local n = #gradShape
	local pad = n - #targetShape
	local padded = t_create(n)
	for i = 1, n do
		padded[i] = if i > pad then targetShape[i - pad] else 1
	end

	local outStride = stridesFor(padded)
	local out = alloc(targetCount)

	local idx = t_create(n, 0)
	for linear = 0, gradCount - 1 do
		unravel(gradShape, linear, idx)
		local off = 0
		for i = 1, n do
			off += (if padded[i] == 1 then 0 else idx[i]) * outStride[i]
		end
		local bo = off * 8
		b_write(out, bo, b_read(out, bo) + b_read(grad, linear * 8))
	end
	return out, targetCount
end

-- ==========================================================================
-- ELEMENTWISE BINARY OPS
-- ==========================================================================

local function asTensor(x: any): Tensor
	if type(x) == "table" and getmetatable(x) == Tensor then return x end
	return Tensor.new(x)
end

--[[
	Opcodes: 1 add, 2 sub, 3 mul, 4 div.

	Paths, fastest first:
	  0. one operand is a single element
	  1. same shape, both contiguous
	  2. {n,m} with {m} row broadcast (the Linear bias case)
	  3. general strided broadcast

	Every loop carries byte offsets so the element-to-byte multiply does not
	appear inside it.
]]

local function fwd2(op: number, ad: buffer, ao: number, bd: buffer, bo: number, out: buffer, n: number)
	local a, b, o = ao * 8, bo * 8, 0
	if op == 1 then
		for _ = 1, n do
			b_write(out, o, b_read(ad, a) + b_read(bd, b)); a += 8; b += 8; o += 8
		end
	elseif op == 2 then
		for _ = 1, n do
			b_write(out, o, b_read(ad, a) - b_read(bd, b)); a += 8; b += 8; o += 8
		end
	elseif op == 3 then
		for _ = 1, n do
			b_write(out, o, b_read(ad, a) * b_read(bd, b)); a += 8; b += 8; o += 8
		end
	else
		for _ = 1, n do
			b_write(out, o, b_read(ad, a) / b_read(bd, b)); a += 8; b += 8; o += 8
		end
	end
end

local function fwdScalar(op: number, ad: buffer, ao: number, s: number, out: buffer, n: number)
	local a, o = ao * 8, 0
	if op == 1 then
		for _ = 1, n do b_write(out, o, b_read(ad, a) + s); a += 8; o += 8 end
	elseif op == 2 then
		for _ = 1, n do b_write(out, o, b_read(ad, a) - s); a += 8; o += 8 end
	elseif op == 3 then
		for _ = 1, n do b_write(out, o, b_read(ad, a) * s); a += 8; o += 8 end
	else
		local inv = 1 / s
		for _ = 1, n do b_write(out, o, b_read(ad, a) * inv); a += 8; o += 8 end
	end
end

local function fwdScalarLeft(op: number, s: number, bd: buffer, bo: number, out: buffer, n: number)
	local b, o = bo * 8, 0
	if op == 1 then
		for _ = 1, n do b_write(out, o, s + b_read(bd, b)); b += 8; o += 8 end
	elseif op == 2 then
		for _ = 1, n do b_write(out, o, s - b_read(bd, b)); b += 8; o += 8 end
	elseif op == 3 then
		for _ = 1, n do b_write(out, o, s * b_read(bd, b)); b += 8; o += 8 end
	else
		for _ = 1, n do b_write(out, o, s / b_read(bd, b)); b += 8; o += 8 end
	end
end

-- Storage for an operand as (buffer, elementOffset), materialising if needed.
local function operand(t: Tensor): (buffer, number)
	if t._contig then
		return t.data, t.offset
	end
	return t:toFlat(), 0
end

local function arithTensorScalar(a: Tensor, s: number, op: number, scalarLeft: boolean): Tensor
	local n = numelOf(a.shape)
	local out = alloc(n)
	local ad, ao = operand(a)

	if scalarLeft then
		fwdScalarLeft(op, s, ad, ao, out, n)
	else
		fwdScalar(op, ad, ao, s, out, n)
	end

	local res = newResult(out, t_clone(a.shape))
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			local o, av = 0, ao * 8
			if op == 1 then
				b_copy(ga, 0, g, 0, n * 8)
			elseif op == 2 then
				if scalarLeft then
					for _ = 1, n do b_write(ga, o, -b_read(g, o)); o += 8 end
				else
					b_copy(ga, 0, g, 0, n * 8)
				end
			elseif op == 3 then
				for _ = 1, n do b_write(ga, o, b_read(g, o) * s); o += 8 end
			else
				if scalarLeft then
					for _ = 1, n do
						local x = b_read(ad, av)
						b_write(ga, o, -b_read(g, o) * s / (x * x))
						o += 8; av += 8
					end
				else
					local inv = 1 / s
					for _ = 1, n do b_write(ga, o, b_read(g, o) * inv); o += 8 end
				end
			end
			accumulate(a, ga, n)
		end
	end
	return res
end

local function arith(a: Tensor, b: Tensor, op: number): Tensor
	-- path 1: identical shapes, both contiguous
	if a._contig and b._contig and shapesEqual(a.shape, b.shape) then
		local n = numelOf(a.shape)
		local ad, ao = a.data, a.offset
		local bd, bo = b.data, b.offset
		local out = alloc(n)
		fwd2(op, ad, ao, bd, bo, out, n)

		local res = newResult(out, t_clone(a.shape))
		if tracking2(a, b) then
			markTracked(res, { a, b })
			res._backward = function()
				local g = res.grad :: buffer
				if a.requiresGrad then
					local ga = alloc(n)
					local o, bv = 0, bo * 8
					if op == 1 or op == 2 then
						b_copy(ga, 0, g, 0, n * 8)
					elseif op == 3 then
						for _ = 1, n do
							b_write(ga, o, b_read(g, o) * b_read(bd, bv)); o += 8; bv += 8
						end
					else
						for _ = 1, n do
							b_write(ga, o, b_read(g, o) / b_read(bd, bv)); o += 8; bv += 8
						end
					end
					accumulate(a, ga, n)
				end
				if b.requiresGrad then
					local gb = alloc(n)
					local o, av, bv = 0, ao * 8, bo * 8
					if op == 1 then
						b_copy(gb, 0, g, 0, n * 8)
					elseif op == 2 then
						for _ = 1, n do b_write(gb, o, -b_read(g, o)); o += 8 end
					elseif op == 3 then
						for _ = 1, n do
							b_write(gb, o, b_read(g, o) * b_read(ad, av)); o += 8; av += 8
						end
					else
						for _ = 1, n do
							local y = b_read(bd, bv)
							b_write(gb, o, -b_read(g, o) * b_read(ad, av) / (y * y))
							o += 8; av += 8; bv += 8
						end
					end
					accumulate(b, gb, n)
				end
			end
		end
		return res
	end

	-- path 0: one operand is a single element
	local nb = numelOf(b.shape)
	if nb == 1 and #b.shape <= #a.shape then
		local n = numelOf(a.shape)
		local ad, ao = operand(a)
		local bv = b_read(b.data, b.offset * 8)
		local out = alloc(n)
		fwdScalar(op, ad, ao, bv, out, n)

		local res = newResult(out, t_clone(a.shape))
		if tracking2(a, b) then
			markTracked(res, { a, b })
			res._backward = function()
				local g = res.grad :: buffer
				if a.requiresGrad then
					local ga = alloc(n)
					local o = 0
					if op == 1 or op == 2 then
						b_copy(ga, 0, g, 0, n * 8)
					elseif op == 3 then
						for _ = 1, n do b_write(ga, o, b_read(g, o) * bv); o += 8 end
					else
						local inv = 1 / bv
						for _ = 1, n do b_write(ga, o, b_read(g, o) * inv); o += 8 end
					end
					accumulate(a, ga, n)
				end
				if b.requiresGrad then
					local total = 0
					local o, av = 0, ao * 8
					if op == 1 then
						for _ = 1, n do total += b_read(g, o); o += 8 end
					elseif op == 2 then
						for _ = 1, n do total -= b_read(g, o); o += 8 end
					elseif op == 3 then
						for _ = 1, n do
							total += b_read(g, o) * b_read(ad, av); o += 8; av += 8
						end
					else
						local inv2 = 1 / (bv * bv)
						for _ = 1, n do
							total -= b_read(g, o) * b_read(ad, av) * inv2; o += 8; av += 8
						end
					end
					local gb = alloc(1)
					b_write(gb, 0, total)
					accumulate(b, gb, 1)
				end
			end
		end
		return res
	end

	local na = numelOf(a.shape)
	if na == 1 and #a.shape <= #b.shape then
		local n = numelOf(b.shape)
		local bd, bo = operand(b)
		local av = b_read(a.data, a.offset * 8)
		local out = alloc(n)
		fwdScalarLeft(op, av, bd, bo, out, n)

		local res = newResult(out, t_clone(b.shape))
		if tracking2(a, b) then
			markTracked(res, { a, b })
			res._backward = function()
				local g = res.grad :: buffer
				if a.requiresGrad then
					local total = 0
					local o, bv = 0, bo * 8
					if op == 1 or op == 2 then
						for _ = 1, n do total += b_read(g, o); o += 8 end
					elseif op == 3 then
						for _ = 1, n do
							total += b_read(g, o) * b_read(bd, bv); o += 8; bv += 8
						end
					else
						for _ = 1, n do
							total += b_read(g, o) / b_read(bd, bv); o += 8; bv += 8
						end
					end
					local ga = alloc(1)
					b_write(ga, 0, total)
					accumulate(a, ga, 1)
				end
				if b.requiresGrad then
					local gb = alloc(n)
					local o, bv = 0, bo * 8
					if op == 1 then
						b_copy(gb, 0, g, 0, n * 8)
					elseif op == 2 then
						for _ = 1, n do b_write(gb, o, -b_read(g, o)); o += 8 end
					elseif op == 3 then
						for _ = 1, n do b_write(gb, o, b_read(g, o) * av); o += 8 end
					else
						for _ = 1, n do
							local y = b_read(bd, bv)
							b_write(gb, o, -b_read(g, o) * av / (y * y))
							o += 8; bv += 8
						end
					end
					accumulate(b, gb, n)
				end
			end
		end
		return res
	end

	-- path 2: {n,m} with {m} or {1,m}, both contiguous
	local aCols = a.shape[2]
	local bIsRow = (#b.shape == 1 and b.shape[1] == aCols)
		or (#b.shape == 2 and b.shape[1] == 1 and b.shape[2] == aCols)

	if #a.shape == 2 and bIsRow and a._contig and b._contig then
		local aRows = a.shape[1]
		local n = aRows * aCols
		local ad, ao = a.data, a.offset * 8
		local bd, bo = b.data, b.offset * 8
		local out = alloc(n)

		local o = 0
		for _ = 1, aRows do
			local bv = bo
			for _ = 1, aCols do
				local x, y = b_read(ad, ao), b_read(bd, bv)
				if op == 1 then b_write(out, o, x + y)
				elseif op == 2 then b_write(out, o, x - y)
				elseif op == 3 then b_write(out, o, x * y)
				else b_write(out, o, x / y) end
				o += 8; ao += 8; bv += 8
			end
		end

		local res = newResult(out, { aRows, aCols })
		if tracking2(a, b) then
			local aStart = a.offset * 8
			markTracked(res, { a, b })
			res._backward = function()
				local g = res.grad :: buffer
				if a.requiresGrad then
					local ga = alloc(n)
					local go = 0
					if op == 1 or op == 2 then
						b_copy(ga, 0, g, 0, n * 8)
					else
						for _ = 1, aRows do
							local bv = bo
							for _ = 1, aCols do
								local y = b_read(bd, bv)
								b_write(ga, go, if op == 3 then b_read(g, go) * y else b_read(g, go) / y)
								go += 8; bv += 8
							end
						end
					end
					accumulate(a, ga, n)
				end
				if b.requiresGrad then
					-- the broadcast dimension sums away
					local gb = alloc(aCols)
					local go, av = 0, aStart
					for _ = 1, aRows do
						local bv, gbo = bo, 0
						for _ = 1, aCols do
							local gi = b_read(g, go)
							local add
							if op == 1 then add = gi
							elseif op == 2 then add = -gi
							elseif op == 3 then add = gi * b_read(ad, av)
							else
								local y = b_read(bd, bv)
								add = -gi * b_read(ad, av) / (y * y)
							end
							b_write(gb, gbo, b_read(gb, gbo) + add)
							go += 8; av += 8; bv += 8; gbo += 8
						end
					end
					accumulate(b, gb, aCols)
				end
			end
		end
		return res
	end

	-- path 3: general strided broadcast
	local shape = broadcastShape(a.shape, b.shape)
	local sa, sb = broadcastStride(a, shape), broadcastStride(b, shape)
	local n = numelOf(shape)
	local ndim = #shape
	local ad, bd = a.data, b.data
	local ao, bo = a.offset, b.offset

	local track = tracking2(a, b)
	local out = alloc(n)
	local offA: {number}? = if track then t_create(n) else nil
	local offB: {number}? = if track then t_create(n) else nil

	local idx = t_create(ndim, 0)
	for linear = 0, n - 1 do
		unravel(shape, linear, idx)
		local oa, ob = ao, bo
		for i = 1, ndim do
			local k = idx[i]
			oa += k * sa[i]
			ob += k * sb[i]
		end
		if offA then (offA :: {number})[linear + 1] = oa end
		if offB then (offB :: {number})[linear + 1] = ob end
		local x, y = b_read(ad, oa * 8), b_read(bd, ob * 8)
		local o = linear * 8
		if op == 1 then b_write(out, o, x + y)
		elseif op == 2 then b_write(out, o, x - y)
		elseif op == 3 then b_write(out, o, x * y)
		else b_write(out, o, x / y) end
	end

	local res = newResult(out, shape)
	if track then
		local oA = offA :: {number}
		local oB = offB :: {number}
		markTracked(res, { a, b })
		res._backward = function()
			local g = res.grad :: buffer
			if a.requiresGrad then
				local ga = alloc(n)
				for i = 1, n do
					local o = (i - 1) * 8
					local gi = b_read(g, o)
					if op == 1 or op == 2 then
						b_write(ga, o, gi)
					elseif op == 3 then
						b_write(ga, o, gi * b_read(bd, oB[i] * 8))
					else
						b_write(ga, o, gi / b_read(bd, oB[i] * 8))
					end
				end
				local reduced, count = reduceGrad(ga, n, shape, a.shape)
				accumulate(a, reduced, count)
			end
			if b.requiresGrad then
				local gb = alloc(n)
				for i = 1, n do
					local o = (i - 1) * 8
					local gi = b_read(g, o)
					if op == 1 then
						b_write(gb, o, gi)
					elseif op == 2 then
						b_write(gb, o, -gi)
					elseif op == 3 then
						b_write(gb, o, gi * b_read(ad, oA[i] * 8))
					else
						local y = b_read(bd, oB[i] * 8)
						b_write(gb, o, -gi * b_read(ad, oA[i] * 8) / (y * y))
					end
				end
				local reduced, count = reduceGrad(gb, n, shape, b.shape)
				accumulate(b, reduced, count)
			end
		end
	end
	return res
end

--- Generic closure-based binary op, for custom ops such as pow.
local function binaryOp(
	a: Tensor, b: Tensor,
	f: (number, number) -> number,
	da: ((number, number, number) -> number)?,
	db: ((number, number, number) -> number)?
): Tensor
	local shape = if shapesEqual(a.shape, b.shape) then t_clone(a.shape) else broadcastShape(a.shape, b.shape)
	local n = numelOf(shape)
	local ndim = #shape
	local sa, sb = broadcastStride(a, shape), broadcastStride(b, shape)
	local ad, bd = a.data, b.data

	local out = alloc(n)
	local offA = t_create(n)
	local offB = t_create(n)

	local idx = t_create(ndim, 0)
	for linear = 0, n - 1 do
		unravel(shape, linear, idx)
		local oa, ob = a.offset, b.offset
		for i = 1, ndim do
			local k = idx[i]
			oa += k * sa[i]
			ob += k * sb[i]
		end
		offA[linear + 1] = oa
		offB[linear + 1] = ob
		b_write(out, linear * 8, f(b_read(ad, oa * 8), b_read(bd, ob * 8)))
	end

	local res = newResult(out, shape)
	if tracking2(a, b) then
		markTracked(res, { a, b })
		res._backward = function()
			local g = res.grad :: buffer
			if a.requiresGrad and da then
				local ga = alloc(n)
				for i = 1, n do
					local o = (i - 1) * 8
					b_write(ga, o, b_read(g, o) * da(
						b_read(ad, offA[i] * 8), b_read(bd, offB[i] * 8), b_read(out, o)))
				end
				local reduced, count = reduceGrad(ga, n, shape, a.shape)
				accumulate(a, reduced, count)
			end
			if b.requiresGrad and db then
				local gb = alloc(n)
				for i = 1, n do
					local o = (i - 1) * 8
					b_write(gb, o, b_read(g, o) * db(
						b_read(ad, offA[i] * 8), b_read(bd, offB[i] * 8), b_read(out, o)))
				end
				local reduced, count = reduceGrad(gb, n, shape, b.shape)
				accumulate(b, reduced, count)
			end
		end
	end
	return res
end

Tensor.binaryOp = binaryOp

function Tensor.add(a: Tensor, b: any): Tensor
	if type(b) == "number" then return arithTensorScalar(a, b, 1, false) end
	return arith(a, asTensor(b), 1)
end

function Tensor.sub(a: Tensor, b: any): Tensor
	if type(b) == "number" then return arithTensorScalar(a, b, 2, false) end
	return arith(a, asTensor(b), 2)
end

function Tensor.mul(a: Tensor, b: any): Tensor
	if type(b) == "number" then return arithTensorScalar(a, b, 3, false) end
	return arith(a, asTensor(b), 3)
end

function Tensor.div(a: Tensor, b: any): Tensor
	if type(b) == "number" then return arithTensorScalar(a, b, 4, false) end
	return arith(a, asTensor(b), 4)
end

function Tensor.pow(a: Tensor, b: any): Tensor
	return binaryOp(a, asTensor(b), function(x, y) return x ^ y end,
	function(x, y) return y * x ^ (y - 1) end,
	function(x, _, out) return if x > 0 then out * m_log(x) else 0 end)
end

function Tensor.__add(a: any, b: any): Tensor
	if type(a) == "number" then return arithTensorScalar(b, a, 1, true) end
	return Tensor.add(a, b)
end
function Tensor.__sub(a: any, b: any): Tensor
	if type(a) == "number" then return arithTensorScalar(b, a, 2, true) end
	return Tensor.sub(a, b)
end
function Tensor.__mul(a: any, b: any): Tensor
	if type(a) == "number" then return arithTensorScalar(b, a, 3, true) end
	return Tensor.mul(a, b)
end
function Tensor.__div(a: any, b: any): Tensor
	if type(a) == "number" then return arithTensorScalar(b, a, 4, true) end
	return Tensor.div(a, b)
end
function Tensor.__pow(a: any, b: any): Tensor
	return Tensor.pow(asTensor(a), asTensor(b))
end
function Tensor.__unm(a: Tensor): Tensor
	return arithTensorScalar(a, -1, 3, false)
end

-- ==========================================================================
-- ELEMENTWISE UNARY OPS
-- ==========================================================================

-- Opcodes: 1 exp, 2 log, 3 sqrt, 4 abs, 5 relu, 6 tanh, 7 sigmoid,
--          8 clamp(p1,p2), 9 leakyRelu(p1), 10 safeExp(p1)
local function unaryFwd(op: number, src: buffer, so: number, out: buffer, n: number, p1: number, p2: number)
	local s, o = so * 8, 0
	if op == 1 then
		for _ = 1, n do b_write(out, o, m_exp(b_read(src, s))); s += 8; o += 8 end
	elseif op == 2 then
		for _ = 1, n do b_write(out, o, m_log(b_read(src, s))); s += 8; o += 8 end
	elseif op == 3 then
		for _ = 1, n do b_write(out, o, m_sqrt(b_read(src, s))); s += 8; o += 8 end
	elseif op == 4 then
		for _ = 1, n do b_write(out, o, m_abs(b_read(src, s))); s += 8; o += 8 end
	elseif op == 5 then
		for _ = 1, n do
			local x = b_read(src, s)
			b_write(out, o, if x > 0 then x else 0); s += 8; o += 8
		end
	elseif op == 6 then
		for _ = 1, n do b_write(out, o, m_tanh(b_read(src, s))); s += 8; o += 8 end
	elseif op == 7 then
		for _ = 1, n do b_write(out, o, 1 / (1 + m_exp(-b_read(src, s)))); s += 8; o += 8 end
	elseif op == 8 then
		for _ = 1, n do b_write(out, o, m_clamp(b_read(src, s), p1, p2)); s += 8; o += 8 end
	elseif op == 9 then
		for _ = 1, n do
			local x = b_read(src, s)
			b_write(out, o, if x > 0 then x else p1 * x); s += 8; o += 8
		end
	else
		for _ = 1, n do b_write(out, o, m_exp(m_min(b_read(src, s), p1))); s += 8; o += 8 end
	end
end

local function unaryBwd(op: number, g: buffer, src: buffer, so: number, out: buffer, ga: buffer, n: number, p1: number, p2: number)
	local s, o = so * 8, 0
	if op == 1 then
		for _ = 1, n do b_write(ga, o, b_read(g, o) * b_read(out, o)); o += 8 end
	elseif op == 2 then
		for _ = 1, n do b_write(ga, o, b_read(g, o) / b_read(src, s)); s += 8; o += 8 end
	elseif op == 3 then
		for _ = 1, n do b_write(ga, o, b_read(g, o) * 0.5 / b_read(out, o)); o += 8 end
	elseif op == 4 then
		for _ = 1, n do
			local x = b_read(src, s)
			b_write(ga, o, if x > 0 then b_read(g, o) elseif x < 0 then -b_read(g, o) else 0)
			s += 8; o += 8
		end
	elseif op == 5 then
		for _ = 1, n do
			b_write(ga, o, if b_read(src, s) > 0 then b_read(g, o) else 0); s += 8; o += 8
		end
	elseif op == 6 then
		for _ = 1, n do
			local y = b_read(out, o)
			b_write(ga, o, b_read(g, o) * (1 - y * y)); o += 8
		end
	elseif op == 7 then
		for _ = 1, n do
			local y = b_read(out, o)
			b_write(ga, o, b_read(g, o) * y * (1 - y)); o += 8
		end
	elseif op == 8 then
		for _ = 1, n do
			local x = b_read(src, s)
			b_write(ga, o, if x < p1 or x > p2 then 0 else b_read(g, o)); s += 8; o += 8
		end
	elseif op == 9 then
		for _ = 1, n do
			b_write(ga, o, if b_read(src, s) > 0 then b_read(g, o) else b_read(g, o) * p1)
			s += 8; o += 8
		end
	else
		for _ = 1, n do
			b_write(ga, o, if b_read(src, s) > p1 then 0 else b_read(g, o) * b_read(out, o))
			s += 8; o += 8
		end
	end
end

local function unaryDirect(a: Tensor, op: number, p1: number?, p2: number?): Tensor
	local n = numelOf(a.shape)
	local out = alloc(n)
	local q1 = p1 or 0
	local q2 = p2 or 0
	local src, so = operand(a)

	unaryFwd(op, src, so, out, n, q1, q2)

	local res = newResult(out, t_clone(a.shape))
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			unaryBwd(op, g, src, so, out, ga, n, q1, q2)
			accumulate(a, ga, n)
		end
	end
	return res
end

--- Generic closure-based unary op, for custom ops.
local function unaryOp(a: Tensor, f: (number) -> number, df: (number, number) -> number): Tensor
	local n = numelOf(a.shape)
	local out = alloc(n)
	local src, so = operand(a)

	local s, o = so * 8, 0
	for _ = 1, n do
		b_write(out, o, f(b_read(src, s))); s += 8; o += 8
	end

	local res = newResult(out, t_clone(a.shape))
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			local si, oi = so * 8, 0
			for _ = 1, n do
				b_write(ga, oi, b_read(g, oi) * df(b_read(src, si), b_read(out, oi)))
				si += 8; oi += 8
			end
			accumulate(a, ga, n)
		end
	end
	return res
end

Tensor.unaryOp = unaryOp

function Tensor.exp(a: Tensor): Tensor return unaryDirect(a, 1) end
function Tensor.log(a: Tensor): Tensor return unaryDirect(a, 2) end
function Tensor.sqrt(a: Tensor): Tensor return unaryDirect(a, 3) end
function Tensor.abs(a: Tensor): Tensor return unaryDirect(a, 4) end
function Tensor.relu(a: Tensor): Tensor return unaryDirect(a, 5) end
function Tensor.tanh(a: Tensor): Tensor return unaryDirect(a, 6) end
function Tensor.sigmoid(a: Tensor): Tensor return unaryDirect(a, 7) end

function Tensor.neg(a: Tensor): Tensor
	return arithTensorScalar(a, -1, 3, false)
end

function Tensor.clamp(a: Tensor, lo: number?, hi: number?): Tensor
	return unaryDirect(a, 8, lo or -m_huge, hi or m_huge)
end

function Tensor.leakyRelu(a: Tensor, slope: number?): Tensor
	return unaryDirect(a, 9, slope or 0.01)
end

function Tensor.safeExp(a: Tensor, cap: number?): Tensor
	return unaryDirect(a, 10, cap or 60)
end

-- ==========================================================================
-- REDUCTIONS
-- ==========================================================================

--[[
	For a reduction along dimension d, split the index space into
	    outer = prod(shape[1..d-1])  size = shape[d]  inner = prod(shape[d+1..])
	so element (o,s,j) sits at o*size*inner + s*inner + j. Three integer
	loops, no per-element unravel.
]]
function Tensor.sum(a: Tensor, dim: number?, keepdim: boolean?): Tensor
	local ndim = #a.shape
	local n = numelOf(a.shape)

	if not dim then
		local src, so = operand(a)
		local total = 0
		local o = so * 8
		for _ = 1, n do
			total += b_read(src, o); o += 8
		end
		local out = alloc(1)
		b_write(out, 0, total)

		local res = newResult(out, { 1 })
		if tracking1(a) then
			markTracked(res, { a })
			res._backward = function()
				local g = b_read(res.grad :: buffer, 0)
				local ga = alloc(n)
				fillBuffer(ga, g, n)
				accumulate(a, ga, n)
			end
		end
		return res
	end

	local d = dim :: number
	if d < 1 or d > ndim then
		error("sum: dim out of range", 2)
	end

	local size = a.shape[d]
	local inner = 1
	for i = d + 1, ndim do inner *= a.shape[i] end
	local outer = n // (size * inner)
	local outCount = outer * inner

	local out = alloc(outCount)
	local src, so = operand(a)

	for o = 0, outer - 1 do
		local srcBase = (so + o * size * inner) * 8
		local outBase = o * inner * 8
		for _ = 1, size do
			local oi = outBase
			for _ = 1, inner do
				b_write(out, oi, b_read(out, oi) + b_read(src, srcBase))
				oi += 8; srcBase += 8
			end
		end
	end

	local finalShape
	if keepdim then
		finalShape = t_clone(a.shape)
		finalShape[d] = 1
	else
		finalShape = {}
		for i = 1, ndim do
			if i ~= d then table.insert(finalShape, a.shape[i]) end
		end
		if #finalShape == 0 then finalShape = { 1 } end
	end

	local res = newResult(out, finalShape)
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			local gi = 0
			for o = 0, outer - 1 do
				local gBase = o * inner * 8
				for _ = 1, size do
					local gb = gBase
					for _ = 1, inner do
						b_write(ga, gi, b_read(g, gb)); gi += 8; gb += 8
					end
				end
			end
			accumulate(a, ga, n)
		end
	end
	return res
end

function Tensor.mean(a: Tensor, dim: number?, keepdim: boolean?): Tensor
	local count = if dim then a.shape[dim] else numelOf(a.shape)
	--[[
		An empty tensor divided 0/0 to nan and returned it. sum() answering 0
		is defensible; mean() has no answer, and handing back a nan that
		poisons everything downstream is the worst of the three options.
		argmax already errors on empty for the same reason.
	]]
	if count == 0 then
		error("mean: cannot average zero elements", 2)
	end
	return Tensor.div(Tensor.sum(a, dim, keepdim), count)
end

function Tensor.max(a: Tensor, dim: number?): (Tensor, {number})
	local n = numelOf(a.shape)
	local vals = a:toFlat()

	if not dim then
		local best, bestIndex = b_read(vals, 0), 1
		for i = 2, n do
			local v = b_read(vals, (i - 1) * 8)
			if v > best then best, bestIndex = v, i end
		end
		local out = alloc(1)
		b_write(out, 0, best)

		local res = newResult(out, { 1 })
		if tracking1(a) then
			markTracked(res, { a })
			res._backward = function()
				local g = b_read(res.grad :: buffer, 0)
				local ga = alloc(n)
				b_write(ga, (bestIndex - 1) * 8, g)
				accumulate(a, ga, n)
			end
		end
		return res, { bestIndex }
	end

	local ndim = #a.shape
	local d = dim :: number
	local size = a.shape[d]
	local inner = 1
	for i = d + 1, ndim do inner *= a.shape[i] end
	local outer = n // (size * inner)
	local outCount = outer * inner

	local out = alloc(outCount)
	fillBuffer(out, -m_huge, outCount)
	local argmax = t_create(outCount, 1)

	for o = 0, outer - 1 do
		local srcBase = o * size * inner
		local outBase = o * inner
		for s = 0, size - 1 do
			local rowBase = srcBase + s * inner
			for j = 1, inner do
				local v = b_read(vals, (rowBase + j - 1) * 8)
				local oi = (outBase + j - 1) * 8
				if v > b_read(out, oi) then
					b_write(out, oi, v)
					argmax[outBase + j] = rowBase + j
				end
			end
		end
	end

	local finalShape = {}
	for i = 1, ndim do
		if i ~= d then table.insert(finalShape, a.shape[i]) end
	end
	if #finalShape == 0 then finalShape = { 1 } end

	local res = newResult(out, finalShape)
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			for o = 1, outCount do
				local target = (argmax[o] - 1) * 8
				b_write(ga, target, b_read(ga, target) + b_read(g, (o - 1) * 8))
			end
			accumulate(a, ga, n)
		end
	end
	return res, argmax
end

function Tensor.argmax(a: Tensor): number
	if numelOf(a.shape) == 0 then
		error("argmax: tensor is empty", 2)
	end
	local n = numelOf(a.shape)
	local vals = a:toFlat()
	local best, bestIndex = b_read(vals, 0), 1
	for i = 2, n do
		local v = b_read(vals, (i - 1) * 8)
		if v > best then best, bestIndex = v, i end
	end
	return bestIndex
end

-- ==========================================================================
-- SOFTMAX / LOGSOFTMAX
-- ==========================================================================

function Tensor.softmax(a: Tensor, dim: number?): Tensor
	local ndim = #a.shape
	local d = dim or ndim
	if d < 1 or d > ndim then
		error("softmax: dim out of range", 2)
	end

	local n = numelOf(a.shape)
	local size = a.shape[d]

	-- fast path: last dimension of a contiguous tensor
	if d == ndim and a._contig then
		local ad, ao = a.data, a.offset * 8
		local rows = n // size
		local out = alloc(n)

		local obase = 0
		for _ = 1, rows do
			local base = ao
			local mx = -m_huge
			for _ = 1, size do
				local v = b_read(ad, base)
				if v > mx then mx = v end
				base += 8
			end

			base = ao
			local total = 0
			local o = obase
			for _ = 1, size do
				local e = m_exp(b_read(ad, base) - mx)
				b_write(out, o, e)
				total += e
				base += 8; o += 8
			end

			local inv = 1 / total
			o = obase
			for _ = 1, size do
				b_write(out, o, b_read(out, o) * inv); o += 8
			end

			ao += size * 8
			obase += size * 8
		end

		local res = newResult(out, t_clone(a.shape))
		if tracking1(a) then
			markTracked(res, { a })
			res._backward = function()
				local g = res.grad :: buffer
				local ga = alloc(n)
				local rb = 0
				for _ = 1, rows do
					local dot = 0
					local o = rb
					for _ = 1, size do
						dot += b_read(g, o) * b_read(out, o); o += 8
					end
					o = rb
					for _ = 1, size do
						b_write(ga, o, b_read(out, o) * (b_read(g, o) - dot)); o += 8
					end
					rb += size * 8
				end
				accumulate(a, ga, n)
			end
		end
		return res
	end

	-- general path: any dimension
	local vals = a:toFlat()
	local reducedShape = t_clone(a.shape)
	reducedShape[d] = 1
	local groupStride = stridesFor(reducedShape)
	local groupCount = numelOf(reducedShape)

	local groups = t_create(groupCount)
	for i = 1, groupCount do groups[i] = t_create(size) end

	local idx = t_create(ndim, 0)
	for linear = 0, n - 1 do
		unravel(a.shape, linear, idx)
		local off = 0
		for i = 1, ndim do
			off += (if i == d then 0 else idx[i]) * groupStride[i]
		end
		table.insert(groups[off + 1], linear + 1)
	end

	local out = alloc(n)
	for gi = 1, groupCount do
		local members = groups[gi]
		local mx = -m_huge
		for _, p in members do
			local v = b_read(vals, (p - 1) * 8)
			if v > mx then mx = v end
		end
		local total = 0
		for _, p in members do
			local e = m_exp(b_read(vals, (p - 1) * 8) - mx)
			b_write(out, (p - 1) * 8, e)
			total += e
		end
		for _, p in members do
			local o = (p - 1) * 8
			b_write(out, o, b_read(out, o) / total)
		end
	end

	local res = newResult(out, t_clone(a.shape))
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			for gi = 1, groupCount do
				local members = groups[gi]
				local dot = 0
				for _, p in members do
					local o = (p - 1) * 8
					dot += b_read(g, o) * b_read(out, o)
				end
				for _, p in members do
					local o = (p - 1) * 8
					b_write(ga, o, b_read(out, o) * (b_read(g, o) - dot))
				end
			end
			accumulate(a, ga, n)
		end
	end
	return res
end

--- Fused log-softmax. One node, and better numerics for small probabilities
--- than log(softmax(x)).
function Tensor.logSoftmax(a: Tensor, dim: number?): Tensor
	local ndim = #a.shape
	local d = dim or ndim

	if d == ndim and a._contig then
		local n = numelOf(a.shape)
		local size = a.shape[d]
		local ad, ao = a.data, a.offset * 8
		local rows = n // size
		local out = alloc(n)

		local obase = 0
		for _ = 1, rows do
			local base = ao
			local mx = -m_huge
			for _ = 1, size do
				local v = b_read(ad, base)
				if v > mx then mx = v end
				base += 8
			end

			base = ao
			local total = 0
			for _ = 1, size do
				total += m_exp(b_read(ad, base) - mx); base += 8
			end

			local shift = mx + m_log(total)
			base = ao
			local o = obase
			for _ = 1, size do
				b_write(out, o, b_read(ad, base) - shift); base += 8; o += 8
			end

			ao += size * 8
			obase += size * 8
		end

		local res = newResult(out, t_clone(a.shape))
		if tracking1(a) then
			markTracked(res, { a })
			res._backward = function()
				local g = res.grad :: buffer
				local ga = alloc(n)
				local rb = 0
				for _ = 1, rows do
					local gsum = 0
					local o = rb
					for _ = 1, size do
						gsum += b_read(g, o); o += 8
					end
					o = rb
					for _ = 1, size do
						b_write(ga, o, b_read(g, o) - m_exp(b_read(out, o)) * gsum); o += 8
					end
					rb += size * 8
				end
				accumulate(a, ga, n)
			end
		end
		return res
	end

	return Tensor.log(Tensor.softmax(a, dim))
end

--- Negative log-likelihood over {batch, classes} log-probabilities with
--- 1-based integer class targets.
function Tensor.nllLoss(logProbs: Tensor, targets: {number}): Tensor
	local batch = logProbs.shape[1]
	local classes = logProbs.shape[2]
	local src, so = operand(logProbs)

	local total = 0
	for row = 1, batch do
		local t = targets[row]
		if t < 1 or t > classes then
			error(("nllLoss: target %d at row %d is outside 1..%d"):format(t, row, classes), 2)
		end
		total += b_read(src, (so + (row - 1) * classes + t - 1) * 8)
	end
	local out = alloc(1)
	b_write(out, 0, -total / batch)

	local res = newResult(out, { 1 })
	if tracking1(logProbs) then
		markTracked(res, { logProbs })
		res._backward = function()
			local g = b_read(res.grad :: buffer, 0)
			local ga = alloc(batch * classes)
			local scale = -g / batch
			for row = 1, batch do
				b_write(ga, ((row - 1) * classes + targets[row] - 1) * 8, scale)
			end
			accumulate(logProbs, ga, batch * classes)
		end
	end
	return res
end

-- ==========================================================================
-- MATRIX MULTIPLY
-- ==========================================================================

--[[
	ikj order, byte offsets throughout, inner loop unrolled 4x.

	This is ~100% of a training step, so it is where every remaining cycle
	lives. The zero-multiplicand skip costs one compare per (row, input)
	against m operations saved: about 1/m overhead when it never fires
	(tanh) and roughly 2x when it fires half the time (ReLU).
]]
function Tensor.matmul(a: Tensor, b: Tensor): Tensor
	if #a.shape ~= 2 or #b.shape ~= 2 then
		error("matmul expects two 2D tensors", 2)
	end
	local n, k = a.shape[1], a.shape[2]
	local k2, m = b.shape[1], b.shape[2]
	if k ~= k2 then
		error(("matmul shape mismatch: %s @ %s"):format(
			shapeToString(a.shape), shapeToString(b.shape)), 2)
	end

	local av = a:toFlat()
	local bv = b:toFlat()
	local out = alloc(n * m)

	local tail = m - 3
	for i = 0, n - 1 do
		local aRow = i * k * 8
		local oRow = i * m * 8
		for p = 0, k - 1 do
			local aik = b_read(av, aRow + p * 8)
			if aik ~= 0 then
				local bRow = p * m * 8
				local j = 0
				while j < tail do
					local o, w = oRow + j * 8, bRow + j * 8
					b_write(out, o, b_read(out, o) + aik * b_read(bv, w))
					b_write(out, o + 8, b_read(out, o + 8) + aik * b_read(bv, w + 8))
					b_write(out, o + 16, b_read(out, o + 16) + aik * b_read(bv, w + 16))
					b_write(out, o + 24, b_read(out, o + 24) + aik * b_read(bv, w + 24))
					j += 4
				end
				while j < m do
					local o, w = oRow + j * 8, bRow + j * 8
					b_write(out, o, b_read(out, o) + aik * b_read(bv, w))
					j += 1
				end
			end
		end
	end

	local res = newResult(out, { n, m })
	if tracking2(a, b) then
		markTracked(res, { a, b })
		res._backward = function()
			local g = res.grad :: buffer

			if a.requiresGrad then
				local ga = alloc(n * k)
				for i = 0, n - 1 do
					local oRow = i * m * 8
					local aRow = i * k * 8
					for p = 0, k - 1 do
						local bRow = p * m * 8
						local s0, s1, s2, s3 = 0, 0, 0, 0
						local j = 0
						while j < tail do
							local o, w = oRow + j * 8, bRow + j * 8
							s0 += b_read(g, o) * b_read(bv, w)
							s1 += b_read(g, o + 8) * b_read(bv, w + 8)
							s2 += b_read(g, o + 16) * b_read(bv, w + 16)
							s3 += b_read(g, o + 24) * b_read(bv, w + 24)
							j += 4
						end
						local s = s0 + s1 + s2 + s3
						while j < m do
							s += b_read(g, oRow + j * 8) * b_read(bv, bRow + j * 8)
							j += 1
						end
						b_write(ga, aRow + p * 8, s)
					end
				end
				accumulate(a, ga, n * k)
			end

			if b.requiresGrad then
				local gb = alloc(k * m)
				-- i-outer, p-inner so reads of `av` are sequential rather
				-- than strided by k. MEASURED AT 1.01x, not the win it looks
				-- like: the strided read happens once per (i,p) and is
				-- amortised over m inner iterations, so it is ~1.5% of the
				-- work, and the whole working set fits in L2 anyway. Kept
				-- because it is never slower and the stride would start to
				-- matter at large k, but do not expect it to show up.
				for i = 0, n - 1 do
					local oRow = i * m * 8
					local aBase = i * k * 8
					for p = 0, k - 1 do
						local aik = b_read(av, aBase + p * 8)
						if aik ~= 0 then
							local bRow = p * m * 8
							local j = 0
							while j < tail do
								local w, o = bRow + j * 8, oRow + j * 8
								b_write(gb, w, b_read(gb, w) + aik * b_read(g, o))
								b_write(gb, w + 8, b_read(gb, w + 8) + aik * b_read(g, o + 8))
								b_write(gb, w + 16, b_read(gb, w + 16) + aik * b_read(g, o + 16))
								b_write(gb, w + 24, b_read(gb, w + 24) + aik * b_read(g, o + 24))
								j += 4
							end
							while j < m do
								local w = bRow + j * 8
								b_write(gb, w, b_read(gb, w) + aik * b_read(g, oRow + j * 8))
								j += 1
							end
						end
					end
				end
				accumulate(b, gb, k * m)
			end
		end
	end
	return res
end

Tensor.dot = Tensor.matmul

-- ==========================================================================
-- VIEWS
-- ==========================================================================

--[[
	Swap two dimensions, materializing the result.

	THE 2D CASE IS THE HOT ONE and gets a dedicated path. The general path calls
	unravel once per element — a division and modulo per dimension — then walks
	the stride array to build an offset. At {64,64} that measured 156 us to move
	4,096 numbers, which is 38 ns each and within 4% of a full {32,64}@{64,64}
	matmul doing 131,072 multiply-adds.

	For two dimensions none of that is needed: the source offset advances by a
	fixed stride down each column, so it is a plain double loop with no
	division, no index array, and no per-element bookkeeping.

	The old backward was worse than the forward: it called table.clone once per
	element to build a swapped index, allocating 4,096 throwaway tables per
	backward pass on a {64,64} transpose.
]]
function Tensor.transpose(a: Tensor, dim1: number?, dim2: number?): Tensor
	local d1 = dim1 or 1
	local d2 = dim2 or 2
	if #a.shape < 2 then
		error("transpose needs at least 2 dimensions", 2)
	end

	local shape = t_clone(a.shape)
	shape[d1], shape[d2] = shape[d2], shape[d1]

	local swappedStride = t_clone(a.stride)
	swappedStride[d1], swappedStride[d2] = swappedStride[d2], swappedStride[d1]

	local ndim = #shape
	local nEl = numelOf(shape)
	local out = alloc(nEl)
	local srcData = a.data

	local fast2D = ndim == 2
	local rows, cols = shape[1], shape[2]

	if fast2D then
		-- out[i][j] = a[j][i]; walking j down a column of the source is a
		-- constant stride, so both offsets are pure additions
		local s0, s1 = swappedStride[1], swappedStride[2]
		local o = 0
		local rowBase = a.offset
		for _ = 1, rows do
			local src = rowBase * 8
			local step = s1 * 8
			for _ = 1, cols do
				b_write(out, o, b_read(srcData, src))
				src += step
				o += 8
			end
			rowBase += s0
		end
	else
		local idx = t_create(ndim, 0)
		for linear = 0, nEl - 1 do
			unravel(shape, linear, idx)
			local off = a.offset
			for i = 1, ndim do
				off += idx[i] * swappedStride[i]
			end
			b_write(out, linear * 8, b_read(srcData, off * 8))
		end
	end

	local sourceStride = stridesFor(a.shape)

	local res = newResult(out, shape)
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local back = alloc(nEl)

			if fast2D then
				-- transpose is its own inverse, so the gradient is the same
				-- walk with source and destination exchanged
				local t0, t1 = sourceStride[1], sourceStride[2]
				local k = 0
				local colBase = 0
				for _ = 1, rows do
					local dst = colBase * 8
					local step = t0 * 8
					for _ = 1, cols do
						b_write(back, dst, b_read(back, dst) + b_read(g, k))
						dst += step
						k += 8
					end
					colBase += t1
				end
			else
				local bidx = t_create(ndim, 0)
				local sidx = t_create(ndim, 0)
				for linear = 0, nEl - 1 do
					unravel(shape, linear, bidx)
					-- reuse one index table instead of cloning per element
					for i = 1, ndim do sidx[i] = bidx[i] end
					sidx[d1], sidx[d2] = bidx[d2], bidx[d1]
					local off = 0
					for i = 1, ndim do
						off += sidx[i] * sourceStride[i]
					end
					local o = off * 8
					b_write(back, o, b_read(back, o) + b_read(g, linear * 8))
				end
			end
			accumulate(a, back, nEl)
		end
	end
	return res
end

function Tensor.reshape(a: Tensor, shape: {number}): Tensor
	local n = numelOf(a.shape)
	if numelOf(shape) ~= n then
		error(("reshape: %s has %d elements, %s needs %d"):format(
			shapeToString(a.shape), n, shapeToString(shape), numelOf(shape)), 2)
	end

	local flat = alloc(n)
	b_copy(flat, 0, a:toFlat(), 0, n * 8)

	local res = newResult(flat, t_clone(shape))
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			accumulate(a, res.grad :: buffer, n)
		end
	end
	return res
end

function Tensor.flatten(a: Tensor): Tensor
	return Tensor.reshape(a, { numelOf(a.shape) })
end

function Tensor.unsqueeze(a: Tensor, dim: number): Tensor
	local shape = t_clone(a.shape)
	table.insert(shape, dim, 1)
	return Tensor.reshape(a, shape)
end

function Tensor.squeeze(a: Tensor, dim: number?): Tensor
	local shape = {}
	for i = 1, #a.shape do
		local keep = if dim then (i ~= dim or a.shape[i] ~= 1) else a.shape[i] ~= 1
		if keep then table.insert(shape, a.shape[i]) end
	end
	if #shape == 0 then shape = { 1 } end
	return Tensor.reshape(a, shape)
end

function Tensor.select(a: Tensor, dim: number, index: number): Tensor
	local ndim = #a.shape
	if dim < 1 or dim > ndim then
		error("select: dim out of range", 2)
	end
	if index < 1 or index > a.shape[dim] then
		error("select: index out of range", 2)
	end

	local nvals = numelOf(a.shape)
	local vals = a:toFlat()

	local outShape = {}
	for i = 1, ndim do
		if i ~= dim then table.insert(outShape, a.shape[i]) end
	end
	if #outShape == 0 then outShape = { 1 } end

	local size = a.shape[dim]
	local inner = 1
	for i = dim + 1, ndim do inner *= a.shape[i] end
	local outer = nvals // (size * inner)

	local outCount = outer * inner
	local out = alloc(outCount)
	local sBase = (index - 1) * inner

	for o = 0, outer - 1 do
		local srcBase = (o * size * inner + sBase) * 8
		local outBase = o * inner * 8
		b_copy(out, outBase, vals, srcBase, inner * 8)
	end

	local res = newResult(out, outShape)
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(nvals)
			for o = 0, outer - 1 do
				local srcBase = (o * size * inner + sBase) * 8
				local outBase = o * inner * 8
				for j = 0, inner - 1 do
					local s = srcBase + j * 8
					b_write(ga, s, b_read(ga, s) + b_read(g, outBase + j * 8))
				end
			end
			accumulate(a, ga, nvals)
		end
	end
	return res
end

function Tensor.narrow(a: Tensor, dim: number, from: number, count: number): Tensor
	local ndim = #a.shape
	if dim < 1 or dim > ndim then
		error("narrow: dim out of range", 2)
	end
	if from < 1 or from + count - 1 > a.shape[dim] then
		error(("narrow: range %d..%d is outside dimension %d of size %d")
			:format(from, from + count - 1, dim, a.shape[dim]), 2)
	end

	local nvals = numelOf(a.shape)
	local vals = a:toFlat()
	local outShape = t_clone(a.shape)
	outShape[dim] = count

	local size = a.shape[dim]
	local inner = 1
	for i = dim + 1, ndim do inner *= a.shape[i] end
	local outer = nvals // (size * inner)

	local block = count * inner
	local outCount = outer * block
	local out = alloc(outCount)

	for o = 0, outer - 1 do
		local srcBase = (o * size * inner + (from - 1) * inner) * 8
		b_copy(out, o * block * 8, vals, srcBase, block * 8)
	end

	local res = newResult(out, outShape)
	if tracking1(a) then
		markTracked(res, { a })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(nvals)
			for o = 0, outer - 1 do
				local srcBase = (o * size * inner + (from - 1) * inner) * 8
				local outBase = o * block * 8
				for j = 0, block - 1 do
					local s = srcBase + j * 8
					b_write(ga, s, b_read(ga, s) + b_read(g, outBase + j * 8))
				end
			end
			accumulate(a, ga, nvals)
		end
	end
	return res
end

-- ==========================================================================
-- IN-PLACE (no autograd; optimisers and initialisation only)
-- ==========================================================================

--[[
	In-place mutation has no backward rule, so it refuses to run on a tensor
	the graph still references. Mutating a result would change the values a
	_backward closure reads later, and the gradients would be quietly wrong.

	`level` is the stack level to blame: 3 when called through the shared
	inPlace helper, 2 when a public function calls this directly.
]]
local function assertMutable(a: Tensor, name: string, level: number?)
	if not a._isLeaf then
		error(name .. " cannot be used on a tensor produced by an operation", level or 3)
	end
end

local function inPlace(a: Tensor, value: any, op: number, name: string): Tensor
	assertMutable(a, name)
	local n = numelOf(a.shape)
	local d = a.data
	local o = a.offset * 8

	if type(value) == "number" then
		for _ = 1, n do
			local x = b_read(d, o)
			b_write(d, o, if op == 1 then x + value elseif op == 2 then x - value else x * value)
			o += 8
		end
	else
		local v = (value :: Tensor):toFlat()
		local vo = 0
		for _ = 1, n do
			local x, y = b_read(d, o), b_read(v, vo)
			b_write(d, o, if op == 1 then x + y elseif op == 2 then x - y else x * y)
			o += 8; vo += 8
		end
	end
	return a
end

function Tensor.addInPlace(a: Tensor, value: any): Tensor
	return inPlace(a, value, 1, "addInPlace")
end

function Tensor.subInPlace(a: Tensor, value: any): Tensor
	return inPlace(a, value, 2, "subInPlace")
end

function Tensor.mulInPlace(a: Tensor, value: any): Tensor
	return inPlace(a, value, 3, "mulInPlace")
end

function Tensor.fillInPlace(a: Tensor, value: number): Tensor
	assertMutable(a, "fillInPlace", 2)
	local n = numelOf(a.shape)
	if a.offset == 0 and a._contig then
		fillBuffer(a.data, value, n)
	else
		local o = a.offset * 8
		for _ = 1, n do b_write(a.data, o, value); o += 8 end
	end
	return a
end

function Tensor.clampInPlace(a: Tensor, lo: number, hi: number): Tensor
	assertMutable(a, "clampInPlace", 2)
	local n = numelOf(a.shape)
	local o = a.offset * 8
	for _ = 1, n do
		b_write(a.data, o, m_clamp(b_read(a.data, o), lo, hi)); o += 8
	end
	return a
end

function Tensor.copyFrom(a: Tensor, other: Tensor): Tensor
	assertMutable(a, "copyFrom", 2)
	local n = numelOf(a.shape)
	local m = numelOf(other.shape)
	if m ~= n then
		error(("copyFrom: source %s has %d elements, destination %s has %d")
			:format(shapeToString(other.shape), m, shapeToString(a.shape), n), 2)
	end
	b_copy(a.data, a.offset * 8, other:toFlat(), 0, n * 8)
	return a
end

-- ==========================================================================
-- GRADCHECK
-- ==========================================================================

--- Compare analytic gradients to a central finite difference.
--- `fn` must be deterministic: it is called three times per element.
--- @return boolean, number -- passed, worst relative error
function Tensor.gradcheck(fn: (...Tensor) -> Tensor, ...: Tensor): (boolean, number)
	local inputs = { ... }
	local epsilon = 1e-5
	local tolerance = 1e-4

	local savedRequires = {}
	local savedGrad = {}
	local savedStale = {}
	for i, t in inputs do
		savedRequires[i] = t.requiresGrad
		savedGrad[i] = t.grad
		savedStale[i] = t._gradStale
		t.requiresGrad = true
		t.grad = alloc(numelOf(t.shape))
		t._gradStale = false
	end

	local out = fn(table.unpack(inputs))
	assert(numelOf(out.shape) == 1, "gradcheck: fn must return a scalar tensor")
	out:backward()

	local analytic = {}
	for i, t in inputs do
		analytic[i] = tableFromBuffer(t.grad :: buffer, numelOf(t.shape), 0)
	end

	local worst = 0
	for ti, t in inputs do
		local n = numelOf(t.shape)
		for i = 1, n do
			local byteOffset = (t.offset + i - 1) * 8
			local original = b_read(t.data, byteOffset)

			local plus, minus
			Config.noGrad(function()
				b_write(t.data, byteOffset, original + epsilon)
				plus = fn(table.unpack(inputs)):item()
				b_write(t.data, byteOffset, original - epsilon)
				minus = fn(table.unpack(inputs)):item()
			end)
			b_write(t.data, byteOffset, original)

			local numeric = (plus - minus) / (2 * epsilon)
			local err = m_abs(numeric - analytic[ti][i]) / m_max(1, m_abs(numeric))
			if err > worst then worst = err end
		end
	end

	for i, t in inputs do
		t.requiresGrad = savedRequires[i]
		t.grad = savedGrad[i]
		t._gradStale = savedStale[i]
	end

	return worst <= tolerance, worst
end

return setmetatable(Tensor, {
	__call = function(_, ...)
		return Tensor.new(...)
	end,
})
