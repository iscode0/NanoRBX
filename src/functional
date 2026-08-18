--!strict
--!native
--!optimize 2
--[[
	Nano.functional — additional Tensor primitives.

	Requiring this module INSTALLS these onto Tensor: after it runs once,
	Tensor.cat and F.cat are the same function. Existing Tensor names are
	never overwritten. Kept out of Tensor.lua so a new op cannot break the
	gradchecked core.

	Tensor.lua keeps newResult/markTracked local, so ops here set the graph
	fields by hand: requiresGrad, _prev, _isLeaf, _backward. Every new op
	must do all four.
]]

local Config = require(script.Parent.Config)
local Tensor = require(script.Parent.Tensor)

type Tensor = Tensor.Tensor

local F = {}

local accumulate = Tensor._accumulate
local b_create = buffer.create
local b_read = buffer.readf64
local b_write = buffer.writef64
local b_copy = buffer.copy
local t_create = table.create
local t_clone = table.clone
local m_exp = math.exp
local m_log = math.log
local m_sqrt = math.sqrt
local m_tanh = math.tanh
local m_clamp = math.clamp


local function numelOf(shape: {number}): number
	local n = 1
	for i = 1, #shape do n *= shape[i] end
	return n
end

-- Build an untracked result tensor from a flat buffer.
local function result(flat: buffer, shape: {number}): Tensor
	return Tensor.new(flat, shape, false)
end

-- Storage for n f64 elements, zero filled.
local function alloc(n: number): buffer
	return b_create(n * 8)
end

-- Turn a result into a graph node. Call AFTER building the backward closure's
-- captures, and assign `res._backward` at the call site so the closure
-- captures an immutable binding.
local function track(res: Tensor, prev: {any})
	res.requiresGrad = true
	res._prev = prev
	res._isLeaf = false
end

local function anyRequiresGrad(tensors: {Tensor}): boolean
	if not Config.gradEnabled then return false end
	for _, t in tensors do
		if t.requiresGrad then return true end
	end
	return false
end

local function tracking(a: Tensor): boolean
	return Config.gradEnabled and a.requiresGrad
end

-- Storage plus element offset, avoiding a copy when already contiguous.
local function flatOf(t: Tensor): (buffer, number)
	if t._contig then
		return t.data, t.offset
	end
	return t:toFlat(), 0
end

-- (outer, size, inner) decomposition: element (o,s,j) sits at
-- o*size*inner + s*inner + j. Three integer loops, no per-element unravel.
local function decompose(shape: {number}, dim: number): (number, number, number)
	local ndim = #shape
	local size = shape[dim]
	local inner = 1
	for i = dim + 1, ndim do inner *= shape[i] end
	local outer = 1
	for i = 1, dim - 1 do outer *= shape[i] end
	return outer, size, inner
end


--- Concatenate tensors along a dimension. All other dimensions must agree.
--- @param dim number? -- defaults to 1
function F.cat(tensors: {Tensor}, dim: number?): Tensor
	local d = dim or 1
	local count = #tensors
	if count == 0 then
		error("cat: no tensors given", 2)
	end
	if count == 1 then
		return tensors[1]
	end

	local first = tensors[1]
	local ndim = #first.shape

	-- every dimension except `d` must agree
	local totalSize = 0
	for k = 1, count do
		local s = tensors[k].shape
		if #s ~= ndim then
			error("cat: tensors must have the same number of dimensions", 2)
		end
		for i = 1, ndim do
			if i ~= d and s[i] ~= first.shape[i] then
				error(("cat: dimension %d disagrees between inputs"):format(i), 2)
			end
		end
		totalSize += s[d]
	end

	local outShape = t_clone(first.shape)
	outShape[d] = totalSize

	local outer = 1
	for i = 1, d - 1 do outer *= first.shape[i] end
	local inner = 1
	for i = d + 1, ndim do inner *= first.shape[i] end

	local out = alloc(outer * totalSize * inner)

	-- per-tensor bookkeeping, reused by the backward pass
	local sizes = t_create(count)
	local starts = t_create(count)
	local cursor = 0
	for k = 1, count do
		sizes[k] = tensors[k].shape[d]
		starts[k] = cursor
		cursor += sizes[k]
	end

	for k = 1, count do
		local src, so = flatOf(tensors[k])
		local size = sizes[k]
		local block = size * inner * 8
		for o = 0, outer - 1 do
			b_copy(out,
				(o * totalSize * inner + starts[k] * inner) * 8,
				src, (so + o * size * inner) * 8, block)
		end
	end

	local res = result(out, outShape)
	if anyRequiresGrad(tensors) then
		track(res, tensors)
		res._backward = function()
			local g = res.grad :: buffer
			for k = 1, count do
				local t = tensors[k]
				if t.requiresGrad then
					local size = sizes[k]
					local count_k = outer * size * inner
					local block = size * inner * 8
					local gk = alloc(count_k)
					for o = 0, outer - 1 do
						b_copy(gk, o * size * inner * 8, g,
							(o * totalSize * inner + starts[k] * inner) * 8, block)
					end
					accumulate(t, gk, count_k)
				end
			end
		end
	end
	return res
end

-- Stack along a NEW dimension. {n,m} x k  ->  {k,n,m} at dim 1.
function F.stack(tensors: {Tensor}, dim: number?): Tensor
	local d = dim or 1
	local expanded = t_create(#tensors)
	for i, t in tensors do
		expanded[i] = Tensor.unsqueeze(t, d)
	end
	return F.cat(expanded, d)
end

--- Split into `count` equal parts along a dimension. This is the LSTM gate
--- split: one fused matmul then four slices beats four matmuls.
--- @param dim number? -- defaults to the last dimension
function F.chunk(t: Tensor, count: number, dim: number?): {Tensor}
	local d = dim or #t.shape
	local size = t.shape[d]
	if size % count ~= 0 then
		error(("chunk: dimension %d of size %d is not divisible by %d"):format(d, size, count), 2)
	end
	local each = size // count
	local out = t_create(count)
	for i = 1, count do
		out[i] = Tensor.narrow(t, d, (i - 1) * each + 1, each)
	end
	return out
end

-- Split at explicit sizes: split(t, {2, 3, 5}, 2)
function F.split(t: Tensor, sizes: {number}, dim: number?): {Tensor}
	local d = dim or #t.shape
	local out = t_create(#sizes)
	local cursor = 1
	for i, size in sizes do
		out[i] = Tensor.narrow(t, d, cursor, size)
		cursor += size
	end
	return out
end


--- Pick one element per row along a dimension: gather(logits, 2, {3,1,4}).
--- The workhorse of discrete RL — Q(s, a_taken), log pi(a_taken|s).
--- Backward is a scatter into the chosen slots.
function F.gather(t: Tensor, dim: number, indices: {number}): Tensor
	local outer, size, inner = decompose(t.shape, dim)
	if inner ~= 1 then
		error("gather: only supported on the last dimension for now", 2)
	end
	if #indices ~= outer then
		error(("gather: expected %d indices, got %d"):format(outer, #indices), 2)
	end

	local src, so = flatOf(t)
	local out = alloc(outer)
	for o = 1, outer do
		local idx = indices[o]
		if idx < 1 or idx > size then
			error(("gather: index %d out of range 1..%d at row %d"):format(idx, size, o), 2)
		end
		b_write(out, (o - 1) * 8, b_read(src, (so + (o - 1) * size + idx - 1) * 8))
	end

	local res = result(out, { outer })
	if tracking(t) then
		track(res, { t })
		res._backward = function()
			local g = res.grad :: buffer
			local total = outer * size
			local ga = alloc(total)
			for o = 1, outer do
				b_write(ga, ((o - 1) * size + indices[o] - 1) * 8, b_read(g, (o - 1) * 8))
			end
			accumulate(t, ga, total)
		end
	end
	return res
end

-- One-hot rows. Not differentiable — it produces a constant.
function F.oneHot(indices: {number}, classes: number): Tensor
	local n = #indices
	local flat = alloc(n * classes)
	for i = 1, n do
		b_write(flat, ((i - 1) * classes + indices[i] - 1) * 8, 1)
	end
	return Tensor.new(flat, { n, classes }, false)
end

--- Replace masked elements with a constant. Gradient flows only through
--- kept elements.
--- @param mask {any} -- 0/1 or booleans, flat layout
function F.maskedFill(t: Tensor, mask: {any}, value: number): Tensor
	local n = numelOf(t.shape)
	local src, so = flatOf(t)
	local out = alloc(n)

	-- normalise once, so the backward loop is a plain numeric test
	local keep = t_create(n)
	for i = 1, n do
		local m = mask[i]
		local masked = (m == true) or (m == 1)
		keep[i] = if masked then 0 else 1
		b_write(out, (i - 1) * 8, if masked then value else b_read(src, (so + i - 1) * 8))
	end

	local res = result(out, t_clone(t.shape))
	if tracking(t) then
		track(res, { t })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			for i = 1, n do
				local o = (i - 1) * 8
				b_write(ga, o, b_read(g, o) * keep[i])
			end
			accumulate(t, ga, n)
		end
	end
	return res
end

-- Elementwise select between two tensors. cond is a plain 0/1 array.
function F.where(cond: {any}, a: Tensor, b: Tensor): Tensor
	local n = numelOf(a.shape)
	local ad, ao = flatOf(a)
	local bd, bo = flatOf(b)
	local out = alloc(n)
	local pick = t_create(n)

	for i = 1, n do
		local c = cond[i]
		local takeA = (c == true) or (c == 1)
		pick[i] = if takeA then 1 else 0
		b_write(out, (i - 1) * 8,
			if takeA then b_read(ad, (ao + i - 1) * 8) else b_read(bd, (bo + i - 1) * 8))
	end

	local res = result(out, t_clone(a.shape))
	if anyRequiresGrad({ a, b }) then
		track(res, { a, b })
		res._backward = function()
			local g = res.grad :: buffer
			if a.requiresGrad then
				local ga = alloc(n)
				for i = 1, n do
					local o = (i - 1) * 8
					b_write(ga, o, b_read(g, o) * pick[i])
				end
				accumulate(a, ga, n)
			end
			if b.requiresGrad then
				local gb = alloc(n)
				for i = 1, n do
					local o = (i - 1) * 8
					b_write(gb, o, b_read(g, o) * (1 - pick[i]))
				end
				accumulate(b, gb, n)
			end
		end
	end
	return res
end


--- Variance along a dimension, or over everything. Two-pass form: the
--- E[x^2]-E[x]^2 shortcut can return negative variance for large means.
--- @param unbiased boolean? -- divide by n-1, defaults to false
function F.var(t: Tensor, dim: number?, unbiased: boolean?): Tensor
	local mean = Tensor.mean(t, dim, true)
	local centered = Tensor.sub(t, mean)
	local squared = Tensor.mul(centered, centered)

	local count = if dim then t.shape[dim] else numelOf(t.shape)
	local divisor = if unbiased then count - 1 else count

	local total = Tensor.sum(squared, dim)
	return Tensor.div(total, divisor)
end

function F.std(t: Tensor, dim: number?, unbiased: boolean?): Tensor
	return Tensor.sqrt(Tensor.add(F.var(t, dim, unbiased), 1e-12))
end

-- L2 norm over everything, as a scalar tensor.
function F.norm(t: Tensor): Tensor
	return Tensor.sqrt(Tensor.add(Tensor.sum(Tensor.mul(t, t)), 1e-12))
end

--- Normalise each row to zero mean and unit variance, then scale and shift.
--- Row-wise, so training and inference agree at any batch size.
--[[
	Layer normalisation, fused into ONE graph node.

	The composed form was nine nodes — mean, sub, mul, mean, add, sqrt, div,
	mul, add — each allocating an intermediate tensor and a backward closure
	over every element. At {32,64} that measured 575 us forward+backward, four
	times a whole batch-32 forward through an 8-64-64-2 network. Same shape as
	the Huber loss: node count dominating arithmetic, so removing the nodes
	removes most of the cost.

	Normalises over the LAST dimension, so a {batch, features} tensor is
	normalised per sample. Statistics come from within the sample, which is the
	whole reason to prefer this over BatchNorm for RL.

	xhat and the per-row inverse standard deviation are both kept from the
	forward pass. They cost n + rows elements of scratch and save the backward
	two full reduction passes over the input.

	The backward is the standard closed form. With N the row width and
	istd = 1/sqrt(var + eps):

	    dxhat = dy * gamma
	    dx    = istd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
]]
function F.layerNorm(x: Tensor, gamma: Tensor?, beta: Tensor?, eps: number?): Tensor
	local e = eps or 1e-5
	local shape = x.shape
	local H = shape[#shape]
	if H == 0 then
		error("layerNorm: last dimension is zero", 2)
	end
	local n = numelOf(shape)
	local rows = n // H

	local xd, xo = flatOf(x)

	-- NOT `if gamma then flatOf(gamma) else nil`: an if-expression keeps only
	-- the first return, which would silently drop a non-zero offset
	local gd: buffer? = nil
	local go = 0
	if gamma then gd, go = flatOf(gamma) end
	local bd: buffer? = nil
	local bo = 0
	if beta then bd, bo = flatOf(beta) end

	local out = alloc(n)
	local xhat = alloc(n)
	local istds = alloc(rows)

	local base = xo * 8
	local o = 0
	for r = 0, rows - 1 do
		local sum = 0
		local p = base
		for _ = 1, H do
			sum += b_read(xd, p)
			p += 8
		end
		local mean = sum / H

		local acc = 0
		p = base
		for _ = 1, H do
			local d = b_read(xd, p) - mean
			acc += d * d
			p += 8
		end
		local istd = 1 / m_sqrt(acc / H + e)
		b_write(istds, r * 8, istd)

		p = base
		local gp, bp = go * 8, bo * 8
		for _ = 1, H do
			local h = (b_read(xd, p) - mean) * istd
			b_write(xhat, o, h)
			local v = h
			if gd then v *= b_read(gd, gp); gp += 8 end
			if bd then v += b_read(bd, bp); bp += 8 end
			b_write(out, o, v)
			p += 8; o += 8
		end
		base += H * 8
	end

	local res = result(out, t_clone(shape))

	local needsX = tracking(x)
	local needsG = if gamma then tracking(gamma) else false
	local needsB = if beta then tracking(beta) else false

	if needsX or needsG or needsB then
		local prev: {any} = { x }
		if gamma then table.insert(prev, gamma) end
		if beta then table.insert(prev, beta) end
		track(res, prev)

		res._backward = function()
			local g = res.grad :: buffer

			local gx = if needsX then alloc(n) else nil
			local gg = if needsG then alloc(H) else nil
			local gb = if needsB then alloc(H) else nil

			local rowStart = 0
			for r = 0, rows - 1 do
				-- gamma and beta hold one slot per FEATURE, so they accumulate
				-- down the batch rather than within a row
				if gg or gb then
					local k = rowStart
					local slot = 0
					for _ = 1, H do
						local dy = b_read(g, k)
						if gg then
							b_write(gg, slot, b_read(gg, slot) + dy * b_read(xhat, k))
						end
						if gb then
							b_write(gb, slot, b_read(gb, slot) + dy)
						end
						k += 8; slot += 8
					end
				end

				if gx then
					local sumD, sumDH = 0, 0
					local k = rowStart
					local gp = go * 8
					for _ = 1, H do
						local dxhat = b_read(g, k)
						if gd then dxhat *= b_read(gd, gp); gp += 8 end
						sumD += dxhat
						sumDH += dxhat * b_read(xhat, k)
						k += 8
					end
					local meanD, meanDH = sumD / H, sumDH / H
					local istd = b_read(istds, r * 8)

					k = rowStart
					gp = go * 8
					for _ = 1, H do
						local dxhat = b_read(g, k)
						if gd then dxhat *= b_read(gd, gp); gp += 8 end
						b_write(gx, k, istd * (dxhat - meanD - b_read(xhat, k) * meanDH))
						k += 8
					end
				end

				rowStart += H * 8
			end

			if gx then accumulate(x, gx, n) end
			if gg then accumulate(gamma :: Tensor, gg, H) end
			if gb then accumulate(beta :: Tensor, gb, H) end
		end
	end

	return res
end


local function elementwise(
	t: Tensor,
	f: (number) -> number,
	df: (number, number) -> number
): Tensor
	local n = numelOf(t.shape)
	local src, so = flatOf(t)
	local out = alloc(n)
	local s, o = so * 8, 0
	for _ = 1, n do
		b_write(out, o, f(b_read(src, s)))
		s += 8; o += 8
	end

	local res = result(out, t_clone(t.shape))
	if tracking(t) then
		track(res, { t })
		res._backward = function()
			local g = res.grad :: buffer
			local ga = alloc(n)
			local si, oi = so * 8, 0
			for _ = 1, n do
				b_write(ga, oi, b_read(g, oi) * df(b_read(src, si), b_read(out, oi)))
				si += 8; oi += 8
			end
			accumulate(t, ga, n)
		end
	end
	return res
end

F.elementwise = elementwise

-- softplus: a smooth ReLU, and the numerically safe way to produce a
-- POSITIVE quantity from an unconstrained one. Prefer it to exp() for
-- standard deviations: exp overflows, softplus grows linearly.
function F.softplus(t: Tensor, beta: number?): Tensor
	local b = beta or 1
	return elementwise(t,
		function(x)
			-- the >20 branch is not an approximation; beyond it exp(b*x)
			-- overflows while log(1+exp(b*x)) is indistinguishable from x
			local bx = b * x
			if bx > 20 then return x end
			return m_log(1 + m_exp(bx)) / b
		end,
		function(x, _)
			return 1 / (1 + m_exp(-b * x))
		end)
end

-- GELU, tanh approximation — the one every transformer actually uses.
local GELU_C = 0.7978845608028654   -- sqrt(2/pi)
function F.gelu(t: Tensor): Tensor
	return elementwise(t,
		function(x)
			local inner = GELU_C * (x + 0.044715 * x * x * x)
			return 0.5 * x * (1 + m_tanh(inner))
		end,
		function(x, _)
			local x3 = x * x * x
			local inner = GELU_C * (x + 0.044715 * x3)
			local tanhInner = m_tanh(inner)
			local sech2 = 1 - tanhInner * tanhInner
			local dInner = GELU_C * (1 + 3 * 0.044715 * x * x)
			return 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * dInner
		end)
end

-- SiLU / Swish: x * sigmoid(x)
function F.silu(t: Tensor): Tensor
	return elementwise(t,
		function(x) return x / (1 + m_exp(-x)) end,
		function(x, _)
			local s = 1 / (1 + m_exp(-x))
			return s * (1 + x * (1 - s))
		end)
end

function F.elu(t: Tensor, alpha: number?): Tensor
	local a = alpha or 1
	return elementwise(t,
		function(x) return if x > 0 then x else a * (m_exp(x) - 1) end,
		function(x, out) return if x > 0 then 1 else out + a end)
end

function F.mish(t: Tensor): Tensor
	return elementwise(t,
		function(x)
			local sp = if x > 20 then x else m_log(1 + m_exp(x))
			return x * m_tanh(sp)
		end,
		function(x, _)
			local sp = if x > 20 then x else m_log(1 + m_exp(x))
			local tsp = m_tanh(sp)
			local sig = 1 / (1 + m_exp(-x))
			return tsp + x * (1 - tsp * tsp) * sig
		end)
end

-- Hard sigmoid / hard tanh: piecewise linear, no exp at all. Meaningfully
-- faster on a CPU interpreter, and usually indistinguishable in quality.
function F.hardSigmoid(t: Tensor): Tensor
	return elementwise(t,
		function(x) return math.clamp(0.2 * x + 0.5, 0, 1) end,
		function(x, _) return if x > -2.5 and x < 2.5 then 0.2 else 0 end)
end


-- Outer product of two 1D tensors: {n} x {m} -> {n, m}
function F.outer(a: Tensor, b: Tensor): Tensor
	local n = numelOf(a.shape)
	local m = numelOf(b.shape)
	return Tensor.matmul(Tensor.reshape(a, { n, 1 }), Tensor.reshape(b, { 1, m }))
end

-- Batched dot product over the last dimension: {N,D} . {N,D} -> {N}
function F.batchDot(a: Tensor, b: Tensor): Tensor
	return Tensor.sum(Tensor.mul(a, b), #a.shape)
end

-- Cosine similarity per row.
function F.cosineSimilarity(a: Tensor, b: Tensor, eps: number?): Tensor
	local e = eps or 1e-8
	local dim = #a.shape
	local dot = Tensor.sum(Tensor.mul(a, b), dim)
	local na = Tensor.sqrt(Tensor.add(Tensor.sum(Tensor.mul(a, a), dim), e))
	local nb = Tensor.sqrt(Tensor.add(Tensor.sum(Tensor.mul(b, b), dim), e))
	return Tensor.div(dot, Tensor.mul(na, nb))
end


--- Orthogonal initialisation via Gram-Schmidt. Essential for recurrent
--- layers: W is applied once per timestep, so over T steps signal is
--- multiplied by W^T. Orthogonal keeps every eigenvalue at magnitude 1, so
--- signal rotates instead of exploding or vanishing.
--- @param gain number? -- sqrt(2) with ReLU, 1 with tanh
function F.orthogonal(shape: {number}, gain: number?): Tensor
	local g = gain or 1
	local rows, cols = shape[1], shape[2]
	if not rows or not cols or #shape ~= 2 then
		error("orthogonal: needs a 2D shape", 2)
	end

	local t = Tensor.randn(shape, false)
	local data = t.data
	local function rd(i: number): number return b_read(data, i * 8) end
	local function wr(i: number, v: number) b_write(data, i * 8, v) end

	-- Gram-Schmidt over columns. O(cols^2 * rows), fine at the sizes a Luau
	-- network reaches, and it runs exactly once per layer.
	for c = 0, cols - 1 do
		for prev = 0, c - 1 do
			local dot = 0
			for r = 0, rows - 1 do
				dot += rd(r * cols + c) * rd(r * cols + prev)
			end
			for r = 0, rows - 1 do
				wr(r * cols + c, rd(r * cols + c) - dot * rd(r * cols + prev))
			end
		end

		local norm = 0
		for r = 0, rows - 1 do
			local v = rd(r * cols + c)
			norm += v * v
		end
		norm = m_sqrt(norm)

		if norm < 1e-8 then
			-- degenerate column: replace it rather than dividing by ~0
			for r = 0, rows - 1 do
				wr(r * cols + c, if r == c % rows then 1 else 0)
			end
		else
			local inv = g / norm
			for r = 0, rows - 1 do
				wr(r * cols + c, rd(r * cols + c) * inv)
			end
		end
	end

	return t
end

function F.constant(shape: {number}, value: number, requiresGrad: boolean?): Tensor
	return Tensor.full(shape, value, requiresGrad)
end

function F.uniform(shape: {number}, lo: number, hi: number, requiresGrad: boolean?): Tensor
	local n = numelOf(shape)
	local flat = alloc(n)
	local span = hi - lo
	local o = 0
	for _ = 1, n do
		b_write(flat, o, lo + Config.random() * span)
		o += 8
	end
	return Tensor.new(flat, shape, requiresGrad)
end


--- Fused affine layer: y = x @ W + b, in ONE graph node.
--- The composed form (matmul then add) allocates an extra {n,m} intermediate
--- and a second backward closure per layer. This is the single most executed
--- op in the library, so the saving compounds across every forward pass.
--- @param bias Tensor? -- {m}, omitted for a bias-free layer
function F.linear(x: Tensor, w: Tensor, bias: Tensor?): Tensor
	if #x.shape ~= 2 or #w.shape ~= 2 then
		error("linear: expects 2D input and weight", 2)
	end
	local n, k = x.shape[1], x.shape[2]
	local k2, m = w.shape[1], w.shape[2]
	if k ~= k2 then
		error(("linear: %dx%d input against %dx%d weight"):format(n, k, k2, m), 2)
	end

	local xd = x:toFlat()
	local wd = w:toFlat()
	local bd = if bias then bias:toFlat() else nil

	local out = alloc(n * m)
	local tail = m - 3

	for i = 0, n - 1 do
		local xRow = i * k * 8
		local oRow = i * m * 8
		if bd then
			b_copy(out, oRow, bd, 0, m * 8)
		end
		for p = 0, k - 1 do
			local xip = b_read(xd, xRow + p * 8)
			-- post-ReLU rows are ~half zeros: one compare against m operations
			if xip ~= 0 then
				local wRow = p * m * 8
				local j = 0
				while j < tail do
					local o, wj = oRow + j * 8, wRow + j * 8
					b_write(out, o, b_read(out, o) + xip * b_read(wd, wj))
					b_write(out, o + 8, b_read(out, o + 8) + xip * b_read(wd, wj + 8))
					b_write(out, o + 16, b_read(out, o + 16) + xip * b_read(wd, wj + 16))
					b_write(out, o + 24, b_read(out, o + 24) + xip * b_read(wd, wj + 24))
					j += 4
				end
				while j < m do
					local o = oRow + j * 8
					b_write(out, o, b_read(out, o) + xip * b_read(wd, wRow + j * 8))
					j += 1
				end
			end
		end
	end

	local res = result(out, { n, m })
	local parents = if bias then { x, w, bias } else { x, w }
	if anyRequiresGrad(parents) then
		track(res, parents)
		res._backward = function()
			local g = res.grad :: buffer

			if x.requiresGrad then
				local gx = alloc(n * k)
				for i = 0, n - 1 do
					local oRow = i * m * 8
					local xRow = i * k * 8
					for p = 0, k - 1 do
						local wRow = p * m * 8
						-- four accumulators, so the adds do not form one
						-- serial dependency chain
						local s0, s1, s2, s3 = 0, 0, 0, 0
						local j = 0
						while j < tail do
							local o, wj = oRow + j * 8, wRow + j * 8
							s0 += b_read(g, o) * b_read(wd, wj)
							s1 += b_read(g, o + 8) * b_read(wd, wj + 8)
							s2 += b_read(g, o + 16) * b_read(wd, wj + 16)
							s3 += b_read(g, o + 24) * b_read(wd, wj + 24)
							j += 4
						end
						local s = s0 + s1 + s2 + s3
						while j < m do
							s += b_read(g, oRow + j * 8) * b_read(wd, wRow + j * 8)
							j += 1
						end
						b_write(gx, xRow + p * 8, s)
					end
				end
				accumulate(x, gx, n * k)
			end

			if w.requiresGrad then
				local gw = alloc(k * m)
				-- i-outer so reads of xd are sequential rather than strided.
				-- Measured at 1.01x; see the note in Tensor.matmul.
				for i = 0, n - 1 do
					local oRow = i * m * 8
					local xBase = i * k * 8
					for p = 0, k - 1 do
						local xip = b_read(xd, xBase + p * 8)
						if xip ~= 0 then
							local wRow = p * m * 8
							local j = 0
							while j < tail do
								local wj, o = wRow + j * 8, oRow + j * 8
								b_write(gw, wj, b_read(gw, wj) + xip * b_read(g, o))
								b_write(gw, wj + 8, b_read(gw, wj + 8) + xip * b_read(g, o + 8))
								b_write(gw, wj + 16, b_read(gw, wj + 16) + xip * b_read(g, o + 16))
								b_write(gw, wj + 24, b_read(gw, wj + 24) + xip * b_read(g, o + 24))
								j += 4
							end
							while j < m do
								local wj = wRow + j * 8
								b_write(gw, wj, b_read(gw, wj) + xip * b_read(g, oRow + j * 8))
								j += 1
							end
						end
					end
				end
				accumulate(w, gw, k * m)
			end

			if bias and bias.requiresGrad then
				-- the batch dimension sums away
				local gb = alloc(m)
				for i = 0, n - 1 do
					local oRow = i * m * 8
					for j = 0, m - 1 do
						local o = j * 8
						b_write(gb, o, b_read(gb, o) + b_read(g, oRow + o))
					end
				end
				accumulate(bias, gb, m)
			end
		end
	end
	return res
end

--- Fused mean/sum squared error. Replaces sub -> mul -> mean with one node.
--- @param reduction string? -- "mean" (default) or "sum"
function F.mseLoss(prediction: Tensor, target: Tensor, reduction: string?): Tensor
	local n = numelOf(prediction.shape)
	local pd, po = flatOf(prediction)
	local td, to = flatOf(target)
	local useMean = reduction ~= "sum"

	local total = 0
	local p, t = po * 8, to * 8
	for _ = 1, n do
		local d = b_read(pd, p) - b_read(td, t)
		total += d * d
		p += 8; t += 8
	end
	local out = alloc(1)
	b_write(out, 0, if useMean then total / n else total)

	local res = result(out, { 1 })
	if anyRequiresGrad({ prediction, target }) then
		track(res, { prediction, target })
		res._backward = function()
			local g = b_read(res.grad :: buffer, 0)
			local scale = if useMean then 2 * g / n else 2 * g
			if prediction.requiresGrad then
				local gp = alloc(n)
				local pi, ti, o = po * 8, to * 8, 0
				for _ = 1, n do
					b_write(gp, o, scale * (b_read(pd, pi) - b_read(td, ti)))
					pi += 8; ti += 8; o += 8
				end
				accumulate(prediction, gp, n)
			end
			if target.requiresGrad then
				local gt = alloc(n)
				local pi, ti, o = po * 8, to * 8, 0
				for _ = 1, n do
					b_write(gt, o, -scale * (b_read(pd, pi) - b_read(td, ti)))
					pi += 8; ti += 8; o += 8
				end
				accumulate(target, gt, n)
			end
		end
	end
	return res
end

--- Fused Huber / smooth L1. Gradient is clamp(d, -delta, delta) / n, which is
--- why the linear tail bounds an outlier's influence.
function F.huber(prediction: Tensor, target: Tensor, delta: number?): Tensor
	local d = delta or 1
	local n = numelOf(prediction.shape)
	local pd, po = flatOf(prediction)
	local td, to = flatOf(target)

	local total = 0
	local p, t = po * 8, to * 8
	for _ = 1, n do
		local diff = b_read(pd, p) - b_read(td, t)
		local a = if diff < 0 then -diff else diff
		if a <= d then
			total += 0.5 * diff * diff
		else
			total += d * (a - 0.5 * d)
		end
		p += 8; t += 8
	end
	local out = alloc(1)
	b_write(out, 0, total / n)

	local res = result(out, { 1 })
	if tracking(prediction) then
		track(res, { prediction })
		res._backward = function()
			local g = b_read(res.grad :: buffer, 0) / n
			local gp = alloc(n)
			local pi, ti, o = po * 8, to * 8, 0
			for _ = 1, n do
				local diff = b_read(pd, pi) - b_read(td, ti)
				b_write(gp, o, g * m_clamp(diff, -d, d))
				pi += 8; ti += 8; o += 8
			end
			accumulate(prediction, gp, n)
		end
	end
	return res
end

-- Install onto Tensor. Existing names win, so the core can never be shadowed.
local installed = {}
for name, fn in F do
	if type(fn) == "function" and (Tensor :: any)[name] == nil then
		(Tensor :: any)[name] = fn
		table.insert(installed, name)
	end
end
table.sort(installed)
F._installed = installed

return F
