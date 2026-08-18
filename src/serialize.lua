--!strict
--[[
	Nano.serialize — saving and loading trained weights.

	Plain getFlat is enough to restore into an IDENTICAL architecture and
	catastrophic otherwise: it will happily load a 64-hidden genome into a
	48-hidden network and produce confident nonsense. Everything here carries
	the shape list and refuses to load on a mismatch.
]]

local Tensor = require(script.Parent.Tensor)

type Tensor = Tensor.Tensor

local serialize = {}

local FORMAT_VERSION = 1

--- Weights plus an architecture fingerprint, as a plain Lua table.
--- Suitable for DataStores and ModuleScripts.
function serialize.toTable(model: any, metadata: {[string]: any}?): {[string]: any}
	local params = model:parameters()
	local shapes = table.create(#params)
	local values = {}

	for i, p in ipairs(params) do
		shapes[i] = table.clone(p.shape)
		local d = p.data
		local base = #values
		local o = p.offset * 8
		for j = 1, p:numel() do
			values[base + j] = buffer.readf64(d, o)
			o += 8
		end
	end

	return {
		version = FORMAT_VERSION,
		shapes = shapes,
		values = values,
		count = #values,
		metadata = metadata or {},
	}
end

local function fingerprintMismatch(shapes: {{number}}, params: {Tensor}): string?
	if #shapes ~= #params then
		return ("expected %d parameter tensors, model has %d"):format(#shapes, #params)
	end
	for i, shape in ipairs(shapes) do
		local target = params[i].shape
		if #shape ~= #target then
			return ("parameter %d: saved %d dimensions, model has %d"):format(i, #shape, #target)
		end
		for j = 1, #shape do
			if shape[j] ~= target[j] then
				return ("parameter %d dimension %d: saved %d, model has %d"):format(
					i, j, shape[j], target[j])
			end
		end
	end
	return nil
end

function serialize.fromTable(model: any, state: {[string]: any}): boolean
	if state.version ~= FORMAT_VERSION then
		error(("serialize: format version %s, expected %d"):format(
			tostring(state.version), FORMAT_VERSION), 2)
	end

	local params = model:parameters()
	local mismatch = fingerprintMismatch(state.shapes, params)
	if mismatch then
		error("serialize: architecture mismatch — " .. mismatch, 2)
	end

	local values = state.values
	local cursor = 1
	for _, p in ipairs(params) do
		local d = p.data
		local n = p:numel()
		local o = p.offset * 8
		for j = 0, n - 1 do
			buffer.writef64(d, o, values[cursor + j])
			o += 8
		end
		cursor += n
	end
	return true
end

--- Weights as one human-readable string. %.9g is past float32 precision and
--- roughly half the size of full round-trip precision; use base64 if you
--- need bit-exact reproduction.
function serialize.toString(model: any, metadata: {[string]: any}?): string
	local state = serialize.toTable(model, metadata)

	local shapeParts = table.create(#state.shapes)
	for i, shape in ipairs(state.shapes) do
		shapeParts[i] = table.concat(shape, "x")
	end

	local valueParts = table.create(#state.values)
	for i, v in ipairs(state.values) do
		-- 17 significant digits is the shortest that round-trips an f64 exactly;
		-- %.9g silently truncated every saved weight to float32 precision
		valueParts[i] = string.format("%.17g", v)
	end

	return table.concat({
		"nano" .. FORMAT_VERSION,
		table.concat(shapeParts, ","),
		table.concat(valueParts, ","),
	}, "|")
end

function serialize.fromString(model: any, encoded: string): boolean
	local header, shapeText, valueText = encoded:match("^([^|]+)|([^|]*)|(.*)$")
	if not header then
		error("serialize: malformed string", 2)
	end
	if header ~= "nano" .. FORMAT_VERSION then
		error(("serialize: header %s, expected nano%d"):format(header, FORMAT_VERSION), 2)
	end

	local shapes = {}
	for part in (shapeText :: string):gmatch("[^,]+") do
		local dims = {}
		for d in part:gmatch("[^x]+") do
			table.insert(dims, tonumber(d))
		end
		table.insert(shapes, dims)
	end

	local values = {}
	for part in (valueText :: string):gmatch("[^,]+") do
		table.insert(values, tonumber(part))
	end

	return serialize.fromTable(model, {
		version = FORMAT_VERSION,
		shapes = shapes,
		values = values,
		count = #values,
		metadata = {},
	})
end

--- Weights as float32 in a buffer, base64 encoded — about a third the size
--- of the text form. float32 is deliberate: weights carry nowhere near
--- seven significant digits of real information.
local B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

local function encodeBase64(buf: buffer): string
	local length = buffer.len(buf)
	local parts = table.create(math.ceil(length / 3))
	local index = 0

	local i = 0
	while i < length - 2 do
		local a, b, c = buffer.readu8(buf, i), buffer.readu8(buf, i + 1), buffer.readu8(buf, i + 2)
		local n = a * 65536 + b * 256 + c
		index += 1
		parts[index] = B64:sub(n // 262144 + 1, n // 262144 + 1)
			.. B64:sub((n // 4096) % 64 + 1, (n // 4096) % 64 + 1)
			.. B64:sub((n // 64) % 64 + 1, (n // 64) % 64 + 1)
			.. B64:sub(n % 64 + 1, n % 64 + 1)
		i += 3
	end

	local remaining = length - i
	if remaining == 1 then
		local a = buffer.readu8(buf, i)
		local n = a * 65536
		index += 1
		parts[index] = B64:sub(n // 262144 + 1, n // 262144 + 1)
			.. B64:sub((n // 4096) % 64 + 1, (n // 4096) % 64 + 1) .. "=="
	elseif remaining == 2 then
		local a, b = buffer.readu8(buf, i), buffer.readu8(buf, i + 1)
		local n = a * 65536 + b * 256
		index += 1
		parts[index] = B64:sub(n // 262144 + 1, n // 262144 + 1)
			.. B64:sub((n // 4096) % 64 + 1, (n // 4096) % 64 + 1)
			.. B64:sub((n // 64) % 64 + 1, (n // 64) % 64 + 1) .. "="
	end

	return table.concat(parts)
end

local DECODE: {[string]: number} = {}
for i = 1, 64 do
	DECODE[B64:sub(i, i)] = i - 1
end

local function decodeBase64(text: string): buffer
	local clean = text:gsub("=", "")
	local length = #clean
	local bytes = (length * 3) // 4
	local buf = buffer.create(bytes)

	local out = 0
	local i = 1
	while i + 3 <= length do
		local n = DECODE[clean:sub(i, i)] * 262144
			+ DECODE[clean:sub(i + 1, i + 1)] * 4096
			+ DECODE[clean:sub(i + 2, i + 2)] * 64
			+ DECODE[clean:sub(i + 3, i + 3)]
		buffer.writeu8(buf, out, n // 65536)
		buffer.writeu8(buf, out + 1, (n // 256) % 256)
		buffer.writeu8(buf, out + 2, n % 256)
		out += 3
		i += 4
	end

	local remaining = length - i + 1
	if remaining == 2 then
		local n = DECODE[clean:sub(i, i)] * 262144 + DECODE[clean:sub(i + 1, i + 1)] * 4096
		buffer.writeu8(buf, out, n // 65536)
	elseif remaining == 3 then
		local n = DECODE[clean:sub(i, i)] * 262144
			+ DECODE[clean:sub(i + 1, i + 1)] * 4096
			+ DECODE[clean:sub(i + 2, i + 2)] * 64
		buffer.writeu8(buf, out, n // 65536)
		buffer.writeu8(buf, out + 1, (n // 256) % 256)
	end

	return buf
end

function serialize.toBase64(model: any): { shapes: {{number}}, data: string }
	local params = model:parameters()
	local shapes = table.create(#params)
	local total = 0
	for i, p in ipairs(params) do
		shapes[i] = table.clone(p.shape)
		total += p:numel()
	end

	local buf = buffer.create(total * 4)
	local offset = 0
	for _, p in ipairs(params) do
		local d = p.data
		local o = p.offset * 8
		for _ = 1, p:numel() do
			-- f32 on the wire: weights carry nowhere near seven significant
			-- digits, and this halves the payload again
			buffer.writef32(buf, offset, buffer.readf64(d, o))
			offset += 4
			o += 8
		end
	end

	return { shapes = shapes, data = encodeBase64(buf) }
end

function serialize.fromBase64(model: any, payload: { shapes: {{number}}, data: string }): boolean
	local params = model:parameters()
	local mismatch = fingerprintMismatch(payload.shapes, params)
	if mismatch then
		error("serialize: architecture mismatch — " .. mismatch, 2)
	end

	local buf = decodeBase64(payload.data)
	local offset = 0
	for _, p in ipairs(params) do
		local d = p.data
		local o = p.offset * 8
		for _ = 1, p:numel() do
			buffer.writef64(d, o, buffer.readf32(buf, offset))
			offset += 4
			o += 8
		end
	end
	return true
end

--- Save optimizer moment estimates alongside weights. Resuming without them
--- leaves the first updates effectively unscaled, which can knock a
--- converged model out of its minimum on its own reload.
function serialize.optimizerToTable(opt: any): {[string]: any}
	local out: {[string]: any} = {
		version = FORMAT_VERSION,
		stepCount = opt.stepCount,
		lr = opt.lr,
	}
	--[[
		Optimizer state is a list of BUFFERS, not tables.

		This used table.clone / table.move and crashed outright on every
		optimizer that keeps state — "table expected, got buffer". It is a
		survivor of the buffer migration: nothing exercised optimizer
		serialization, so the whole resume-training path was dead.

		Values go out as plain arrays so the result stays JSON-encodable for a
		DataStore, which is the only reason to serialize this at all.
	]]
	for _, field in { "m", "v", "velocity" } do
		local slot = opt[field]
		if slot then
			local copy = {}
			for i, buf in ipairs(slot) do
				local n = buffer.len(buf) // 8
				local arr = table.create(n)
				for j = 1, n do
					arr[j] = buffer.readf64(buf, (j - 1) * 8)
				end
				copy[i] = arr
			end
			out[field] = copy
		end
	end
	return out
end

function serialize.optimizerFromTable(opt: any, state: {[string]: any}): boolean
	opt.stepCount = state.stepCount or 0
	if state.lr then opt.lr = state.lr end
	for _, field in { "m", "v", "velocity" } do
		local saved = state[field]
		local slot = opt[field]
		if saved and slot then
			for i, arr in ipairs(saved) do
				local buf = slot[i]
				if buf then
					local n = math.min(#arr, buffer.len(buf) // 8)
					for j = 1, n do
						buffer.writef64(buf, (j - 1) * 8, arr[j])
					end
				end
			end
		end
	end
	return true
end

return serialize
