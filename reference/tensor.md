# Tensor

Strided buffer storage with reverse-mode autograd. The core of the library.

```lua
local T = nano.Tensor
```

`T` is callable: `T({{1,2},{3,4}})` is the same as `T.new({{1,2},{3,4}})`.

Requiring [`functional`](functional.md) installs further operations onto this table, so
`T.cat` and `F.cat` are the same function. `init.lua` does this for you.

For what the storage layout and the graph actually are, see
[Tensors and autograd](../concepts/autograd.md).

## Properties

Fields on a tensor instance.

| Member | Description |
| --- | --- |
| `t.data: buffer` | Flat f64 storage. **Not a table** — `t.data[i]` returns `nil`, not an error. Use `t:at(i)`. |
| `t.shape: {number}` | Size along each dimension. |
| `t.stride: {number}` | Element step per dimension. Element `(i,j)` is at `offset + (i-1)*stride[1] + (j-1)*stride[2]`, counted in elements. |
| `t.offset: number` | Starting element index into `data`. |
| `t.requiresGrad: boolean` | Whether gradients flow into this tensor. |
| `t.grad: buffer?` | Accumulated gradient, allocated on the first backward pass. |
| `t._prev: {any}` | The input tensors this one was computed from. |
| `t._backward: (() -> ())?` | Closure pushing gradient into `_prev`. |
| `t._isLeaf: boolean` | `true` if you constructed it, `false` if any operation produced it — including under `noGrad`. In-place ops refuse to touch results. |
| `t._contig: boolean` | Whether strides are plain row-major. Every fast path checks this. |
| `t._gradStale: boolean` | Set by `Optimizer:zeroGrad`; the next accumulation overwrites instead of adding. |

## Construction

| Member | Description |
| --- | --- |
| `T.new(data, shape?, requiresGrad?)` → `Tensor` | Build from a nested table, a flat array plus shape, or a single number. |
| `T.fromTable(values, shape?, requiresGrad?)` → `Tensor` | Build from a flat Lua array. |
| `T.zeros(shape, requiresGrad?)` → `Tensor` | Filled with zeros. |
| `T.ones(shape, requiresGrad?)` → `Tensor` | Filled with ones. |
| `T.full(shape, value, requiresGrad?)` → `Tensor` | Filled with `value`. |
| `T.randn(shape, requiresGrad?)` → `Tensor` | Standard normal, from the seeded stream. |
| `T.rand(shape, requiresGrad?)` → `Tensor` | Uniform in `[0, 1)`, from the seeded stream. |
| `T.isTensor(x)` → `boolean` | Whether `x` is a tensor. |

Shapes are `{batch, features}`. A single sample is `T.new({{0.3, 0.7}})` — double braces.

## Reading

| Member | Description |
| --- | --- |
| `t:numel()` → `number` | Total element count. |
| `t:dim()` → `number` | Number of dimensions. |
| `t:item()` → `number` | The single value. Errors unless `numel` is 1. |
| `t:get(i, j, ...)` → `number` | One element, by index per dimension. |
| `t:at(i)` → `number` | One element of the flat row-major layout, 1-based. |
| `t:setAt(i, value)` → `()` | Write one element of the flat layout. |
| `t:toFlat()` → `buffer` | Contiguous **buffer** — the internal accessor. Returns the underlying storage directly when already contiguous at offset 0, so it is free in the common case. |
| `t:toTable()` → `{number}` | Plain **Lua array** — the interop boundary. Always copies. |
| `t:contiguous()` → `Tensor` | A contiguous copy. Always allocates, even when the tensor is already contiguous. |
| `t:isContiguous()` → `boolean` | Whether strides are plain row-major. |
| `tostring(t)` → `string` | Formatted shape and values. |

`toFlat()` returns a buffer. `action[1]` on one raises "attempt to index buffer with
number", and passing it where an array is expected fails silently in any loop using `#`.

## Arithmetic

Either side may be a plain number. Broadcasting follows NumPy rules, and the gradient
reduces correctly back down — `{4,2} + {2}` sums to `{2}`, which is what makes batched
training possible.

| Member | Description |
| --- | --- |
| `T.add(a, b)` · `a + b` → `Tensor` | Elementwise addition. |
| `T.sub(a, b)` · `a - b` → `Tensor` | Elementwise subtraction. |
| `T.mul(a, b)` · `a * b` → `Tensor` | Elementwise multiplication. |
| `T.div(a, b)` · `a / b` → `Tensor` | Elementwise division. |
| `T.pow(a, b)` · `a ^ b` → `Tensor` | Elementwise power. |
| `-a` → `Tensor` | Unary negation. |
| `T.binaryOp(a, b, f, da?, db?)` → `Tensor` | Build a custom differentiable binary op. Derivatives receive `(x, y, out)`. |

Four dispatch paths are chosen once per op: scalar → identical contiguous shapes →
`{n,m}` with `{m}` (the bias case) → general strided broadcast.

## Unary math

| Member | Description |
| --- | --- |
| `T.exp(a)` → `Tensor` | Elementwise `e^x`. |
| `T.log(a)` → `Tensor` | Natural log. |
| `T.sqrt(a)` → `Tensor` | Square root. |
| `T.abs(a)` → `Tensor` | Absolute value. |
| `T.relu(a)` → `Tensor` | `max(0, x)`. |
| `T.tanh(a)` → `Tensor` | Hyperbolic tangent. |
| `T.sigmoid(a)` → `Tensor` | Logistic sigmoid. |
| `T.neg(a)` → `Tensor` | Negation. |
| `T.clamp(a, lo?, hi?)` → `Tensor` | Clamp into range. |
| `T.leakyRelu(a, slope?)` → `Tensor` | Leaky ReLU, `slope` defaults to `0.01`. |
| `T.safeExp(a, cap?)` → `Tensor` | `exp` with the exponent capped, default `60`. |
| `T.unaryOp(a, f, df)` → `Tensor` | Build a custom differentiable unary op. `df` receives `(input, output)`. |

`safeExp` exists because a policy's log-std can spike early in training, and a plain `exp`
gives `inf` → `nan` → a permanently dead network with no error anywhere.

## Reductions

| Member | Description |
| --- | --- |
| `T.sum(a, dim?, keepdim?)` → `Tensor` | Sum, over everything or along one dimension. |
| `T.mean(a, dim?, keepdim?)` → `Tensor` | Mean, over everything or along one dimension. |
| `T.max(a, dim?)` → `(Tensor, {number})` | Maximum values and their argmax indices. |
| `T.argmax(a)` → `number` | Flat index of the largest element. |

## Softmax family

| Member | Description |
| --- | --- |
| `T.softmax(a, dim?)` → `Tensor` | Softmax. `dim` defaults to the **last** dimension, which for `{batch, classes}` is per-row. |
| `T.logSoftmax(a, dim?)` → `Tensor` | Fused log-softmax, better numerics than `log(softmax(x))`. |
| `T.nllLoss(logProbs, targets)` → `Tensor` | Negative log-likelihood. `targets` are 1-based class indices. |

A softmax normalising over the whole tensor makes every row depend on every other —
silently wrong, and hard to see. Leave `dim` alone unless you mean it.

## Matmul and views

All views are differentiable, and all of them **materialize** — none returns a view into
the original storage.

| Member | Description |
| --- | --- |
| `T.matmul(a, b)` · `T.dot(a, b)` → `Tensor` | 2D matrix multiply. Skips zero multiplicands, which pays off on ReLU nets. |
| `T.transpose(a, dim1?, dim2?)` → `Tensor` | Swap two dimensions, defaulting to the last two. |
| `T.reshape(a, shape)` → `Tensor` | Same elements, new shape. |
| `T.flatten(a)` → `Tensor` | Collapse to one dimension. |
| `T.unsqueeze(a, dim)` → `Tensor` | Insert a size-1 dimension. |
| `T.squeeze(a, dim?)` → `Tensor` | Remove size-1 dimensions, or one named dimension. |
| `T.select(a, dim, index)` → `Tensor` | One slice along `dim`, with that dimension removed. |
| `T.narrow(a, dim, from, count)` → `Tensor` | A contiguous range along `dim`. |

## Autograd control

| Member | Description |
| --- | --- |
| `t:backward()` → `()` | Propagate gradients from a **scalar** through the graph. Sorts iteratively, so deep rollouts cannot blow a recursion limit. |
| `t:zeroGrad()` → `()` | Zero this tensor's gradient buffer. |
| `t:markGradStale()` → `()` | Mark the gradient for overwrite on next accumulation instead of zero-filling it. |
| `t:detach()` → `Tensor` | Same values, cut out of the graph. |
| `T.accumulateTable(t, values)` → `()` | Add a Lua array of gradient values into `t.grad`. Errors if the count does not match `numel`. |
| `T.gradcheck(fn, ...)` → `(boolean, number)` | Compare `fn`'s backward pass against a central finite difference. Returns whether it passed and the worst relative error. |

`gradcheck` requires `fn` to be **deterministic** — it calls `fn` three times per element,
so dropout or unseeded sampling makes the comparison meaningless. Build models outside the
closure.

## In-place (leaves only)

These refuse to run on a tensor produced by an operation, rather than quietly corrupting
the graph.

| Member | Description |
| --- | --- |
| `T.addInPlace(a, value)` → `Tensor` | Add in place. |
| `T.subInPlace(a, value)` → `Tensor` | Subtract in place. |
| `T.mulInPlace(a, value)` → `Tensor` | Multiply in place. |
| `T.fillInPlace(a, value)` → `Tensor` | Fill with a constant. |
| `T.clampInPlace(a, lo, hi)` → `Tensor` | Clamp in place. |
| `T.copyFrom(a, other)` → `Tensor` | Copy another tensor's values in. Errors if the element counts differ. |
