# rnn

Recurrent cells, stacks, state helpers and sequence runners.

```lua
local rnn = nano.rnn
```

## State is explicit

Cells are `forward(x, state) -> (output, newState)`. State is **never** stored on the
module.

Module-owned state is the obvious API and wrong for these use cases: RL runs N
environments through one network, each needing its own hidden state, and truncated
backprop needs to detach at a chosen boundary. Both become mutations of the module.
PyTorch made the same call.

```lua
local cell = rnn.LSTM(inputSize, hiddenSize, layers?, dropout?)
local state = rnn.zeroState(cell, batchSize)

local out
out, state = cell:forward(x, state)

state = rnn.detachState(state)      -- at a TBPTT boundary
```

## Cells

| Member | Description |
| --- | --- |
| `rnn.RNNCell(inputSize, hiddenSize, activation?)` | Plain recurrent cell. `activation` is `"tanh"` (default) or `"relu"`. |
| `rnn.LSTMCell(inputSize, hiddenSize)` | LSTM cell with separate hidden and cell state. |
| `rnn.GRUCell(inputSize, hiddenSize)` | Gated recurrent unit. |
| `cell:forward(x, state?)` → `(Tensor, State)` | One timestep. |

## Stacks

Multi-layer wrappers over the cells, with optional dropout between layers.

| Member | Description |
| --- | --- |
| `rnn.LSTM(inputSize, hiddenSize, layers?, dropout?)` | Stacked LSTM. |
| `rnn.GRU(inputSize, hiddenSize, layers?, dropout?)` | Stacked GRU. |
| `rnn.RNN(inputSize, hiddenSize, layers?, dropout?)` | Stacked plain RNN. |
| `rnn.Recurrent(cellType, inputSize, hiddenSize, layers?, dropout?)` | The generic constructor the three above call. `cellType` is `"lstm"`, `"gru"` or `"rnn"`. |

**Start with GRU.** One update gate replaces the LSTM's separate input and forget gates so
they cannot disagree: ~25% fewer parameters and less compute for comparable quality, which
matters on a CPU interpreter. Reach for LSTM only if a measurement says the separate cell
line helps.

## State helpers

| Member | Description |
| --- | --- |
| `rnn.zeroState(cell, batch)` → `State` | A zeroed state matching the cell's shape and layer count. |
| `rnn.detachState(state)` → `State` | Keep the values, cut the graph. |
| `rnn.sliceState(state, index)` → `State` | One environment's state out of a batched one. |
| `rnn.resetRows(state, doneMask)` → `()` | Zero the rows whose environment just reset, in place. |

## Sequence runners

| Member | Description |
| --- | --- |
| `rnn.runSequence(cell, inputs, state?)` → `({Tensor}, State)` | Run a whole sequence and return every output plus the final state. |
| `rnn.runTruncated(cell, inputs, window, onWindow, state?)` → `State` | Run `window` steps, invoke `onWindow(outputs, from, to)` so you can take an optimizer step, then detach and continue. |

```lua
local outputs, finalState = rnn.runSequence(cell, inputs, state)

rnn.runTruncated(cell, inputs, 32, function(outputs, from, to)
    opt:zeroGrad()
    local loss = criterion:forward(F.cat(outputs, 1), targets)
    loss:backward()
    opt:step()
end)
```

`runTruncated` bounds both graph size and backward cost: gradient reaches back `window`
steps and no further, while state still carries across the whole sequence. Without it, a
1000-step sequence builds a 1000-deep graph and one backward pass through all of it.

## Why the defaults are what they are

The LSTM's **forget-gate bias starts at 1**. At 0 the gate is `sigmoid(0) = 0.5`, so cell
state halves every timestep and information is gone within a handful of steps — the
network must learn to remember before it can learn anything requiring memory. Starting it
open makes remembering the default.

The **recurrent matrix is orthogonal per gate block**. One orthogonal `{H, 4H}` is not the
same thing: its columns would be orthogonal across gate boundaries, which is meaningless,
and no individual block would be norm-preserving.

## Using this for RL

For partially observable RL tasks, prefer
[`algorithms.RecurrentPPO`](algorithms.md) over wiring cells up yourself. It handles the
three things that are easy to get wrong — sequence minibatching, storing and replaying the
hidden state at each sequence start, and masking state at episode boundaries — and each of
them produces plausible gradients on a broken objective when done incorrectly.
