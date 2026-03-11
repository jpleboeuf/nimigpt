## The most atomic way to train and run inference for a GPT in pure, dependency-free Nim.
## This file is the complete algorithm.
## Everything else is just efficiency.
##
## Faithful port of @karpathy's `microgpt.py` to Nim.
## < https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95 >
##
## Compile and run:
##   nim c -d:release -d:ssl -r nimigpt.nim
##
## Note: if `input.txt` is not present, it will be downloaded automatically.

import std/os         # fileExists, lines
import std/math       # ln, exp, pow
import std/random     # randomize, gauss, shuffle, sample
import std/httpclient # newHttpClient, downloadFile (replaces Python's urllib.request.urlretrieve)
import std/strformat  # &"" string interpolation
import std/strutils   # strip
import std/sequtils   # zip, toSeq, cumsummed
import std/sugar      # collect
import std/sets       # HashSet
import std/tables     # OrderedTable, toOrderedTable
import std/algorithm  # sort
import std/unicode    # Rune, runes, `$`
randomize(42) # Let there be order among chaos.
# Deterministic across Nim runs, but not equivalent to Python's `random.seed(42)`.

# --- Dataset ---

# Let there be a Dataset `docs`: seq[string] of documents (e.g. a list of names)
if not fileExists("input.txt"):
  let namesUrl = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
  let httpClient = newHttpClient()
  httpClient.downloadFile(namesUrl, "input.txt")
  httpClient.close()
var docs = collect:
  for line in lines("input.txt"):
    let s = line.strip()
    if s.len > 0: s
docs.shuffle()
echo &"num docs: {docs.len}"

# --- Tokenizer ---

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
var charSet = initHashSet[Rune]()
for doc in docs:
  for ch in doc.runes: charSet.incl(ch)
# unique characters in the dataset become token ids 0..n-1
var uchars = toSeq(charSet)
uchars.sort(proc (a, b: Rune): int = cmp(int32(a), int32(b))) # Python: `sorted(set(''.join(docs)))`
let BOS = uchars.len # token id for a special Beginning of Sequence (BOS) token
let vocabSize = uchars.len + 1 # total number of unique tokens, +1 is for BOS
echo &"vocab size: {vocabSize}"

# --- Autograd ---

# Let there be Autograd to recursively apply the chain rule through a computation graph

type
  Value = ref object # Nim's `ref object` serves the same purpose as Python's `__slots__`
    data: float64            # scalar value of this node calculated during forward pass
    grad: float64            # derivative of the loss w.r.t. this node, calculated in backward pass
    # Python's `_` prefix marks the next two as internal slots; Nim identifiers can't start with `_`
    children: seq[Value]     # children of this node in the computation graph
    localGrads: seq[float64] # local derivative of this node w.r.t. its children

proc newValue(data: float64, children: seq[Value] = @[], localGrads: seq[float64] = @[]): Value =
  Value(data: data, grad: 0.0, children: children, localGrads: localGrads)

proc `+`(val: Value, other: Value): Value = # __add__ (`isinstance(other, Value)`)
  newValue(val.data + other.data, @[val, other], @[1.0, 1.0])
proc `+`(val: Value, other: float64): Value = val + newValue(other)

proc `*`(val: Value, other: Value): Value = # __mul__ (`isinstance(other, Value)`)
  newValue(val.data * other.data, @[val, other], @[other.data, val.data])
proc `*`(val: Value, other: float64): Value = val * newValue(other)

# In this Nim implementation, `**` has the same precedence as `*` and `/` - use parentheses!
proc `**`(x, y: float64): float64 = pow(x, y)

proc `**`(val: Value, other: float64): Value = # __pow__
  newValue(val.data ** other, @[val], @[other * (val.data ** (other - 1))])

proc log(val: Value): Value = # log
  newValue(ln(val.data), @[val], @[1.0 / val.data])

proc exp(val: Value): Value = # exp
  newValue(math.exp(val.data), @[val], @[math.exp(val.data)])

proc relu(val: Value): Value = # relu (Rectified Linear Unit)
  # In Python, `float(self.data > 0)` resolves to 1.0 when `float(True)`, 0.0 when `float(False)`
  newValue(max(0.0, val.data), @[val], @[float64(ord(val.data > 0))])

# In Python, `__r*__` methods (e.g. `__radd__`) are called when a non-Value is on the left:
#   `2.0 + v` → `v.__radd__(2.0)` where `self` is `v` (the Value).
# Python swaps the receiver so `self` is always first.
# Nim doesn't, parameter order must match expression order,
#  so `val` (Python's `self`) appears second in the `__r*__` procs below.

proc `-`(val: Value): Value = val * (-1.0) # __neg__

proc `+`(other: float64, val: Value): Value = val + other # __radd__(self, other)

proc `-`(val: Value, other: Value | float64): Value = val + (-other) # __sub__

proc `-`(other: float64, val: Value): Value = other + (-val) # __rsub__(self, other)

proc `*`(other: float64, val: Value): Value = val * other # __rmul__(self, other)

proc `/`(val: Value, other: Value | float64): Value = val * (other ** (-1.0)) # __truediv__

proc `/`(other: float64, val: Value): Value = other * (val ** (-1.0)) # __rtruediv__(self, other)

# Nim's `sum` only works on numeric types, not `Value`
proc sum(vals: openArray[Value]): Value =
  result = vals[0]
  for i in 1 ..< vals.len:
    result = result + vals[i]

proc backward(self: Value) =
  var topo: seq[Value] = @[]
  var visited = initHashSet[pointer]()
  proc buildTopo(v: Value) =
    # In Nim, `HashSet` can't hash `ref` types by identity;
    #  cast to raw pointer (like Python's `id()`)
    let vPtr = cast[pointer](v)
    if vPtr notin visited:
      visited.incl(vPtr)
      for child in v.children:
        buildTopo(child)
      topo.add(v)
  buildTopo(self)
  self.grad = 1.0
  for i in countdown(topo.high, 0):
    let v = topo[i]
    for (child, localGrad) in zip(v.children, v.localGrads):
      child.grad += localGrad * v.grad

# --- Model parameters ---

# Initialize the parameters, to store the knowledge of the model
const
  nLayer = 1     # depth of the transformer neural network (number of layers)
  nEmbd = 16     # width of the network (embedding dimension)
  blockSize = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
  nHead = 4      # number of attention heads
  headDim = nEmbd div nHead # derived dimension of each head
# equivalent to Python's `matrix = lambda nout, nin, std=0.08: ...`
proc matrix(nout, nin: int, std: float64 = 0.08): seq[seq[Value]] =
  collect:
    for _ in 0 ..< nout:
      collect:
        for _ in 0 ..< nin:
          newValue(gauss(mu = 0.0, sigma = std))
var stateDict = {
  "wte": matrix(vocabSize, nEmbd),
  "wpe": matrix(blockSize, nEmbd),
  "lm_head": matrix(vocabSize, nEmbd)
}.toOrderedTable
for i in 0 ..< nLayer:
  stateDict[&"layer{i}.attn_wq"] = matrix(nEmbd, nEmbd)
  stateDict[&"layer{i}.attn_wk"] = matrix(nEmbd, nEmbd)
  stateDict[&"layer{i}.attn_wv"] = matrix(nEmbd, nEmbd)
  stateDict[&"layer{i}.attn_wo"] = matrix(nEmbd, nEmbd)
  stateDict[&"layer{i}.mlp_fc1"] = matrix(4 * nEmbd, nEmbd)
  stateDict[&"layer{i}.mlp_fc2"] = matrix(nEmbd, 4 * nEmbd)
let params = collect: # flatten params into a single seq[Value]
  for mat in stateDict.values:
    for row in mat:
      for p in row: p
echo &"num params: {params.len}"

# --- Model architecture ---

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

proc linear(x: seq[Value], w: seq[seq[Value]]): seq[Value] =
  collect(for wo in w: sum(collect(for (wi, xi) in zip(wo, x): wi * xi)))

proc softmax(logits: seq[Value]): seq[Value] =
  let maxVal = max(collect(for val in logits: val.data))
  let exps = collect(for val in logits: (val - maxVal).exp())
  let total = sum(exps)
  collect(for e in exps: e / total)

proc rmsnorm(x: seq[Value]): seq[Value] =
  let ms = sum(collect(for xi in x: xi * xi)) / x.len.float64
  let scale = (ms + 1e-5) ** (-0.5)
  collect(for xi in x: xi * scale)

proc gpt(tokenId, posId: int, keys, values: var seq[seq[seq[Value]]]): seq[Value] =
  let tokEmb = stateDict["wte"][tokenId] # token embedding
  let posEmb = stateDict["wpe"][posId] # position embedding
  var x = collect(for (t, p) in zip(tokEmb, posEmb): t + p) # joint token and position embedding
  x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

  for li in 0 ..< nLayer:
    # 1) Multi-head Attention block
    var xResidual = x
    x = rmsnorm(x)
    let q = linear(x, stateDict[&"layer{li}.attn_wq"])
    let k = linear(x, stateDict[&"layer{li}.attn_wk"])
    let v = linear(x, stateDict[&"layer{li}.attn_wv"])
    keys[li].add(k)
    values[li].add(v)
    var xAttn: seq[Value] = @[]
    for h in 0 ..< nHead:
      let hs = h * headDim
      let qH = q[hs ..< hs + headDim]
      let kH = collect(for ki in keys[li]: collect(for j in hs ..< hs + headDim: ki[j]))
      let vH = collect(for vi in values[li]: collect(for j in hs ..< hs + headDim: vi[j]))
      let attnLogits = collect:
        for t in 0 ..< kH.len:
          sum(collect(for j in 0 ..< headDim: qH[j] * kH[t][j])) / (headDim.float64 ** 0.5)
      let attnWeights = softmax(attnLogits)
      let headOut = collect:
        for j in 0 ..< headDim:
          sum(collect(for t in 0 ..< vH.len: attnWeights[t] * vH[t][j]))
      xAttn &= headOut
    x = linear(xAttn, stateDict[&"layer{li}.attn_wo"])
    x = collect(for (a, b) in zip(x, xResidual): a + b)
    # 2) MLP block
    xResidual = x
    x = rmsnorm(x)
    x = linear(x, stateDict[&"layer{li}.mlp_fc1"])
    x = collect(for xi in x: xi.relu())
    x = linear(x, stateDict[&"layer{li}.mlp_fc2"])
    x = collect(for (a, b) in zip(x, xResidual): a + b)

  let logits = linear(x, stateDict["lm_head"])
  return logits

# --- Adam optimizer ---

# Let there be Adam, the blessed optimizer and its buffers
const
  learningRate = 0.01
  beta1 = 0.85
  beta2 = 0.99
  epsAdam = 1e-8
# `newSeq` zero-initializes, matching Python's `[0.0] * len(params)`
var m = newSeq[float64](params.len) # first moment buffer
var v = newSeq[float64](params.len) # second moment buffer

# --- Training ---

# Repeat in sequence
const numSteps = 1000 # number of training steps
for step in 0 ..< numSteps:

  # Take single document, tokenize it, surround it with BOS special token on both sides
  let doc = docs[step mod docs.len]
  let tokens = @[BOS] & collect(for ch in doc.runes: uchars.find(ch)) & @[BOS]
  let n = min(blockSize, tokens.len - 1)

  # Forward the token sequence through the model, building up the computation graph all the way to the loss
  var keys = newSeq[seq[seq[Value]]](nLayer)
  var values = newSeq[seq[seq[Value]]](nLayer)
  var losses: seq[Value] = @[]
  for posId in 0 ..< n:
    let tokenId = tokens[posId]
    let targetId = tokens[posId + 1]
    let logits = gpt(tokenId, posId, keys, values)
    let probs = softmax(logits)
    let lossT = -probs[targetId].log()
    losses.add(lossT)
  let loss = (1.0 / n.float64) * sum(losses) # final average loss over the document sequence. May yours be low.

  # Backward the loss, calculating the gradients with respect to all model parameters
  loss.backward()

  # Adam optimizer update: update the model parameters based on the corresponding gradients
  let lrT = learningRate * (1.0 - step.float64 / numSteps.float64) # linear learning rate decay
  for i, p in params.pairs:
    m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad
    v[i] = beta2 * v[i] + (1.0 - beta2) * (p.grad ** 2.0)
    let mHat = m[i] / (1.0 - (beta1 ** (step + 1).float64))
    let vHat = v[i] / (1.0 - (beta2 ** (step + 1).float64))
    p.data -= lrT * mHat / ((vHat ** 0.5) + epsAdam)
    p.grad = 0.0

  stdout.write(&"\rstep {step+1:4} / {numSteps:4} | loss {loss.data:.4f}")
  stdout.flushFile()

# --- Inference ---

# Inference: may the model babble back to us
const temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
echo "\n--- inference (new, hallucinated names) ---"
for sampleIdx in 0 ..< 20:
  var keys = newSeq[seq[seq[Value]]](nLayer)
  var values = newSeq[seq[seq[Value]]](nLayer)
  var tokenId = BOS
  var sample = ""
  for posId in 0 ..< blockSize:
    let logits = gpt(tokenId, posId, keys, values)
    let probs = softmax(collect(for l in logits: l / temperature))
    # Python: `random.choices(range(vocab_size), weights=[p.data for p in probs])[0]`
    #   picks a random token from `range(vocab_size)` weighted by probability;
    #   returns a 1-element list, so `[0]` extracts the chosen token
    # Nim: `sample(population, cdf)` picks one element directly,
    #   but expects a cumulative distribution function (CDF), a running total of the weights,
    #   e.g. weights [0.1, 0.3, 0.6] → CDF [0.1, 0.4, 1.0]. `cumsummed` does this.
    tokenId = (0 ..< vocabSize).toSeq.sample(collect(for p in probs: p.data).cumsummed)
    if tokenId == BOS:
      break
    sample.add($uchars[tokenId])
  echo &"sample {sampleIdx+1:2}: {sample}"
