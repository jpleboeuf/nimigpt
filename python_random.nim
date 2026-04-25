## Python-compatible random helper for the faithful `microgpt.py` port.
##
## This module implements the subset of Python's `random` API
##  used by Karpathy's `microgpt.py`:
## - `random.seed`
## - `random.shuffle`
## - `random.gauss`
## - `random.choices(..., weights=...)[0]`
##
## The core generator is MT19937 (Mersenne Twister),
##  following CPython's `_randommodule.c`
##   state layout, seeding, twist/temper logic, and 53-bit float generation.
## Source:
##  https://github.com/python/cpython/blob/main/Modules/_randommodule.c
##
## Higher-level helpers such as `gauss` and `shuffle`
##  follow the corresponding behavior from Python's `Lib/random.py`.
## Source:
##  https://github.com/python/cpython/blob/main/Lib/random.py
##
## This is a compatibility layer,
##  not a full reimplementation of Python's `random` module.

import std/math

const
  # Standard MT19937 parameters used by CPython's `_random` module.
  N = 624
  M = 397
  MatrixA = 0x9908b0df'u32
  UpperMask = 0x80000000'u32
  LowerMask = 0x7fffffff'u32
  # Python's `random()` builds a 53-bit float from two 32-bit MT outputs.
  RecipBpf = 1.0 / 9007199254740992.0
  TwoPi = 2.0 * PI

type PythonRandom* = object
  ## CPython-compatible RNG state backed by MT19937
  ##  plus Python's cached spare sample for `gauss()`.
  index: int
  state: array[N, uint32]
  gaussNext: float64
  hasGaussNext: bool

proc initGenrand(rng: var PythonRandom, seed: uint32) =
  ## Initialize the MT19937 state from a single 32-bit seed.
  rng.state[0] = seed
  for i in 1 ..< N:
    rng.state[i] = 1812433253'u32 * (rng.state[i - 1] xor (rng.state[i - 1] shr 30)) + uint32(i)
  rng.index = N
  rng.gaussNext = 0.0
  rng.hasGaussNext = false

proc initByArray(rng: var PythonRandom, initKey: openArray[uint32]) =
  ## Initialize the MT19937 state from an array of 32-bit words.
  rng.initGenrand(19650218'u32)
  var i = 1
  var j = 0
  var k = max(N, initKey.len)
  while k > 0:
    rng.state[i] = (rng.state[i] xor ((rng.state[i - 1] xor (rng.state[i - 1] shr 30)) * 1664525'u32)) +
      initKey[j] + uint32(j)
    inc i
    inc j
    if i >= N:
      rng.state[0] = rng.state[N - 1]
      i = 1
    if j >= initKey.len:
      j = 0
    dec k
  k = N - 1
  while k > 0:
    rng.state[i] = (rng.state[i] xor ((rng.state[i - 1] xor (rng.state[i - 1] shr 30)) * 1566083941'u32)) -
      uint32(i)
    inc i
    if i >= N:
      rng.state[0] = rng.state[N - 1]
      i = 1
    dec k
  rng.state[0] = 0x80000000'u32

proc seed*(rng: var PythonRandom, value: int) =
  ## Seed the generator from an integer value.
  ##
  ## This follows CPython's integer-seed path
  ##  by splitting the integer into 32-bit words and feeding them to `init_by_array`.
  var n =
    if value < 0:
      uint64(-(value + 1)) + 1'u64
    else:
      uint64(value)
  var key: seq[uint32] = @[]
  while n != 0:
    key.add(uint32(n and 0xffffffff'u64))
    n = n shr 32
  if key.len == 0:
    key.add(0'u32)
  rng.initByArray(key)

proc genrandInt32(rng: var PythonRandom): uint32 =
  ## Return the next tempered 32-bit MT19937 output.
  if rng.index >= N:
    const mag01 = [0'u32, MatrixA]
    for kk in 0 ..< N - M:
      let y = (rng.state[kk] and UpperMask) or (rng.state[kk + 1] and LowerMask)
      rng.state[kk] = rng.state[kk + M] xor (y shr 1) xor mag01[int(y and 0x1'u32)]
    for kk in N - M ..< N - 1:
      let y = (rng.state[kk] and UpperMask) or (rng.state[kk + 1] and LowerMask)
      rng.state[kk] = rng.state[kk + (M - N)] xor (y shr 1) xor mag01[int(y and 0x1'u32)]
    let y = (rng.state[N - 1] and UpperMask) or (rng.state[0] and LowerMask)
    rng.state[N - 1] = rng.state[M - 1] xor (y shr 1) xor mag01[int(y and 0x1'u32)]
    rng.index = 0

  result = rng.state[rng.index]
  inc rng.index
  result = result xor (result shr 11)
  result = result xor ((result shl 7) and 0x9d2c5680'u32)
  result = result xor ((result shl 15) and 0xefc60000'u32)
  result = result xor (result shr 18)

proc random*(rng: var PythonRandom): float64 =
  ## Return the next random floating-point number in the range `[0.0, 1.0)`.
  ##
  ## This uses the same 53-bit float construction as Python's `random.Random.random`.
  let a = uint64(rng.genrandInt32() shr 5)
  let b = uint64(rng.genrandInt32() shr 6)
  ((a shl 26) + b).float64 * RecipBpf

proc getrandbits(rng: var PythonRandom, k: int): uint64 =
  ## Return an integer with `k` random bits.
  if k <= 0:
    return 0
  var remaining = k
  var value = 0'u64
  var shift = 0
  while remaining >= 32:
    value = value or (uint64(rng.genrandInt32()) shl shift)
    shift += 32
    remaining -= 32
  if remaining > 0:
    value = value or (uint64(rng.genrandInt32() shr (32 - remaining)) shl shift)
  value

proc bitLength(n: int): int =
  ## Return the number of bits needed to represent `n`.
  var value = n
  while value > 0:
    inc result
    value = value shr 1

proc randBelow(rng: var PythonRandom, n: int): int =
  ## Return a random integer in the range `[0, n)`.
  if n <= 0:
    return 0
  let bits = bitLength(n)
  var r = int(rng.getrandbits(bits))
  while r >= n:
    r = int(rng.getrandbits(bits))
  r

proc shuffle*[T](rng: var PythonRandom, xs: var seq[T]) =
  ## Shuffle the sequence in place.
  ##
  ## This follows Python's `random.Random.shuffle`,
  ##  using Fisher-Yates with `_randbelow`-style index selection.
  for i in countdown(xs.high, 1):
    let j = rng.randBelow(i + 1)
    swap(xs[i], xs[j])

proc gauss*(rng: var PythonRandom, mu, sigma: float64): float64 =
  ## Gaussian distribution.
  ##
  ## This follows Python's `random.Random.gauss`,
  ##  including the cached spare sample stored between calls.
  var z = rng.gaussNext
  let hasNext = rng.hasGaussNext
  rng.gaussNext = 0.0
  rng.hasGaussNext = false
  if not hasNext:
    let x2pi = rng.random() * TwoPi
    let g2rad = sqrt(-2.0 * ln(1.0 - rng.random()))
    z = cos(x2pi) * g2rad
    rng.gaussNext = sin(x2pi) * g2rad
    rng.hasGaussNext = true
  mu + z * sigma

proc weightedChoice*(rng: var PythonRandom, weights: openArray[float64]): int =
  ## Return one weighted draw compatible with `random.choices(...)[0]`.
  ##
  ## This intentionally implements only the weighted single-sample path
  ##  needed by `microgpt.py`, not the full Python `choices` API.
  if weights.len == 0:
    raise newException(ValueError, "empty weights")
  var cumulative = newSeq[float64](weights.len)
  var total = 0.0
  for i, weight in weights:
    total += weight
    cumulative[i] = total
  let kind = classify(total)
  if total <= 0.0 or kind == fcInf or kind == fcNegInf or kind == fcNan:
    raise newException(ValueError, "invalid total weight")
  let target = rng.random() * total
  var lo = 0
  var hi = weights.high
  while lo < hi:
    let mid = (lo + hi) shr 1
    if target < cumulative[mid]:
      hi = mid
    else:
      lo = mid + 1
  lo
