import std/os
import std/strutils
import std/syncio
import test_common

proc mismatchLine(expected, actual: string): int =
  let expectedLines = expected.splitLines()
  let actualLines = actual.splitLines()
  let lineCount = min(expectedLines.len, actualLines.len)
  for i in 0 ..< lineCount:
    if expectedLines[i] != actualLines[i]:
      return i + 1
  if expectedLines.len != actualLines.len:
    return lineCount + 1
  return 0

let exePath = getExePath()
let fixturePath = getFixturePath()

if not fileExists(fixturePath):
  stderr.writeLine("Fixture not found: " & fixturePath)
  quit(QuitFailure)

let runResult = runProgram(exePath)
if runResult.exitCode != 0:
  stderr.write(runResult.output)
  quit(runResult.exitCode)

let expected = readFile(fixturePath)
let actual = normalizeOutput(runResult.output)
if expected != actual:
  let lineNo = mismatchLine(expected, actual)
  stderr.writeLine("Output mismatch against " & fixturePath)
  if lineNo > 0:
    let expectedLines = expected.splitLines()
    let actualLines = actual.splitLines()
    stderr.writeLine("First differing line: " & $lineNo)
    if lineNo <= expectedLines.len:
      stderr.writeLine("Expected: " & expectedLines[lineNo - 1])
    if lineNo <= actualLines.len:
      stderr.writeLine("Actual:   " & actualLines[lineNo - 1])
  quit(QuitFailure)

echo "Output matches ", fixturePath
