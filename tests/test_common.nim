import std/os
import std/osproc
import std/streams
import std/strutils

proc normalizeOutput*(text: string): string =
  text.replace("\r", "\n")

proc runProgram*(exePath: string): tuple[exitCode: int, output: string] =
  var process = startProcess(exePath, options = {poStdErrToStdOut})
  defer: close(process)
  result.output = process.outputStream.readAll()
  result.exitCode = waitForExit(process)

proc getExePath*(argIndex: int = 1): string =
  if paramCount() >= argIndex:
    paramStr(argIndex)
  else:
    getCurrentDir() / ("nimigpt" & ExeExt)

proc getFixturePath*(argIndex: int = 2): string =
  if paramCount() >= argIndex:
    paramStr(argIndex)
  else:
    getCurrentDir() / "tests" / "testdata" / "nimigpt_output_42.txt"
