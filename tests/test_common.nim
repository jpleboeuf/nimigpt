import std/os
import std/osproc
import std/streams
import std/strutils

type ToolKind* = enum
  tkGenFixture
  tkTestOutput

type FixtureKind* = enum
  fkPython
  fkNim

type ToolArgs* = object
  progPath*: string
  fixtureKind*: FixtureKind
  fixturePath*: string

const
  PythonFixtureFile* = "microgpt_py_output_42.txt"
  NimFixtureFile* = "nimigpt_nim_output_42.txt"

type RunResult* = object
  exitCode*: int
  output*: string

proc normalizeOutput*(text: string): string =
  text.replace("\r\n", "\n").replace("\r", "\n")

proc runProgram*(exePath: string, args: openArray[string] = []): RunResult =
  var process = startProcess(exePath, args = @args, options = {poStdErrToStdOut, poUsePath})
  defer: close(process)
  result.output = process.outputStream.readAll()
  result.exitCode = waitForExit(process)

proc getPythonDefaultScriptPath*(): string =
  getCurrentDir() / "microgpt.py"

proc getNimExePath*(): string =
  getCurrentDir() / ("nimigpt" & ExeExt)

proc defaultProgPath*(toolKind: ToolKind, fixtureKind: FixtureKind): string =
  if toolKind == tkGenFixture and fixtureKind == fkPython:
    getPythonDefaultScriptPath()
  else:
    getNimExePath()

proc progPathLabel*(toolKind: ToolKind, fixtureKind: FixtureKind): string =
  if toolKind == tkGenFixture and fixtureKind == fkPython:
    "python_script_path"
  else:
    "nim_exe_path"

proc getTestdataDir*(): string =
  getCurrentDir() / "tests" / "testdata"

proc ensureTestdataDir*() =
  createDir(getTestdataDir())

proc defaultFixturePath*(kind: FixtureKind): string =
  case kind
    of fkPython:
      getTestdataDir() / PythonFixtureFile
    of fkNim:
      getTestdataDir() / NimFixtureFile

proc getFixturePath*(kind: FixtureKind, argIndex: int): string =
  if paramCount() >= argIndex:
    paramStr(argIndex)
  else:
    defaultFixturePath(kind)

proc toolName*(kind: ToolKind): string =
  case kind
    of tkGenFixture:
      "tests/gen_fixture"
    of tkTestOutput:
      "tests/test_output"

proc writeModeUsage*(toolKind: ToolKind) =
  let name = toolName(toolKind)
  stderr.writeLine("Usage:")
  stderr.writeLine("  " & name & " --python [" & progPathLabel(toolKind, fkPython) & "] [fixture_path]")
  stderr.writeLine("  " & name & " --nim [" & progPathLabel(toolKind, fkNim) & "] [fixture_path]")

proc parseFixtureKind(toolKind: ToolKind): FixtureKind =
  if paramCount() == 0:
    writeModeUsage(toolKind)
    quit(QuitFailure)

  case paramStr(1)
    of "--python":
      fkPython
    of "--nim":
      fkNim
    else:
      writeModeUsage(toolKind)
      quit(QuitFailure)

proc parseToolArgs*(toolKind: ToolKind): ToolArgs =
  result.fixtureKind = parseFixtureKind(toolKind)
  result.progPath =
    if paramCount() >= 2: paramStr(2) else: defaultProgPath(toolKind, result.fixtureKind)
  result.fixturePath = getFixturePath(result.fixtureKind, 3)

proc ensureProgExists*(kind: FixtureKind, progPath: string) =
  if fileExists(progPath):
    return
  case kind
    of fkPython:
      stderr.writeLine("Python microgpt.py not found: " & progPath)
    of fkNim:
      stderr.writeLine("Nim executable not found: " & progPath)
  quit(QuitFailure)

proc ensureFixtureInTestdata*(fixturePath: string) =
  let normalizedTestdataDir = absolutePath(getTestdataDir()).normalizedPath()
  let normalizedFixturePath = absolutePath(fixturePath).normalizedPath()
  if parentDir(normalizedFixturePath) != normalizedTestdataDir:
    stderr.writeLine("Fixture path must be inside " & getTestdataDir() & ": " & fixturePath)
    quit(QuitFailure)

proc ensureFixtureExists*(fixturePath: string) =
  if fileExists(fixturePath):
    return
  stderr.writeLine("Fixture not found: " & fixturePath)
  quit(QuitFailure)
