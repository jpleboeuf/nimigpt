import test_common

let args = parseToolArgs(tkGenFixture)
var runResult: RunResult

ensureProgExists(args.fixtureKind, args.progPath)
ensureTestdataDir()
ensureFixtureInTestdata(args.fixturePath)

case args.fixtureKind
  of fkPython:
    runResult = runProgram("python3", [args.progPath])
  of fkNim:
    runResult = runProgram(args.progPath)

if runResult.exitCode != 0:
  stderr.write(runResult.output)
  quit(runResult.exitCode)

writeFile(args.fixturePath, normalizeOutput(runResult.output))
echo "Wrote fixture to ", args.fixturePath
