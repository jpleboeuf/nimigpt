import std/os
import test_common

let exePath = getExePath()
let fixturePath = getFixturePath()

let runResult = runProgram(exePath)
if runResult.exitCode != 0:
  stderr.write(runResult.output)
  quit(runResult.exitCode)

createDir(parentDir(fixturePath))
writeFile(fixturePath, normalizeOutput(runResult.output))
echo "Wrote fixture to ", fixturePath
