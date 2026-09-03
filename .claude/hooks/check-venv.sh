#!/bin/bash
# PreToolUse hook for Bash commands.
# If a .venv exists in the project but VIRTUAL_ENV is not set,
# blocks bare python/pip commands and tells Claude to use the venv path.
#
# If there is no venv, one is demanded only from projects that actually declare
# Python dependencies or packaging (see declares_python_deps). A project whose
# only Python metadata is linter config runs stdlib-only utility scripts with
# the system interpreter, and nothing is gained by making it build a venv.
#
# This hook operates on the actual shell command (tool_input.command extracted
# from the Claude Code hook JSON), NOT the raw JSON payload. Matching the raw
# JSON is wrong on two counts: (1) the command is wrapped in quotes, so a
# legitimate ".venv/bin/python ..." is preceded by a '"' rather than
# whitespace/start, defeating the allow-check below; (2) the word
# "python"/"pip" can appear as a plain argument (e.g. `find python -type f`).
INPUT=$(cat)

# Does this project declare Python dependencies or packaging?
#
# The presence of pyproject.toml alone is not evidence, which is what this
# predicate replaces. devkit's own setup.sh merges [tool.ruff] / [tool.mypy] /
# [tool.pytest] into every consumer that opts into Python tooling — C++ and
# cpp-python projects included — and config/pyproject.toml.template holds
# nothing but those tables. So devkit was handing projects a pyproject.toml and
# then reading that same file back as proof they needed a venv, blocking
# stdlib-only scripts (and, via resolve-venv.sh, their commits and pushes).
#
# Look instead for what a venv is actually for: third-party dependencies to
# install, or a package to build.
declares_python_deps() {
  [ -f "setup.py" ] && return 0
  [ -f "setup.cfg" ] && return 0
  [ -f "Pipfile" ] && return 0
  [ -f "poetry.lock" ] && return 0
  [ -f "uv.lock" ] && return 0

  # requirements.txt, requirements-dev.txt, requirements_test.txt, ...
  for req in requirements*.txt; do
    [ -e "$req" ] && return 0
  done

  # PEP 621 [project], PEP 517 [build-system], PEP 735 [dependency-groups], a
  # build backend's own table, or a bare `dependencies = [...]`.
  if [ -f "pyproject.toml" ] && grep -qE \
    '^[[:space:]]*(\[(project|build-system|dependency-groups)\]|\[tool\.(poetry|pdm|hatch|flit|uv|pixi|setuptools)|dependencies[[:space:]]*=)' \
    pyproject.toml; then
    return 0
  fi

  return 1
}

# Extract the shell command from the hook payload. Prefer jq; fall back to
# python3. If we can't parse it out, fail OPEN (exit 0) — never block on a
# payload we can't reliably read, since scanning the raw JSON misfires.
CMD=""
if command -v jq >/dev/null 2>&1; then
  CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
fi
if [ -z "$CMD" ] && command -v python3 >/dev/null 2>&1; then
  CMD=$(printf '%s' "$INPUT" | python3 -c \
    'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' \
    2>/dev/null)
fi
[ -z "$CMD" ] && exit 0

# Always allow venv-creation/bootstrap commands. Otherwise the "no virtualenv
# found" branch below would block the very command it tells the user to run.
if echo "$CMD" | grep -qE '(python3?([.][0-9]+)?[[:space:]]+-m[[:space:]]+venv|(^|[[:space:]])(uv[[:space:]]+venv|virtualenv)([[:space:]]|$))'; then
  exit 0
fi

# Strip quoted string literals so tool names inside messages/arguments (e.g. a
# commit message "fix; mypy clean") are not mistaken for real invocations.
SCAN=$(printf '%s' "$CMD" | sed -E "s/'[^']*'//g; s/\"[^\"]*\"//g")

# Detect a Python tool only when it sits at *command position*: the start of the
# command or right after a shell separator (; & | ( { && ||), optionally behind
# leading VAR=value assignments. This matches `python ...`, `cd x && pytest`,
# `FOO=1 mypy ...` but not `find python` or `cat foo_pytest.txt`, and not a
# path-qualified `.venv/bin/python` (which is already correct usage).
TOOLS='python3?([.][0-9]+)?|pip3?|pytest|mypy|dmypy|ruff'
CMDPOS='(^|[;&|({]|&&|\|\|)[[:space:]]*([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*'
if echo "$SCAN" | grep -qE "${CMDPOS}(${TOOLS})([[:space:]]|$)"; then
  # Already in a venv — all good
  [ -n "$VIRTUAL_ENV" ] && exit 0

  # Look for a venv in the project
  VENV_DIR=""
  [ -d ".venv" ] && VENV_DIR=".venv"
  [ -d "venv" ] && VENV_DIR="venv"

  if [ -n "$VENV_DIR" ]; then
    # Venv exists but not activated — allow only if the command already invokes
    # the venv's tools directly (.venv/bin/...) or targets it (uv --python ...).
    if ! echo "$CMD" | grep -qE "(^|[[:space:]])\.?/?$VENV_DIR/bin/"; then
      # Claude Code feeds a blocking hook's *stderr* back to the agent on
      # exit 2 and discards its stdout, so guidance printed to stdout never
      # arrives — the agent sees only "No stderr output" and has to open this
      # file to find out what it did wrong.
      {
        echo "BLOCK: A virtualenv exists at $VENV_DIR/ but is not activated."
        echo "Use the venv's Python directly instead of bare commands:"
        echo "  $VENV_DIR/bin/python instead of python"
        echo "  $VENV_DIR/bin/pip instead of pip"
        echo "  $VENV_DIR/bin/pytest instead of pytest"
      } >&2
      exit 2
    fi
  elif declares_python_deps; then
    # Real Python project with no venv — fix that before anything gets
    # installed into the system interpreter.
    {
      echo "BLOCK: No virtualenv found. Create one before running Python commands:"
      echo "  python3 -m venv .venv"
      echo "  uv venv"
    } >&2
    exit 2
  fi
  # Otherwise: no venv, and no declared dependencies to install into one. A
  # stdlib-only script in a project that merely carries linter config — run it.
fi

exit 0
