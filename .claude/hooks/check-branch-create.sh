#!/bin/bash
# PreToolUse hook for Bash commands.
# Blocks branch creation when not on main/master.
INPUT=$(cat)

if echo "$INPUT" | grep -qE 'git (checkout -b|switch -c|branch )'; then
  CURRENT=$(git branch --show-current 2>/dev/null)
  if [ "$CURRENT" != "main" ] && [ "$CURRENT" != "master" ]; then
    # stderr, not stdout: Claude Code feeds a blocking hook's stderr back to
    # the agent on exit 2 and discards stdout, so a message printed to stdout
    # is lost and the agent just sees an unexplained block.
    echo "BLOCK: You're on '$CURRENT', not main. Switch to main before creating a new branch." >&2
    exit 2
  fi
fi

exit 0
