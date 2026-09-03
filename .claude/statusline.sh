#!/bin/bash
# Shared statusline for Claude Code.
# Shows: directory | session | branch ahead/behind | branches (stale) | context %
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd')

dir="$cwd"

# Git info
branch=""
branch_detail=""
branches_info=""
if git -C "$cwd" rev-parse --git-dir > /dev/null 2>&1; then
    branch=$(git -C "$cwd" symbolic-ref --short HEAD 2>/dev/null || git -C "$cwd" rev-parse --short HEAD 2>/dev/null)

    # Determine main branch name
    main_branch="main"
    git -C "$cwd" rev-parse --verify main &>/dev/null || main_branch="master"

    # Ahead/behind
    if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
        # On main: show behind origin
        if git -C "$cwd" rev-parse --verify "origin/$branch" &>/dev/null; then
            ahead=$(git -C "$cwd" rev-list --count "origin/$branch..$branch" 2>/dev/null || echo 0)
            behind=$(git -C "$cwd" rev-list --count "$branch..origin/$branch" 2>/dev/null || echo 0)
            ab=""
            [ "$ahead" -gt 0 ] 2>/dev/null && ab="${ahead}↑"
            [ "$behind" -gt 0 ] 2>/dev/null && ab="${ab:+$ab }${behind}↓"
            if [ -n "$ab" ]; then
                branch_detail="$branch $ab"
            else
                branch_detail="$branch"
            fi
        else
            branch_detail="$branch"
        fi
    else
        # On feature branch: show ahead/behind main
        if git -C "$cwd" rev-parse --verify "$main_branch" &>/dev/null; then
            ahead=$(git -C "$cwd" rev-list --count "$main_branch..$branch" 2>/dev/null || echo 0)
            behind=$(git -C "$cwd" rev-list --count "$branch..$main_branch" 2>/dev/null || echo 0)
            ab=""
            [ "$ahead" -gt 0 ] 2>/dev/null && ab="${ahead}↑"
            [ "$behind" -gt 0 ] 2>/dev/null && ab="${ab:+$ab }${behind}↓"
            if [ -n "$ab" ]; then
                branch_detail="$branch $ab"
            else
                branch_detail="$branch"
            fi
        else
            branch_detail="$branch"
        fi
    fi

    # Dirty working tree (tracked files only)
    if git -C "$cwd" diff --quiet HEAD 2>/dev/null && git -C "$cwd" diff --cached --quiet HEAD 2>/dev/null; then
        :
    else
        branch_detail="${branch_detail} *"
    fi

    # Other branches: count total and stale
    other_branches=$(git -C "$cwd" branch 2>/dev/null \
        | grep -v "^\*" \
        | grep -v "HEAD" \
        | grep -vE "^\s*(main|master)\s*$" \
        | sed 's/^[[:space:]]*//')

    total=0
    stale=0
    if [ -n "$other_branches" ]; then
        now=$(date +%s)
        one_week_ago=$((now - 7 * 86400))

        while IFS= read -r b; do
            [ -z "$b" ] && continue
            total=$((total + 1))

            is_stale=0

            # Check 1: last commit age > 1 week
            last_commit_ts=$(git -C "$cwd" log -1 --format=%ct "$b" 2>/dev/null || echo "$now")
            if [ "$last_commit_ts" -lt "$one_week_ago" ] 2>/dev/null; then
                is_stale=1
            fi

            # Check 2: merge-base is 25+ commits behind main HEAD
            if [ "$is_stale" -eq 0 ] && git -C "$cwd" rev-parse --verify "$main_branch" &>/dev/null; then
                merge_base=$(git -C "$cwd" merge-base "$b" "$main_branch" 2>/dev/null)
                if [ -n "$merge_base" ]; then
                    distance=$(git -C "$cwd" rev-list --count "$merge_base..$main_branch" 2>/dev/null || echo 0)
                    [ "$distance" -ge 25 ] 2>/dev/null && is_stale=1
                fi
            fi

            [ "$is_stale" -eq 1 ] && stale=$((stale + 1))
        done <<< "$other_branches"
    fi

    if [ "$total" -gt 0 ]; then
        if [ "$stale" -gt 0 ]; then
            branches_info="$total branches ($stale stale)"
        else
            branches_info="$total branches"
        fi
    else
        branches_info="0 branches"
    fi
fi

# Review status
review_status=""
if git -C "$cwd" rev-parse --git-dir > /dev/null 2>&1; then
    current_head=$(git -C "$cwd" rev-parse HEAD 2>/dev/null)
    stamp="$(git -C "$cwd" rev-parse --absolute-git-dir 2>/dev/null)/.last-review"
    if [ -f "$stamp" ]; then
        last_reviewed=$(tr -d '[:space:]' < "$stamp")
        if [ "$last_reviewed" = "$current_head" ]; then
            review_status="ok"
        else
            commits_since=$(git -C "$cwd" rev-list --count "$last_reviewed..$current_head" 2>/dev/null || echo "?")
            review_status="${commits_since} behind"
        fi
    else
        review_status="none"
    fi
fi

# Session name
session_name=$(echo "$input" | jq -r '.session_name // empty')

# Context remaining
remaining=$(echo "$input" | jq -r '.context_window.remaining_percentage // empty')

# Build output
parts=()
# dim label helper: \033[2m is dim
label='\033[2m'
reset='\033[00m'

parts+=("$(printf "${label}dir:${reset}\033[01;34m%s\033[00m" "$dir")")

if [ -n "$session_name" ]; then
    parts+=("$(printf "${label}session:${reset}\033[00;36m%s\033[00m" "$session_name")")
fi

if [ -n "$branch_detail" ]; then
    # Color based on behind count
    behind_count=0
    if echo "$branch_detail" | grep -qE '[0-9]+↓'; then
        behind_count=$(echo "$branch_detail" | grep -oE '[0-9]+↓' | grep -oE '[0-9]+')
    fi
    if [ "$behind_count" -ge 10 ] 2>/dev/null; then
        parts+=("$(printf "${label}branch:${reset}\033[01;31m%s\033[00m" "$branch_detail")")
    elif [ "$behind_count" -ge 1 ] 2>/dev/null; then
        parts+=("$(printf "${label}branch:${reset}\033[01;33m%s\033[00m" "$branch_detail")")
    else
        parts+=("$(printf "${label}branch:${reset}\033[00;32m%s\033[00m" "$branch_detail")")
    fi
fi

if [ -n "$branches_info" ]; then
    if [ "$stale" -gt 0 ] 2>/dev/null; then
        parts+=("$(printf '\033[01;31m%s\033[00m' "$branches_info")")
    elif [ "$total" -gt 0 ] 2>/dev/null; then
        parts+=("$(printf '\033[01;33m%s\033[00m' "$branches_info")")
    else
        parts+=("$(printf '\033[00;32m%s\033[00m' "$branches_info")")
    fi
fi

if [ -n "$review_status" ]; then
    case "$review_status" in
        ok)
            parts+=("$(printf "${label}review:${reset}\033[00;32m%s\033[00m" "$review_status")")
            ;;
        none)
            parts+=("$(printf "${label}review:${reset}\033[01;31m%s\033[00m" "$review_status")")
            ;;
        *)
            parts+=("$(printf "${label}review:${reset}\033[01;33m%s\033[00m" "$review_status")")
            ;;
    esac
fi

if [ -n "$remaining" ]; then
    remaining_int=$(printf '%.0f' "$remaining")
    if [ "$remaining_int" -le 20 ]; then
        color='\033[01;31m'
    elif [ "$remaining_int" -le 40 ]; then
        color='\033[01;33m'
    else
        color='\033[00;32m'
    fi
    parts+=("$(printf "${label}ctx:${reset}${color}%s%%\033[00m" "$remaining_int")")
fi

# Join with separator
result=""
for part in "${parts[@]}"; do
    if [ -z "$result" ]; then
        result="$part"
    else
        result="$result $(printf '\033[00;37m|\033[00m') $part"
    fi
done

printf "%s" "$result"
