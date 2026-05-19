#!/usr/bin/env bash
# Posts a PR comment and fails the check when a PR touches migrated EPP/BBR paths.
set -euo pipefail

if [[ -z "${GITHUB_REPOSITORY:-}" || -z "${PR_NUMBER:-}" ]]; then
  echo "GITHUB_REPOSITORY and PR_NUMBER must be set"
  exit 1
fi

CONFIG=".github/migrated-paths.yaml"
MARKER=$(grep '^comment_marker:' "$CONFIG" | sed 's/^comment_marker:[[:space:]]*//;s/^"//;s/"$//')
MIGRATION_ISSUE=$(grep '^migration_issue:' "$CONFIG" | sed 's/^migration_issue:[[:space:]]*//')
EPP_REPO=$(awk '/^repositories:/{r=1;next} r && /^  epp:/{print $2; exit}' "$CONFIG")
BBR_REPO=$(awk '/^repositories:/{r=1;next} r && /^  bbr:/{print $2; exit}' "$CONFIG")

PREFIXES=()
while read -r prefix; do
  PREFIXES+=("$prefix")
done < <(awk '/^prefixes:/{p=1;next} p && /^  - /{print $2; next} p && /^[a-z]/{exit}' "$CONFIG")

if [[ ${#PREFIXES[@]} -eq 0 ]]; then
  echo "No prefixes found in ${CONFIG}"
  exit 1
fi

is_migrated_path() {
  local file=$1 prefix
  for prefix in "${PREFIXES[@]}"; do
    if [[ "$file" == "$prefix"* ]]; then
      return 0
    fi
  done
  return 1
}

# Collect PR files that touch migrated paths
matched_files=()
hit_epp=false
hit_bbr=false
page=1

while true; do
  response=$(gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/files?per_page=100&page=${page}")
  count=$(echo "$response" | jq 'length')
  if [[ "$count" -eq 0 ]]; then
    break
  fi

  while IFS= read -r file; do
    for prefix in "${PREFIXES[@]}"; do
      if [[ "$file" != "$prefix"* ]]; then
        continue
      fi
      matched_files+=("$file")
      if [[ "$prefix" == cmd/epp/ || "$prefix" == pkg/epp/ ]]; then
        hit_epp=true
      fi
      if [[ "$prefix" == cmd/bbr/ || "$prefix" == pkg/bbr/ ]]; then
        hit_bbr=true
      fi
      break
    done
  done < <(echo "$response" | jq -r '.[].filename')

  [[ "$count" -lt 100 ]] && break
  page=$((page + 1))
done

if [[ ${#matched_files[@]} -eq 0 ]]; then
  echo "No migrated EPP/BBR paths changed."
  exit 0
fi

destinations=""
if [[ "$hit_epp" == true ]]; then
  destinations="${destinations}- [llm-d-inference-scheduler](${EPP_REPO}) (EPP)"$'\n'
fi
if [[ "$hit_bbr" == true ]]; then
  destinations="${destinations}- [llm-d-inference-payload-processor](${BBR_REPO}) (BBR)"$'\n'
fi

path_names=$(printf '\`%s\` ' "${PREFIXES[@]}")
file_list=$(printf '%s\n' "${matched_files[@]}" | sort -u | head -20 | sed 's/^/- `/;s/$/`/')
file_count=$(printf '%s\n' "${matched_files[@]}" | sort -u | wc -l | tr -d ' ')
if [[ "$file_count" -gt 20 ]]; then
  file_list="${file_list}"$'\n'"- ... and $((file_count - 20)) more"
fi

comment="${MARKER}

This pull request changes code that has moved to llm-d. We are not cutting new releases from these paths in this repository.

Open your change in:

${destinations}
Background: ${MIGRATION_ISSUE} · [README](https://github.com/${GITHUB_REPOSITORY}/blob/main/README.md)

Changed files:
${file_list}

Maintainers: this check fails until the change is moved to llm-d or the PR is closed. Please review for valid exceptions before closing !!"

comments=$(gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments?per_page=100" --jq '.[].body')
if echo "$comments" | grep -qF "$MARKER"; then
  echo "Migration notice already on PR #${PR_NUMBER}."
else
  if ! gh pr comment "$PR_NUMBER" --body "$comment"; then
    echo "warning: could not post comment on PR #${PR_NUMBER}"
  else
    echo "Posted migration notice on PR #${PR_NUMBER}."
  fi
fi

path_list=$(IFS=, ; echo "${PREFIXES[*]}")
echo "::error::PR modifies migrated paths (${path_list}). Open the change in llm-d instead."
exit 1
