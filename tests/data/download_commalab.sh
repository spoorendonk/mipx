#!/usr/bin/env bash
# Fetch the CommaLAB grid and planar instances of Larsson and Yuan (2004).
# https://commalab.di.unipi.it/datasets/mmcf/#Plnr
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="${1:-$script_dir/commalab}"
base=https://commalab.di.unipi.it/files/Data/MMCF
mkdir -p "$dest"

download_archive() (
    checksum=$1
    archive=$2
    target="$dest/$archive"
    if [[ -f "$target" ]] &&
        printf '%s  %s\n' "$checksum" "$target" | sha256sum --check --status; then
        return
    fi
    download=$(mktemp "$dest/.$archive.XXXXXX")
    trap 'rm -f -- "$download"' EXIT
    curl --fail --location --silent --show-error --retry 2 --max-time 60 \
        "$base/$archive" -o "$download"
    printf '%s  %s\n' "$checksum" "$download" | sha256sum --check
    mv -f -- "$download" "$target"
)

while read -r checksum archive; do
    download_archive "$checksum" "$archive"
done < "$script_dir/commalab_archives.sha256"
