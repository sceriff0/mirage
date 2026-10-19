#!/usr/bin/env bash
# Download and unpack ANHIR (medium scale) into <dest>, the folder `regbench prepare
# anhir --source <dest>` reads:
#
#   <dest>/dataset_medium.csv     the cover table
#   <dest>/images/<set>/...       the image archive, rejoined and unzipped
#   <dest>/landmarks/<set>/...    the landmark archive, unzipped
#
#   ANHIR_URLS=<file> benchmarks/regbench/download/anhir.sh <dest>
#
# ANHIR has no stable public URL: the files are behind the challenge's registration
# (https://anhir.grand-challenge.org/Data/ [FULL: the saved copy of
# https://anhir.grand-challenge.org/Data/ fetched 2026-10-10]). <file> holds one
# `url<TAB>name[<TAB>md5]` line per file, the names being the download's own:
# dataset_medium.csv, dataset_medium.z01 .. dataset_medium.z05, dataset_medium.zip and
# landmarks.zip. Without ANHIR_URLS the script refuses before any download.
#
# The image archive is split; it is rejoined with Info-ZIP's `zip -s 0` and unzipped
# into images/, the landmark archive into landmarks/, as
# 5a46f778:benchmarks/anhir/prepare.py:1-25, 75-92 did. A download is resumable: run
# the script again. ANHIR is CC BY-NC-SA: never commit what this writes.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=benchmarks/regbench/download/common.sh
. "$here/common.sh"

dest=${1:?usage: ANHIR_URLS=<file> anhir.sh <dest>}
if [ -z "${ANHIR_URLS:-}" ]; then
	echo "anhir.sh: set ANHIR_URLS to a file of 'url<TAB>name[<TAB>md5]' lines" >&2
	echo "  (the links from https://anhir.grand-challenge.org/Data/ after registering)" >&2
	exit 2
fi
urls=$(rb_urls "$ANHIR_URLS")
for want in dataset_medium.csv dataset_medium.zip landmarks.zip; do
	if ! printf '%s\n' "$urls" | awk -F'\t' -v w="$want" '$2 == w {f = 1} END {exit !f}'; then
		echo "anhir.sh: $ANHIR_URLS names no '$want'" >&2
		exit 2
	fi
done
rb_need curl zip unzip
mkdir -p "$dest"

while IFS=$'\t' read -r url name md5; do
	rb_fetch "$url" "$dest/$name" "${md5:-}"
done <<<"$urls"

joined="$dest/dataset_medium_joined.zip"
if [ ! -e "$joined" ]; then
	if ! compgen -G "$dest/dataset_medium.z[0-9][0-9]" >/dev/null; then
		echo "anhir.sh: no split parts dataset_medium.z01.. in $dest" >&2
		exit 1
	fi
	zip -s 0 "$dest/dataset_medium.zip" --out "$joined"
fi
mkdir -p "$dest/images" "$dest/landmarks"
unzip -q -n "$joined" -d "$dest/images"
unzip -q -n "$dest/landmarks.zip" -d "$dest/landmarks"
echo "anhir.sh: done; next: python -m benchmarks.regbench prepare anhir --source $dest --cases <cases>"
