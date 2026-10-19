#!/usr/bin/env bash
# Download and unpack HyReCo into <dest>, the folder `regbench prepare hyreco
# --data-root <dest>` reads:
#
#   <dest>/HyReCo/{HE,PHH3}/<id>.tif|.csv              subset A (HyReCo.zip, 233.36 GB)
#   <dest>/HyReCo-Additional/{HE,PHH3}/<id>.tif|.csv   subset B (two zips, 273.73 and
#                                                      232.58 GB), only with
#                                                      HYRECO_ADDITIONAL=1
#
#   HYRECO_URLS=<file> [HYRECO_ADDITIONAL=1] benchmarks/regbench/download/hyreco.sh <dest>
#
# Layout from the dataset page [FULL: the dataset page's text, fetched 2026-10-10]; the
# file names and sizes from IEEE DataPort [PARTIAL: IEEE DataPort page, file list fetched
# 2026-10-17]. The files need a DataPort login, so <file> holds one
# `url<TAB>name[<TAB>md5]` line per file: HyReCo.zip, and for subset B
# HyReCo-Additional-HE.zip and HyReCo-Additional-PHH3.zip. Without HYRECO_URLS the
# script refuses before any download.
#
# Only HE and PHH3 are unpacked (the pair regbench scores); the core zip also holds
# CD45, CD8 and KI67, which stay in the zip. Before each download the free space of
# <dest> is printed against twice the zip (the zip and what it unpacks to). A download
# is resumable: run the script again. HYRECO_ADDITIONAL only chooses what is
# downloaded; `prepare hyreco` takes subset B whenever it is there.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=benchmarks/regbench/download/common.sh
. "$here/common.sh"

dest=${1:?usage: HYRECO_URLS=<file> [HYRECO_ADDITIONAL=1] hyreco.sh <dest>}
if [ -z "${HYRECO_URLS:-}" ]; then
	echo "hyreco.sh: set HYRECO_URLS to a file of 'url<TAB>name[<TAB>md5]' lines" >&2
	echo "  (the links from the IEEE DataPort page, after logging in)" >&2
	exit 2
fi
# name, size in bytes (DataPort's file list, as recorded under "Verified on the dataset
# page" at 5a46f778:benchmarks/regbench/README.md:3141-3142; only the free-space hint), the folder
wanted=("HyReCo.zip 233360000000 HyReCo")
if [ "${HYRECO_ADDITIONAL:-0}" = 1 ]; then
	wanted+=("HyReCo-Additional-HE.zip 273730000000 HyReCo-Additional"
		"HyReCo-Additional-PHH3.zip 232580000000 HyReCo-Additional")
fi
urls=$(rb_urls "$HYRECO_URLS")
for w in "${wanted[@]}"; do
	read -r name _ _ <<<"$w"
	if ! printf '%s\n' "$urls" | awk -F'\t' -v w="$name" '$2 == w {f = 1} END {exit !f}'; then
		echo "hyreco.sh: $HYRECO_URLS names no '$name'" >&2
		exit 2
	fi
done
rb_need curl unzip df
mkdir -p "$dest"

for w in "${wanted[@]}"; do
	read -r name size folder <<<"$w"
	line=$(printf '%s\n' "$urls" | awk -F'\t' -v w="$name" '$2 == w {print; exit}')
	IFS=$'\t' read -r url _ md5 <<<"$line"
	rb_free "$dest" $((2 * size))
	rb_fetch "$url" "$dest/$name" "${md5:-}"
	case $name in
	HyReCo.zip) unzip -q -n "$dest/$name" "$folder/HE/*" "$folder/PHH3/*" -d "$dest" ;;
	*) unzip -q -n "$dest/$name" -d "$dest" ;;
	esac
done
for s in HE PHH3; do
	[ -d "$dest/HyReCo/$s" ] || {
		echo "hyreco.sh: $dest/HyReCo/$s is missing after unzip" >&2
		exit 1
	}
done
echo "hyreco.sh: done; next: python -m benchmarks.regbench prepare hyreco --data-root $dest --cases <cases>"
