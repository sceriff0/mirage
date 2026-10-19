# shellcheck shell=bash
# Shared by the regbench download scripts (anhir.sh, hyreco.sh); sourced, never run.
#
#   rb_need <command>...            refuse (exit 2) when a command is not on PATH
#   rb_fetch <url> <dest> [<md5>]   download <url> to <dest>: curl into <dest>.part,
#                                   resumable (-C -), 5 retries; with <md5>, the
#                                   checksum must match before the .part becomes <dest>;
#                                   a mismatch leaves the .part and fails. An existing
#                                   <dest> is kept (and checked when <md5> is given).
#   rb_urls <file>                  the `url<TAB>name[<TAB>md5]` lines of a URL file,
#                                   blank lines and `#` comments dropped
#   rb_free <dir> <bytes>           print the free space of <dir> against <bytes>
#
# Nothing here swallows a failure: every function returns non-zero and says why.

rb_need() {
	local c
	for c in "$@"; do
		if ! command -v "$c" >/dev/null 2>&1; then
			echo "regbench download: '$c' is not on PATH" >&2
			exit 2
		fi
	done
}

rb_md5() {
	if command -v md5sum >/dev/null 2>&1; then
		md5sum "$1" | awk '{print $1}'
	else
		md5 -q "$1"
	fi
}

rb_check_md5() { # <file> <md5>
	local got
	got=$(rb_md5 "$1") || return 1
	if [ "$got" != "$2" ]; then
		echo "regbench download: $1: md5 $got, expected $2" >&2
		return 1
	fi
}

rb_fetch() { # <url> <dest> [<md5>]
	local url=$1 dest=$2 md5=${3:-}
	if [ -e "$dest" ]; then
		echo "present: $dest"
		if [ -n "$md5" ]; then rb_check_md5 "$dest" "$md5" || return 1; fi
		return 0
	fi
	mkdir -p "$(dirname "$dest")"
	echo "fetch: $url -> $dest"
	curl -fL --retry 5 -C - -o "$dest.part" "$url" || {
		echo "regbench download: curl failed for $url (partial kept at $dest.part)" >&2
		return 1
	}
	if [ -n "$md5" ]; then
		rb_check_md5 "$dest.part" "$md5" || {
			echo "regbench download: $dest.part kept; delete it to start over" >&2
			return 1
		}
	fi
	mv "$dest.part" "$dest"
}

rb_urls() { # <file>
	if [ ! -r "$1" ]; then
		echo "regbench download: URL file '$1' is not readable" >&2
		return 1
	fi
	grep -v '^[[:space:]]*#' "$1" | grep -v '^[[:space:]]*$'
}

rb_free() { # <dir> <bytes>
	local kb
	kb=$(df -Pk "$1" | awk 'NR == 2 {print $4}')
	awk -v kb="$kb" -v need="$2" -v d="$1" 'BEGIN {
		printf "free space of %s: %.1f GB; needed (twice the archive): %.1f GB%s\n",
			d, kb * 1024 / 1e9, need / 1e9, (kb * 1024 < need ? "  -- NOT ENOUGH" : "")
	}'
}
