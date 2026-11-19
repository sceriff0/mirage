"""SegBackends' ctx.params reads must all be supplied by SEGMENT's ctx construction.

SEGMENT receives a slice of params, declared by SegBackends.CTX_PARAM_KEYS and
applied in subworkflows/local/segmentation.nf, because passing the whole map
(`params: params`) puts every parameter into SEGMENT's task hash and defeats
`-resume`.

The trap: `ctx.params.some_new_key` on a map that lacks the key yields null,
not an error, so the backend builds a command with an empty flag and the run
stays green. This test enforces "add the key in both places".
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEGMENT_NF = ROOT / "modules" / "local" / "segment.nf"
SEGMENTATION_NF = ROOT / "subworkflows" / "local" / "segmentation.nf"
SEG_BACKENDS = ROOT / "lib" / "SegBackends.groovy"


def _ctx_params_reads() -> set:
    """Keys SegBackends actually reads off ctx.params."""
    return set(
        re.findall(r"ctx\.params\.([a-zA-Z_][a-zA-Z0-9_]*)", SEG_BACKENDS.read_text())
    )


def _ctx_params_supplied() -> set:
    """Keys SegBackends.CTX_PARAM_KEYS declares, i.e. what reaches ctx.params."""
    text = SEG_BACKENDS.read_text()
    match = re.search(r"CTX_PARAM_KEYS\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert match, (
        "Could not find CTX_PARAM_KEYS in lib/SegBackends.groovy. If the declaration moved, "
        "update this test rather than deleting it."
    )
    return set(re.findall(r"'([a-zA-Z_][a-zA-Z0-9_]*)'", match.group(1)))


def test_every_ctx_params_read_is_supplied():
    reads = _ctx_params_reads()
    supplied = _ctx_params_supplied()
    assert reads, (
        "Found no ctx.params.* reads in lib/SegBackends.groovy - the regex has drifted."
    )
    missing = reads - supplied
    assert not missing, (
        f"lib/SegBackends.groovy reads ctx.params.{sorted(missing)} but CTX_PARAM_KEYS "
        f"does not declare {'it' if len(missing) == 1 else 'them'}. "
        "That read would evaluate to null and the backend would emit an empty flag with no "
        "error. Add the key to SegBackends.CTX_PARAM_KEYS."
    )


def test_no_unused_keys_are_supplied():
    """A supplied key nothing reads is dead weight that re-broadens the task hash."""
    unused = _ctx_params_supplied() - _ctx_params_reads()
    assert not unused, (
        f"lib/SegBackends.groovy declares CTX_PARAM_KEYS {sorted(unused)} but never reads "
        "them off ctx.params. Drop them: every key widens the set of parameters SEGMENT's "
        "cache key depends on, which is the whole point of the slice."
    )


def test_segment_does_not_pass_the_whole_params_map():
    """`params: params` would broaden the hash to every parameter."""
    text = SEGMENT_NF.read_text()
    assert not re.search(r"params\s*:\s*params\b", text), (
        "modules/local/segment.nf passes the whole params map into ctx again. Nextflow hashes "
        "the script block's free variables, so this binds SEGMENT's cache key to EVERY pipeline "
        "parameter and any unrelated param change re-runs segmentation and everything after it."
    )
