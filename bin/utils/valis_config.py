"""The BioFormats JVM-heap sizer for bin/warp_seg_qc.py.

`init_jvm(...)` sizes and starts the JVM for the reg_qc>=2 segmentation-overlap QC, which
constructs a real ``Valis`` and reads slides via BioFormats.
"""

import os


def _system_memory_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) // (1024**3)
    except Exception:
        return None


def init_jvm(input_dir, override_gb=None):
    """Size and start the BioFormats JVM heap to the inputs, for bin/warp_seg_qc.py.

    Heap (GB): total input size * 3 + 8, min 8, capped at 75% of system memory."""
    # Must run before the JVM starts: redirects the jgo cache off a read-only $HOME.
    from jvm_cache import point_jvm_cache_off_readonly_home
    from valis import registration

    point_jvm_cache_off_readonly_home()

    if override_gb is not None and override_gb > 0:
        mem_gb = int(override_gb)
    else:
        total_gb = 0.0
        for f in os.listdir(input_dir):
            if f.lower().endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
                total_gb += os.path.getsize(os.path.join(input_dir, f)) / (1024**3)
        sys_mem = _system_memory_gb()
        max_heap = int(sys_mem * 0.75) if sys_mem else 64
        mem_gb = max(8, min(max_heap, int(total_gb * 3 + 8)))
    registration.init_jvm(mem_gb=mem_gb)
    return mem_gb
