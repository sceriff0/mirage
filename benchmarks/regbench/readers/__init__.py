"""Raw-tile readers: each turns a vendor's tile folder into ``regbench.tiles`` cycles.

Memory (user ruling 2026-10-14): a reader holds ONE tile at a time (one z-stack for best
focus: the best plane and its score, never the stack), writes it, then reads the next.
"""
