"""ANHIR landmark-accuracy harness for the pipeline's registration backends.

Scores any registration method on the public ANHIR challenge
(https://anhir.grand-challenge.org/) with the challenge's own metrics -- rTRE
(TRE / image diagonal), robustness, and the mean rank of per-case median rTRE --
and packages the evaluation cases in the challenge's submission format.

Pipeline of modules, in run order:

    dataset   -- read the challenge's cover CSV into typed cases
    prepare   -- join the split archive, convert JPEGs to OME-TIFF, write the samplesheet
    (run_pairs.sh drives the pipeline with --start/--stop registration)
    warp      -- push source landmarks through each method's published transform
    evaluate  -- score warped landmarks and emit the hand-off tables + submission

Only training cases carry public target landmarks; evaluation cases are scored
server-side, so for them this harness produces the submission package only.
"""
