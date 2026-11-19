/*
========================================================================================
    SUBWORKFLOW: CHECKPOINT_WRITER
========================================================================================
    The only writer of a checkpoint manifest. Takes a step name and a channel of row
    Maps; writes `<outdir>/csv/<step>.csv`.

    The caller builds each row (which Layout kind, passthrough or prior-run path); this
    file decides whether, where, in what order and under what header it is written.
    Keep the Layout calls at the call sites: tests/test_layout.py scans them there.

    `sort: true` is collectFile's default but is stated so nobody sets `sort: false`,
    which writes rows in completion order and makes the manifest differ between runs.

    At a cleaning level no file is written at all: the source is filtered, and an empty
    channel into a seeded collectFile writes nothing.

    This is the only subworkflow that reads params.cleanup_level or collects into
    Layout.checkpointDir (tests/test_checkpoint_writer_is_the_only_writer.py).

    Input:
        step:    one of Checkpoint.STEPS[*].name
                 ('preprocessed' | 'registered' | 'segmented' | 'postprocessed'),
                 a plain String.
        ch_rows: channel of Map (column -> value), already passed through
                 Layout.publishedPath / Layout.publishedOrAsIs by the caller.

    Output:
        csv: the collected manifest (an empty channel at a cleaning level)
========================================================================================
*/

workflow CHECKPOINT_WRITER {
    take:
    step        // String: a Checkpoint.STEPS name
    ch_rows     // channel of Map: column -> value

    main:
    ch_checkpoint_csv = ch_rows
        .filter { Checkpoint.writesAtLevel(step, params.cleanup_level) }
        // Orders values by the declared columns; throws on a missing or unknown key.
        .map { row -> Checkpoint.row(step, row) }
        .collectFile(
            name: Layout.checkpointCsvName(step),
            newLine: true,
            sort: true,
            storeDir: Layout.checkpointDir(params.outdir),
            seed: Checkpoint.header(step)
        )

    emit:
    csv = ch_checkpoint_csv
}
