include { parseMetadata } from './metadata'

def loadCheckpointCsv(csv_path, image_column) {

    channel.fromPath(csv_path, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            [
                parseMetadata(row, "CSV ${csv_path}"),
                file(row[image_column])
            ]
        }
}
