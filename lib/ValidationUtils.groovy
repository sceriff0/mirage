/*
========================================================================================
    ValidationUtils - Input validation utilities
========================================================================================
*/

class ValidationUtils {

    /**
     * Validate input CSV has required columns
     */
    static def validateInputCSV(csv_path, required_cols) {
        def csv_file = new File(csv_path)

        if (!csv_file.exists()) {
            throw new Exception("Input CSV not found: ${csv_path}")
        }

        csv_file.withReader { reader ->
            def header = reader.readLine()
            if (!header) {
                throw new Exception("Input CSV is empty: ${csv_path}")
            }

            def columns = header.split(',').collect { it.trim() }

            required_cols.each { col ->
                if (!(col in columns)) {
                    throw new Exception("Input CSV missing required column: ${col}. Found columns: ${columns}")
                }
            }
        }

        return true
    }

    /**
     * Validate file exists and is readable
     */
    static def validateFileExists(file_path, description = "File") {
        def file = new File(file_path.toString())

        if (!file.exists()) {
            throw new Exception("${description} not found: ${file_path}")
        }

        if (!file.canRead()) {
            throw new Exception("${description} not readable: ${file_path}")
        }

        return true
    }

    /**
     * Validate parameter values
     */
    static def validateParameter(param_name, param_value, valid_values) {
        if (!(param_value in valid_values)) {
            throw new Exception("Invalid ${param_name}: '${param_value}'. Valid values: ${valid_values.join(', ')}")
        }
        return true
    }
}
