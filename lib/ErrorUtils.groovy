/*
========================================================================================
    ErrorUtils - Standardized error messaging
========================================================================================
*/

class ErrorUtils {

    /**
     * Create a formatted pipeline error message
     */
    static String pipelineError(String step, String patient_id, String message, String hint = null) {
        def error_msg = """
        ❌ Pipeline Error in ${step}
        📍 Patient: ${patient_id}
        💬 Message: ${message}
        ${hint ? "💡 Hint: ${hint}" : ""}
        """.stripIndent()
        return error_msg
    }

    /**
     * Create a formatted warning message
     */
    static String pipelineWarning(String step, String patient_id, String message, String hint = null) {
        def warning_msg = """
        ⚠️  Pipeline Warning in ${step}
        📍 Patient: ${patient_id}
        💬 Message: ${message}
        ${hint ? "💡 Hint: ${hint}" : ""}
        """.stripIndent()
        return warning_msg
    }
}
