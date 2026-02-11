class ParamUtils {

    static void validateStep(String step) {
        def valid = ['preprocessing', 'registration', 'postprocessing']
        if (!(step in valid)) {
            throw new IllegalArgumentException("Invalid --step '${step}'. Valid values: ${valid}")
        }
    }

    static void validateRegistrationMethod(String method) {
        def valid = ['valis', 'valis_pairs','gpu', 'cpu', 'cpu_tiled']
        if (!(method in valid)) {
            throw new IllegalArgumentException("Invalid --registration_method '${method}'. Valid values: ${valid}")
        }
    }

    static List requiredColumnsForStep(String step) {
        def requirements = [
            preprocessing : ['patient_id','path_to_file','is_reference','channels'],
            registration  : ['patient_id','preprocessed_image','is_reference','channels'],
            postprocessing: ['patient_id','registered_image','is_reference','channels']
        ]

        if (!requirements.containsKey(step)) {
            throw new IllegalArgumentException("No column requirements defined for step: ${step}")
        }

        return requirements[step]
    }
}
