class ParamUtils {

    static void validateStep(String step) {
        def valid = ['preprocessing', 'registration', 'postprocessing', 'results', 'copy_results']
        if (!(step in valid)) {
            throw new IllegalArgumentException("Invalid --step '${step}'. Valid values: ${valid}")
        }
    }

    static void validateRegistrationMethod(String method) {
        def valid = ['valis', 'valis_pairs','gpu', 'cpu']
        if (!(method in valid)) {
            throw new IllegalArgumentException("Invalid --registration_method '${method}'. Valid values: ${valid}")
        }
    }

    static List requiredColumnsForStep(String step) {
        def requirements = [
            preprocessing : ['patient_id','path_to_file','is_reference','channels'],
            registration  : ['patient_id','preprocessed_image','is_reference','channels'],
            postprocessing: ['patient_id','registered_image','is_reference','channels'],
            results       : [
                'patient_id',
                'phenotype_csv','phenotype_geojson',
                'phenotype_mapping','merged_csv','cell_mask','pyramid'
            ],
            copy_results  : []  // No CSV required - just copies outdir to savedir
        ]

        if (!requirements.containsKey(step)) {
            throw new IllegalArgumentException("No column requirements defined for step: ${step}")
        }

        return requirements[step]
    }
}