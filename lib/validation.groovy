def validateStep(step) {
    def valid = ['preprocessing', 'registration', 'postprocessing', 'results']
    if (!(step in valid))
        error "Invalid --step '${step}'. Valid values: ${valid}"
}

def validateRegistrationMethod(method) {
    def valid = ['valis', 'gpu', 'cpu']
    if (!(method in valid))
        error "Invalid --registration_method '${method}'. Valid values: ${valid}"
}

def requiredColumnsForStep(step) {
    [
        preprocessing : ['patient_id','path_to_file','is_reference','channels'],
        registration  : ['patient_id','preprocessed_image','is_reference','channels'],
        postprocessing: ['patient_id','registered_image','is_reference','channels'],
        results       : [
            'patient_id','is_reference',
            'phenotype_csv','phenotype_mask',
            'phenotype_mapping','merged_csv','cell_mask'
        ]
    ][step]
}

def validateInputCSV(csv, required_cols) {

    def file = new File(csv)
    if (!file.exists())
        error "Input CSV not found: ${csv}"

    def header = file.readLines().first()?.split(',')*.trim()
    if (!header)
        error "CSV is empty: ${csv}"

    required_cols.each {
        if (!(it in header))
            error "CSV missing required column '${it}'"
    }
}
