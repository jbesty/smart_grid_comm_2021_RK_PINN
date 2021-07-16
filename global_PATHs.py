import pathlib

# global PATH_data, PATH_results, PATH_training_history, PATH_model_weights, PATH_quantile, PATH_code, \
#     PATH_runge_kutta_parameters

# define the path where data are loaded from and saved to. Intended to store files that are not included in git.
# Declared such that relative paths can be used subsequently.
PATH_data = pathlib.Path('C:/Users/testuser/data')

PATH_datasets = PATH_data / 'datasets'

PATH_results = PATH_data / 'result_datasets'

PATH_model_weights = PATH_data / 'model_weights'

PATH_quantile = PATH_data / 'quantiles'

PATH_setup_tables = PATH_data / 'setup_tables'

PATH_code = pathlib.Path('C:/Users/testuser/code')

PATH_runge_kutta_parameters = PATH_code / 'IRK_parameters'


def check_directory_existence():
    if not PATH_data.exists():
        raise Exception("PATH_data doesn't exist!")

    if not PATH_code.exists():
        raise Exception("PATH_code doesn't exist!")

    if not PATH_datasets.exists():
        raise Exception("PATH_datasets doesn't exist!")

    if not PATH_results.exists():
        raise Exception("PATH_results doesn't exist!")

    if not PATH_model_weights.exists():
        raise Exception("PATH_model_weights doesn't exist!")

    if not PATH_quantile.exists():
        raise Exception("PATH_quantile doesn't exist!")

    if not PATH_setup_tables.exists():
        raise Exception("PATH_setup_tables doesn't exist!")

    if not PATH_runge_kutta_parameters.exists():
        raise Exception("PATH_runge_kutta_parameters doesn't exist!")

    print('All relevant global paths exist.')

    pass