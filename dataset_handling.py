import numpy as np


def filter_dataset(dataset, filter_indices):
    """
    Filter a dataset consisting of a dictionary for the different variables by a vector of True/False indicating if
    the data point (a row index) should be kept or not.
    Returning a copy to leave the original dataset untouched.
    """
    dataset_copy = dataset.copy()
    for key in dataset_copy.keys():
        dataset_copy[key] = dataset_copy[key][filter_indices, :]

    return dataset_copy


def divide_dataset(dataset,
                   n_training_datapoints,
                   n_validation_datapoints,
                   sampling_seed=65652):
    """
    Split the dataset into training, validation, and test dataset defined by the number of points in the former two.
    """
    np.random.seed(sampling_seed)
    dataset_length = len(dataset['time'])

    dataset_indices = np.arange(0, stop=dataset_length)
    training_indices = np.random.choice(dataset_indices, size=(n_training_datapoints,), replace=False)
    remaining_dataset_indices = np.setdiff1d(dataset_indices, training_indices)
    validation_indices = np.random.choice(remaining_dataset_indices, size=(n_validation_datapoints,), replace=False)
    testing_indices = np.setdiff1d(remaining_dataset_indices, validation_indices)

    training_data = filter_dataset(dataset=dataset, filter_indices=training_indices)
    validation_data = filter_dataset(dataset=dataset, filter_indices=validation_indices)
    testing_data = filter_dataset(dataset=dataset, filter_indices=testing_indices)

    return training_data, validation_data, testing_data


def prepare_data(dataset, n_runge_kutta_stages, n_states):
    """
    Preparing the data to give the right formats for the training, evaluation process.
    """
    X_dataset = [dataset['time'],
                 dataset['power'],
                 dataset['states_initial']]

    y_dataset = np.split(dataset['states_results'], indices_or_sections=n_states, axis=1) + \
                [np.zeros((dataset['states_initial'].shape[0], 1))] * n_states + \
                [np.zeros((dataset['states_initial'].shape[0], 1))] * n_states * n_runge_kutta_stages

    return X_dataset, y_dataset
