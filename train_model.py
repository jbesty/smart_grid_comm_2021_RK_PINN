import numpy as np
import pickle
import tensorflow as tf

from dataset_handling import divide_dataset, prepare_data
import global_PATHs
from PINN_RK import PinnModel
from power_system_functions import create_SMIB_system


def train_model(simulation_id, N_datapoints, n_runge_kutta_stages, seed_tensorflow, seed_numpy):
    """
    Complete training process including
    - loading and splitting dataset
    - setup of the PINN
    - training of the PINN
    - evaluation of the PINN
    - saving result statistics and the weights of the trained model.

    :param simulation_id: unique identifier for each training run, used for naming files
    :param N_datapoints: number of data points in the training dataset
    :param n_runge_kutta_stages: number of Runge-Kutta stages
    :param seed_tensorflow: specify the initialisation of the weights and biases
    :param seed_numpy: control the random selection of the data points and validation points
    :return: nothing directly, but saves relevant statistics and trained weights
    """

    # check types of input variables
    if type(simulation_id) is not str:
        raise Exception('Provide simulation_id as string.')

    if type(N_datapoints) is not int:
        raise Exception('Provide N_datapoints as integer.')

    if type(n_runge_kutta_stages) is not int:
        raise Exception('Provide n_runge_kutta_stages as integer.')

    if type(seed_tensorflow) is not int:
        raise Exception('Provide seed_tensorflow as integer.')

    if type(seed_numpy) is not int:
        raise Exception('Provide seed_numpy as integer.')

    global_PATHs.check_directory_existence()
    # general training parameters
    epochs_first = 20000
    epochs_second = 80000

    initial_learning_rate = 0.05
    decay_learning_rate = 0.995
    neurons_in_hidden_layer = [50]

    # load/create the power system
    power_system = create_SMIB_system()
    n_states = power_system['n_states']

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_output_1_loss', min_delta=0, patience=100000, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )

    model = PinnModel(neurons_in_hidden_layer=neurons_in_hidden_layer,
                      n_runge_kutta_stages=n_runge_kutta_stages,
                      power_system=power_system,
                      seed=seed_tensorflow)

    # splitting and preparation of datasets
    with open(global_PATHs.PATH_datasets / 'complete_dataset.pickle', "rb") as file_loader:
        complete_dataset = pickle.load(file_loader)

    training_data, validation_data, testing_data = divide_dataset(dataset=complete_dataset,
                                                                  n_training_datapoints=N_datapoints,
                                                                  n_validation_datapoints=1000,
                                                                  sampling_seed=seed_numpy)
    X_training, y_training = prepare_data(dataset=training_data,
                                          n_runge_kutta_stages=n_runge_kutta_stages,
                                          n_states=n_states)

    X_validation, y_validation = prepare_data(dataset=validation_data,
                                              n_runge_kutta_stages=n_runge_kutta_stages,
                                              n_states=n_states)

    X_complete, y_complete = prepare_data(dataset=complete_dataset,
                                          n_runge_kutta_stages=n_runge_kutta_stages,
                                          n_states=n_states)

    training_weights = 1.0 + training_data['time'][:, 0] ** 2

    # weighing of the physics loss terms (dt)
    lambda_physics_delta = 1.0
    lambda_physics_omega = 1.0

    loss_weights_complete = np.concatenate([np.ones(n_states, dtype=np.float32) * 0.0,
                                            np.ones(1, dtype=np.float32) * lambda_physics_delta,
                                            np.ones(1, dtype=np.float32) * lambda_physics_omega,
                                            np.ones(n_runge_kutta_stages, dtype=np.float32) * 1.0,
                                            np.ones(n_runge_kutta_stages, dtype=np.float32) * 10.0,
                                            ],
                                           axis=0)

    # check that the length of loss_weights_complete, the output of PINNs and the training data have the same number
    # of components / lengths
    model.loss_term_dimension_check(loss_weights_complete=loss_weights_complete, y_training=y_training)

    # learning parameters and compilations
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_rate=decay_learning_rate,
        decay_steps=100)

    mse = tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler),
                  loss=[mse] * model.loss_terms,
                  loss_weights=loss_weights_complete.tolist())

    # ---- initial training without validation and early stopping ------------------
    _ = model.fit(X_training,
                  y_training,
                  sample_weight=training_weights,
                  initial_epoch=model.epoch_count,
                  epochs=model.epoch_count + epochs_first,
                  batch_size=int(X_training[0].shape[0]),
                  verbose=0,
                  shuffle=True,
                  )
    model.epoch_count = model.epoch_count + epochs_first

    # ---- training with validation and early stopping ------------------
    _ = model.fit(X_training,
                  y_training,
                  sample_weight=training_weights,
                  initial_epoch=model.epoch_count,
                  epochs=model.epoch_count + epochs_second,
                  batch_size=int(X_training[0].shape[0]),
                  validation_data=(X_validation, y_validation),
                  validation_freq=1,
                  verbose=0,
                  shuffle=True,
                  callbacks=[early_stopping_callback],
                  )

    model.epoch_count = model.epoch_count + epochs_second

    # set and save best weights
    model.set_weights(early_stopping_callback.best_weights)
    model.save_weights(filepath=global_PATHs.PATH_model_weights / f'weights_{simulation_id}.h5')

    # evaluate the quantiles of the squared error of delta
    complete_dataset['states_prediction'] = model.predict(X_complete)

    error = (complete_dataset['states_prediction'][:, 0:1] - complete_dataset['states_results'][:, 0:1]) ** 2
    print(f'{simulation_id} MSE delta: {np.mean(error):.03}, max SE delta: {np.max(error):.05}')

    quantile_values = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                                0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995,
                                0.998, 0.999, 1.000])

    quantile_results = np.quantile(error, quantile_values)

    # save quantile values
    with open(global_PATHs.PATH_quantile / f'quantiles_{simulation_id}.pickle', 'wb') as file_opener:
        pickle.dump(quantile_results, file_opener)

    # save prediction if space allows
    with open(global_PATHs.PATH_results / f'dataset_{simulation_id}.pickle', 'wb') as file_opener:
        pickle.dump(complete_dataset, file_opener)

    pass