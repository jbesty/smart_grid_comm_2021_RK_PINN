# !/usr/bin/env python3
import hashlib
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time

import global_PATHs
from train_model import train_model

tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)
tf.config.threading.set_intra_op_parallelism_threads(num_threads=1)


def setup_and_run():
    """
    This function create a setup table in which the parameters for several training runs of RK-PINNs are defined,
    each identified with a id (that practically guarantees uniqueness). The function furthermore runs the training
    processes in parallel.
    """
    tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads=1)

    current_time = int(time.time() * 1000)
    setup_id = hashlib.md5(str(current_time).encode())

    # --------------------------
    # definition of the different parameters for which PINNs shall be trained
    setup_table_names = ['N_datapoints',
                         'n_runge_kutta_stages',
                         'seed_tensorflow',
                         'seed_numpy']

    # elements to be usable in "itertools.product(*parameters)"
    N_datapoints = [50, 100, 200, 1000]
    n_runge_kutta_stages = [4, 8, 16, 32]
    seed_tensorflow = np.random.randint(0, 1000000, 7).tolist()
    seed_numpy = np.random.randint(0, 1000000, 7).tolist()

    parameters = [N_datapoints,
                  n_runge_kutta_stages,
                  seed_tensorflow,
                  seed_numpy]

    # create the 'setup_table' that contains a combination of all the parameters
    setup_table = pd.DataFrame(itertools.product(*parameters), columns=setup_table_names)

    # add setupID and a quasi unique simulation_id
    setup_table.insert(0, "setupID", setup_id.hexdigest())

    simulation_ids_unhashed = current_time + 1 + setup_table.index.values
    simulation_ids = []
    for simulation_id in simulation_ids_unhashed:
        simulation_ids_hashed = hashlib.md5(str(simulation_id).encode())
        simulation_ids.append(simulation_ids_hashed.hexdigest())

    setup_table.insert(1, "simulation_id", simulation_ids)

    # save the setup_table
    with open(global_PATHs.PATH_setup_tables / f'setupID_{setup_id.hexdigest()}.pickle', "wb") as f:
        pickle.dump(setup_table, f)

    print('Created setup table with %i entries' % setup_table.shape[0])

    # map the relevant variables for the parallelisation
    starmap_variables = [(simulation_id, N_datapoints, n_runge_kutta_stages, seed_tensorflow, seed_numpy) for
                         (simulation_id, N_datapoints, n_runge_kutta_stages, seed_tensorflow, seed_numpy) in
                         zip(setup_table['simulation_id'], setup_table['N_datapoints'],
                             setup_table['n_runge_kutta_stages'],
                             setup_table['seed_tensorflow'], setup_table['seed_numpy'])]

    # train the models in parallel
    with mp.Pool(4) as pool:
        pool.starmap(train_model, starmap_variables)

    pass


if __name__ == '__main__':
    setup_and_run()
