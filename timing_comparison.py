import itertools
import functools
import numpy as np
import pandas as pd
import pickle
from scipy import integrate
import scipy.optimize
import timeit

import global_PATHs
from dataset_handling import filter_dataset, prepare_data
from PINN_RK import PinnModel
from power_system_functions import create_system_matrices, ode_right_hand_side_solve, create_SMIB_system
from read_runge_kutta_parameters import read_runge_kutta_parameters


# --------------------------
# Timing function to compare the performance of RK-PINNs versus classical implicit and explicit solution schemes,
# used to produce the results shown in the paper.

# --------------------------
# Implementation to solve the full implicit Runge-Kutta scheme
def solve_RK_stages(x_init,
                    delta_t,
                    u,
                    kM_init,
                    n_runge_kutta_stages,
                    IRK_alpha,
                    IRK_beta,
                    A, B, C, D, F, ):
    Resid_P = functools.partial(RK_residual,
                                x_0=x_init,
                                delta_t=delta_t,
                                u=u,
                                n_runge_kutta_stages=n_runge_kutta_stages,
                                IRK_alpha=IRK_alpha, A=A, B=B,
                                C=C, D=D,
                                F=F)

    h_RK = scipy.optimize.newton_krylov(Resid_P, kM_init, f_tol=1.0e-13, verbose=False)

    x_prediction = x_init.reshape((-1, 1)) + delta_t * (h_RK @ IRK_beta.T)

    return x_prediction


def RK_residual(h_k, x_0, delta_t, u, n_runge_kutta_stages, IRK_alpha, A, B, C, D, F):
    x_0_extended = np.repeat(x_0.reshape((-1, 1)), repeats=n_runge_kutta_stages, axis=1)
    u_extended = np.repeat(u.reshape((-1, 1)), repeats=n_runge_kutta_stages, axis=1)
    x_adjusted = x_0_extended + delta_t * (h_k @ IRK_alpha.T)
    FCX = D @ np.sin(C @ x_adjusted)
    f_h_k = A @ x_adjusted + F @ FCX + B @ u_extended

    return h_k - f_h_k


def solve_mapped_RK(solver_func_RK, x_init, delta_t, u, kM_init):
    solver_results = map(solver_func_RK, x_init, delta_t, u, kM_init)
    list_solver_results = list(solver_results)
    states_results = np.concatenate([single_solver_result.T for single_solver_result in list_solver_results], axis=0)
    return states_results


# --------------------------
# functions to run integration schemes implemented in scipy.integrate
def solve_ode(t_span,
              t_eval,
              states_initial,
              u, A, B, C, D, F,
              tolerance=2.3e-14,
              method='RK45'):
    ode_solution = integrate.solve_ivp(ode_right_hand_side_solve,
                                       t_span=t_span,
                                       y0=states_initial.flatten(),
                                       args=[u, A, B, C, D, F],
                                       t_eval=t_eval,
                                       rtol=tolerance,
                                       method=method)

    return ode_solution.y


def solve_mapped_ode(solver_func, t_span, t_eval, states_initial, u):
    solver_results = map(solver_func, t_span, t_eval, states_initial, u)
    list_solver_results = list(solver_results)
    states_results = np.concatenate([single_solver_result.T for single_solver_result in list_solver_results], axis=0)
    return states_results


# --------------------------

with open(global_PATHs.PATH_datasets / 'complete_dataset.pickle', "rb") as f:
    complete_data = pickle.load(f)

power_system = create_SMIB_system()
A, B, C, D, F, G, u_0, x_0 = create_system_matrices(power_system=power_system)

# timing function parameters: repetitions is within each execution, and repeats is the number of executions of the
# timit command.
n_repetitions = 20
n_repeats = 5

# values to explore for the time steps and the number of Runge-Kutta stages in the different schemes
time_steps_list = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
n_runge_kutta_stages_list = list([4, 32])

# list to store all results
result_dict_list = list()

# time various implicit Runge-Kutta (IRK) methods
for time_step, n_runge_kutta_stages in itertools.product(time_steps_list, n_runge_kutta_stages_list):
    delta_t = time_step

    indices_time_steps = np.isin(np.around(complete_data['time'][:, 0], decimals=5), time_step)

    dataset_filtered_pre = filter_dataset(dataset=complete_data, filter_indices=indices_time_steps)

    indices_single = np.zeros(51, dtype=bool)
    indices_single[0::5] = True
    indices_reduced_dataset_power = np.logical_and(np.repeat(indices_single, repeats=51),
                                                   np.tile(indices_single, 51))

    dataset_filtered = filter_dataset(dataset=dataset_filtered_pre, filter_indices=indices_reduced_dataset_power)
    states_initial = dataset_filtered['states_initial']
    t_span = np.concatenate([dataset_filtered['time'] * 0,
                             dataset_filtered['time']], axis=1)

    u_disturbance = dataset_filtered['power'] @ G.T
    u = u_0.T + u_disturbance

    t_eval = dataset_filtered["time"]

    IRK_alpha, IRK_beta, IRK_gamma = read_runge_kutta_parameters(runge_kutta_stages=n_runge_kutta_stages)

    x_init = states_initial

    FCX = np.sin(x_init @ C.T) @ D.T
    k0 = x_init @ A.T + FCX @ F.T + u @ B.T
    xM = np.repeat(np.expand_dims(x_init, axis=2), repeats=n_runge_kutta_stages, axis=2)
    kM = np.repeat(np.expand_dims(k0, axis=2), repeats=n_runge_kutta_stages, axis=2)
    xM = xM + delta_t * kM @ np.diag(IRK_gamma.flatten())

    xM_T = np.transpose(xM, axes=[0, 2, 1])
    # Pass through f(x)
    FCX_T = np.sin(xM_T @ C.T) @ D.T
    kM_init_T = xM_T @ A.T + FCX_T @ F.T + np.expand_dims(u @ B.T, axis=1)
    kM_init = np.transpose(kM_init_T, axes=[0, 2, 1])

    solver_func_rk = functools.partial(solve_RK_stages,
                                       n_runge_kutta_stages=n_runge_kutta_stages,
                                       IRK_alpha=IRK_alpha,
                                       IRK_beta=IRK_beta,
                                       A=A, B=B, C=C, D=D, F=F)

    total_time_rk = timeit.repeat('solve_mapped_ode(solver_func_rk, x_init, t_eval, u, kM_init)',
                                  number=n_repetitions,
                                  repeat=n_repeats,
                                  globals=globals(),
                                  )

    rk_results = solve_mapped_ode(solver_func_rk, x_init, t_eval, u, kM_init)
    error_rk = rk_results - dataset_filtered['states_results']
    print(f'Delta t = {time_step}s')
    print(f'RK-{n_runge_kutta_stages}: {np.max(np.abs(error_rk))}')
    print(f'RK-{n_runge_kutta_stages}: {min(total_time_rk) / n_repetitions / 121}s')

    result_dict = {'method': 'IRK',
                   'n_runge_kutta_stages': n_runge_kutta_stages,
                   'time_step': time_step,
                   'time_min': min(total_time_rk) / n_repetitions / 121,
                   'max_error': np.max(np.abs(error_rk))}

    result_dict_list.append(result_dict)

# time Radau method
for time_step in time_steps_list:
    delta_t = time_step

    indices_time_steps = np.isin(np.around(complete_data['time'][:, 0], decimals=5), time_step)

    dataset_filtered_pre = filter_dataset(dataset=complete_data, filter_indices=indices_time_steps)

    indices_single = np.zeros(51, dtype=bool)
    indices_single[0::5] = True
    indices_reduced_dataset_power = np.logical_and(np.repeat(indices_single, repeats=51),
                                                   np.tile(indices_single, 51))

    dataset_filtered = filter_dataset(dataset=dataset_filtered_pre, filter_indices=indices_reduced_dataset_power)
    states_initial = dataset_filtered['states_initial']
    t_span = np.concatenate([dataset_filtered['time'] * 0,
                             dataset_filtered['time']], axis=1)

    u_disturbance = dataset_filtered['power'] @ G.T
    u = u_0.T + u_disturbance

    t_eval = dataset_filtered["time"]

    solver_func_radau = functools.partial(solve_ode, A=A, B=B, C=C, D=D, F=F, method='Radau')

    total_time_radau = timeit.repeat('solve_mapped_ode(solver_func_radau, t_span, t_eval, states_initial, u)',
                                     number=n_repetitions,
                                     repeat=n_repeats,
                                     globals=globals(),
                                     )

    radau_results = solve_mapped_ode(solver_func_radau, t_span, t_eval, states_initial, u)

    error_radau = radau_results - dataset_filtered['states_results']

    print(f'Delta t = {time_step}s')
    print(f'Radau: {np.max(np.abs(error_radau))}')
    print(f'Radau: {min(total_time_radau) / n_repetitions / 121}')

    result_dict = {'method': 'Radau',
                   'n_runge_kutta_stages': 1,
                   'time_step': time_step,
                   'time_min': min(total_time_radau) / n_repetitions / 121,
                   'max_error': np.max(np.abs(error_radau))}

    result_dict_list.append(result_dict)

# time RK-45 method
for time_step in time_steps_list:
    delta_t = time_step

    indices_time_steps = np.isin(np.around(complete_data['time'][:, 0], decimals=5), time_step)

    dataset_filtered_pre = filter_dataset(dataset=complete_data, filter_indices=indices_time_steps)

    indices_single = np.zeros(51, dtype=bool)
    indices_single[0::5] = True
    indices_reduced_dataset_power = np.logical_and(np.repeat(indices_single, repeats=51),
                                                   np.tile(indices_single, 51))

    dataset_filtered = filter_dataset(dataset=dataset_filtered_pre, filter_indices=indices_reduced_dataset_power)
    states_initial = dataset_filtered['states_initial']
    t_span = np.concatenate([dataset_filtered['time'] * 0,
                             dataset_filtered['time']], axis=1)

    u_disturbance = dataset_filtered['power'] @ G.T
    u = u_0.T + u_disturbance

    t_eval = dataset_filtered["time"]

    solver_func_rk45 = functools.partial(solve_ode, A=A, B=B, C=C, D=D, F=F, method='RK45')

    total_time_rk45 = timeit.repeat('solve_mapped_ode(solver_func_rk45, t_span, t_eval, states_initial, u)',
                                    number=n_repetitions,
                                    repeat=n_repeats,
                                    globals=globals(),
                                    )

    rk45_results = solve_mapped_ode(solver_func_rk45, t_span, t_eval, states_initial, u)

    error_rk45 = rk45_results - dataset_filtered['states_results']

    print(f'Delta t = {time_step}s')
    print(f'RK45: {np.max(np.abs(error_rk45))}')
    print(f'RK45: {min(total_time_rk45) / n_repetitions / 121}')

    result_dict = {'method': 'RK45',
                   'n_runge_kutta_stages': 0,
                   'time_step': time_step,
                   'time_min': min(total_time_rk45) / n_repetitions / 121,
                   'max_error': np.max(np.abs(error_rk45))}

    result_dict_list.append(result_dict)

# time a RK-PINN
for time_step, n_runge_kutta_stages in itertools.product(time_steps_list, n_runge_kutta_stages_list):
    delta_t = time_step

    model = PinnModel(neurons_in_hidden_layer=[50],
                      n_runge_kutta_stages=n_runge_kutta_stages,
                      power_system=power_system,
                      case='normal')

    indices_time_steps = np.isin(np.around(complete_data['time'][:, 0], decimals=5), time_step)

    dataset_filtered_pre = filter_dataset(dataset=complete_data, filter_indices=indices_time_steps)

    indices_single = np.zeros(51, dtype=bool)
    indices_single[0::5] = True
    indices_reduced_dataset_power = np.logical_and(np.repeat(indices_single, repeats=51),
                                                   np.tile(indices_single, 51))

    dataset_filtered = filter_dataset(dataset=dataset_filtered_pre, filter_indices=indices_reduced_dataset_power)

    X_test, y_test = prepare_data(dataset_filtered, n_runge_kutta_stages=n_runge_kutta_stages, n_states=2)

    total_time_PINN = timeit.repeat('model.predict(X_test)',
                                    number=n_repetitions,
                                    repeat=n_repeats,
                                    globals=globals(),
                                    )

    error_PINN = model.predict(X_test) - dataset_filtered['states_results']

    result_dict = {'method': 'PINN',
                   'n_runge_kutta_stages': n_runge_kutta_stages,
                   'time_step': time_step,
                   'time_min': min(total_time_PINN) / n_repetitions / 121,
                   'max_error': np.max(np.abs(error_PINN))}

    print(f'Delta t = {time_step}s')
    print(f'PINN: {np.max(np.abs(error_PINN))}')
    print(f'PINN: {min(total_time_PINN) / n_repetitions / 121}')

    result_dict_list.append(result_dict)

# time a larger PINN
for time_step, n_runge_kutta_stages in itertools.product(time_steps_list, n_runge_kutta_stages_list):
    delta_t = time_step

    model = PinnModel(neurons_in_hidden_layer=[500, 500, 500],
                      n_runge_kutta_stages=n_runge_kutta_stages,
                      power_system=power_system)

    indices_time_steps = np.isin(np.around(complete_data['time'][:, 0], decimals=5), time_step)

    dataset_filtered_pre = filter_dataset(dataset=complete_data, filter_indices=indices_time_steps)

    indices_single = np.zeros(51, dtype=bool)
    indices_single[0::5] = True
    indices_reduced_dataset_power = np.logical_and(np.repeat(indices_single, repeats=51),
                                                   np.tile(indices_single, 51))

    dataset_filtered = filter_dataset(dataset=dataset_filtered_pre, filter_indices=indices_reduced_dataset_power)

    X_test, y_test = prepare_data(dataset_filtered, n_runge_kutta_stages=n_runge_kutta_stages, n_states=2)

    total_time_PINN = timeit.repeat('model.predict(X_test)',
                                    number=n_repetitions,
                                    repeat=n_repeats,
                                    globals=globals(),
                                    )

    error_PINN = model.predict(X_test) - dataset_filtered['states_results']

    result_dict = {'method': 'PINN_larger',
                   'n_runge_kutta_stages': n_runge_kutta_stages,
                   'time_step': time_step,
                   'time_min': min(total_time_PINN) / n_repetitions / 121,
                   'max_error': np.max(np.abs(error_PINN))}

    print(f'Delta t = {time_step}s')
    print(f'PINN: {np.max(np.abs(error_PINN))}')
    print(f'PINN: {min(total_time_PINN) / n_repetitions / 121}')

    result_dict_list.append(result_dict)

# concatenate all results and write into a table
results = np.hstack([np.stack([element['method'] for element in result_dict_list]).reshape((-1, 1)),
                     np.stack([element['n_runge_kutta_stages'] for element in result_dict_list]).reshape((-1, 1)),
                     np.stack([element['time_step'] for element in result_dict_list]).reshape((-1, 1)),
                     np.stack([element['time_min'] for element in result_dict_list]).reshape((-1, 1)),
                     np.stack([element['max_error'] for element in result_dict_list]).reshape((-1, 1))])

results_pd = pd.DataFrame(results)
with open(global_PATHs.PATH_data / 'timing_metrics' / 'timing_table.pickle', 'wb') as f:
    pickle.dump(results_pd, f)
