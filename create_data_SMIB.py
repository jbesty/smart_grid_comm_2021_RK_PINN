import functools
import numpy as np
import scipy.integrate

from power_system_functions import create_system_matrices, ode_right_hand_side_solve, create_SMIB_system


def input_data_initialised(n_ops, power_system):
    """
    standard initialisation of the dataset dictionary
    """
    time_zeros = np.zeros((n_ops, 1))
    power_zeros = np.zeros((n_ops, power_system['n_buses']))
    states_initial = np.zeros((n_ops, power_system['n_states']))

    states_results_zeros = np.zeros((n_ops, power_system['n_states']))
    states_t_results_zeros = np.zeros((n_ops, power_system['n_states']))
    data_type_zeros = np.zeros((n_ops, power_system['n_states']))

    data_initialised = {'time': time_zeros,
                        'power': power_zeros,
                        'states_initial': states_initial,
                        'states_results': states_results_zeros,
                        'states_t_results': states_t_results_zeros,
                        'data_type': data_type_zeros}

    return data_initialised


def evaluate_op_trajectory(data_ops, n_time_steps, power_system):
    """
    evaluate the trajectory for each operating point (op), e.g., solve the ODE and then adjust the length of the
    dataset dictionary accordingly such that each point on a trajectory corresponds to a point.
    """
    n_ops = data_ops['time'].shape[0]
    t_max = power_system['t_max']

    t_span = np.concatenate([np.zeros(data_ops['time'].shape),
                             np.ones(data_ops['time'].shape) * t_max], axis=1)
    t_eval_vector = np.linspace(start=0, stop=t_max, num=n_time_steps).reshape((1, -1))
    t_eval = np.repeat(t_eval_vector, repeats=n_ops, axis=0)

    states_initial = data_ops['states_initial']
    A, B, C, D, F, G, u_0, x_0 = create_system_matrices(power_system=power_system)

    u_disturbance = data_ops['power'] @ G.T
    u = u_0.T + u_disturbance

    solver_func = functools.partial(solve_ode, A=A, B=B, C=C, D=D, F=F)

    solver_results = map(solver_func,
                         t_span,
                         t_eval,
                         states_initial,
                         u)

    list_solver_results = list(solver_results)

    states_results = np.concatenate([single_solver_result.T for single_solver_result in list_solver_results], axis=0)

    data_ops.update(time=t_eval.flatten().reshape((-1, 1)),
                    power=np.repeat(data_ops['power'], repeats=n_time_steps, axis=0),
                    states_initial=np.repeat(data_ops['states_initial'], repeats=n_time_steps, axis=0),
                    states_results=states_results,
                    data_type=np.repeat(data_ops['data_type'], repeats=n_time_steps, axis=0))

    return data_ops


def solve_ode(t_span,
              t_eval,
              states_initial,
              u, A, B, C, D, F):
    ode_solution = scipy.integrate.solve_ivp(ode_right_hand_side_solve,
                                             t_span=t_span,
                                             y0=states_initial.flatten(),
                                             args=[u, A, B, C, D, F],
                                             t_eval=t_eval,
                                             rtol=1e-13)

    return ode_solution.y


def calculate_data_ode_right_hand_side(data_ops, power_system):
    """
    evaluate the right hand side of the update rule and adjust the dataset dictionary accordingly
    """
    states_results = data_ops['states_results']
    A, B, C, D, F, G, u_0, x_0 = create_system_matrices(power_system=power_system)

    u_disturbance = data_ops['power'] @ G.T
    u = u_0.T + u_disturbance

    solver_func = functools.partial(ode_right_hand_side_solve, A=A, B=B, C=C, D=D, F=F)

    solver_results = map(solver_func,
                         data_ops['time'],
                         states_results,
                         u)

    list_solver_results = list(solver_results)

    states_t_results = np.concatenate([single_solver_result.reshape((1, -1)) for single_solver_result in
                                       list_solver_results],
                                      axis=0)

    data_ops.update(states_t_results=states_t_results)

    return data_ops


def create_dataset(n_power_steps, n_initial_conditions, n_time_steps, power_system):
    """
    dataset creation tailored to this case, i.e., operating points (ops) are defined by a initial rotor angle delta
    and a power disturbance.
    """
    data_ops = input_data_initialised(n_ops=n_power_steps * n_initial_conditions,
                                      power_system=power_system)

    power_values = np.linspace(0.0, 0.2, n_power_steps)
    delta_initial = np.linspace(-np.pi / 2, np.pi / 2, n_initial_conditions)

    power_ops_grid, delta_ops_grid = np.meshgrid(power_values, delta_initial)

    power_ops = power_ops_grid.reshape((-1, 1))
    delta_ops = delta_ops_grid.reshape((-1, 1))

    data_ops.update(time=np.ones(power_ops.shape) * power_system['t_max'],
                    power=power_ops,
                    states_initial=np.concatenate([delta_ops, power_system['omega_0'] + np.ones(delta_ops.shape) *
                                                   0.1], axis=1))

    data_ops = evaluate_op_trajectory(data_ops, n_time_steps=n_time_steps, power_system=power_system)

    data_ops = calculate_data_ode_right_hand_side(data_ops, power_system)

    return data_ops


if __name__ == '__main__':
    import pickle
    import global_PATHs

    power_system = create_SMIB_system()

    complete_dataset = create_dataset(n_power_steps=51,
                                      n_initial_conditions=51,
                                      n_time_steps=101,
                                      power_system=power_system)

    with open(global_PATHs.PATH_datasets / 'complete_dataset.pickle', 'wb') as f:
        pickle.dump(complete_dataset, f)
