import numpy as np


def create_SMIB_system():
    """
    Create a dictionary containing all relevant parameters of the power system.
    """
    t_max = 10
    n_buses = 1
    n_generators = 1
    n_non_generators = n_buses - n_generators
    n_states = 2 * n_generators + 1 * n_non_generators

    omega_0 = 2 * np.pi * 50

    output_scaling = np.ones((n_states, 1))
    output_scaling[n_generators:2 * n_generators] = omega_0

    output_offset = np.zeros((n_states, 1))
    output_offset[n_generators:2 * n_generators] = -omega_0

    H_generators = np.array([omega_0 / 2 / 0.4])
    D_generators = np.array([0.15])
    K_g_generators = 0.5 * np.ones(n_generators)
    T_m_generators = np.array([0.1])
    T_g_generators = 0.01 * np.ones(n_generators)

    D_non_generators = np.array([0.1])

    P_load_set_point = np.array([0])
    P_generator_set_point = np.array([0])
    P_set_point = np.hstack([P_generator_set_point, P_load_set_point])

    P_disturbance = np.zeros(n_buses)
    slack_bus_idx = 2
    #
    V_i_V_j_B_full = np.array([0.2])
    V_i_V_j_B_short_circuit = np.array([0.2])
    V_i_V_j_B_line_tripped = np.array([0.2])
    incidence_matrix = np.ones((1, 1))

    system_parameters = {'n_buses': n_buses,
                         'n_generators': n_generators,
                         'n_non_generators': n_non_generators,
                         'n_states': n_states,
                         'slack_bus_idx': slack_bus_idx,
                         'H_generators': H_generators,
                         'D_generators': D_generators,
                         'omega_0': omega_0,
                         'output_scaling': output_scaling,
                         'T_m_generators': T_m_generators,
                         'T_g_generators': T_g_generators,
                         'K_g_generators': K_g_generators,
                         'D_non_generators': D_non_generators,
                         'P_disturbance': P_disturbance,
                         'P_set_point': P_set_point,
                         'V_i_V_j_B_full': V_i_V_j_B_full,
                         'V_i_V_j_B_short_circuit': V_i_V_j_B_short_circuit,
                         'V_i_V_j_B_line_tripped': V_i_V_j_B_line_tripped,
                         'incidence_matrix': incidence_matrix,
                         't_max': t_max,
                         'output_offset': output_offset}

    print('Successfully created a SMIB system (1 bus, 1 generator)!')

    return system_parameters


def create_system_matrices(power_system):
    """
    define all relevant system matrices
    f(x) = A @ x + B @ u + F @ FCX
    C and D are used in the calculation of the non-linear term FCX
    G maps a power disturbance (P) onto u -> u = u_0 + G @ P
    """

    A = np.array([[0.0, 1.0],
                  [0.0, (-power_system['omega_0'] / (2 * power_system['H_generators']) * (power_system[
                      'D_generators']))[0]]])

    B = np.array([[-1.0, 0.0],
                  [0.0, (power_system['omega_0'] / (2 * power_system['H_generators']))[0]]])

    C = np.array([[1.0, 0.0]])

    D = np.array([[power_system['V_i_V_j_B_full'][0]]])

    F = np.array([[0.0],
                  [(-power_system['omega_0'] / (2 * power_system['H_generators']))[0]]])

    G = np.array([[0.0],
                  [1.0]])

    u_0 = np.hstack([power_system['omega_0'],
                    power_system['D_generators'] * power_system['omega_0']]).reshape((-1, 1))

    # initial value for equilibrium computation
    x_0 = np.array([[0.0, power_system['omega_0']]]).reshape((-1, 1))

    return A, B, C, D, F, G, u_0, x_0


def ode_right_hand_side_solve(t, x, u, A, B, C, D, F):
    """
    update rule for solving the ODE for example with scipy.integrate
    """
    x_vector = np.reshape(x, (-1, 1))
    u_vector = np.reshape(u, (-1, 1))

    FCX = D @ np.sin(C @ x_vector)

    dx = A @ x_vector + F @ FCX + B @ u_vector
    return dx[:, 0]



