import global_PATHs
import numpy as np


def read_runge_kutta_parameters(runge_kutta_stages):
    tmp = np.float32(np.loadtxt(str(global_PATHs.PATH_runge_kutta_parameters / f'Butcher_IRK{runge_kutta_stages}.txt'),
                                ndmin=2))
    IRK_alpha = np.reshape(tmp[0:runge_kutta_stages ** 2], (runge_kutta_stages, runge_kutta_stages))
    IRK_beta = np.reshape(tmp[runge_kutta_stages ** 2:runge_kutta_stages ** 2 + runge_kutta_stages],
                          (1, runge_kutta_stages))
    IRK_gamma = np.reshape(tmp[runge_kutta_stages ** 2 + runge_kutta_stages:],
                           (runge_kutta_stages, 1))

    return IRK_alpha, IRK_beta, IRK_gamma
