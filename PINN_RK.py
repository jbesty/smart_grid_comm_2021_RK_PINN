import numpy as np
import tensorflow as tf

from power_system_functions import create_system_matrices
from read_runge_kutta_parameters import read_runge_kutta_parameters


class PinnModel(tf.keras.models.Model):
    """
    The central model that predicts the Runge-Kutta stages h_hat and combines them into a prediction x1_hat and
    returns the mismatch in the implicit non-linear Runge-Kutta formulas.
    """

    def __init__(self, neurons_in_hidden_layer, n_runge_kutta_stages, power_system, seed=12345, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        tf.random.set_seed(seed)

        self.CoreNetwork = CoreNetwork(n_states=power_system['n_states'],
                                       n_runge_kutta_stages=n_runge_kutta_stages,
                                       neurons_in_hidden_layer=neurons_in_hidden_layer)

        self.n_states = power_system['n_states']
        self.n_generators = power_system['n_generators']
        self.n_buses = power_system['n_buses']
        self.epoch_count = 0
        self.seed = seed

        self.n_runge_kutta_stages = n_runge_kutta_stages

        self.loss_terms = self.n_states + self.n_states + self.n_states * self.n_runge_kutta_stages

        self.IRK_alpha, self.IRK_beta, self.IRK_gamma = read_runge_kutta_parameters(
            runge_kutta_stages=n_runge_kutta_stages)

        self.A, self.B, self.C, self.D, self.F, self.G, self.u_0, self.x_0, = create_system_matrices(
            power_system=power_system)

    def call(self, inputs, training=None, mask=None):
        # bring the inputs to 'CoreNetwork' in the right format and the other variables needed in the physics and RK
        # calculation
        x_time, x_power, x_y_0 = inputs
        delta_0 = x_y_0[:, 0:1]
        u_disturbance = x_power @ self.G.T
        u_physics = u_disturbance + self.u_0.T

        x_time_expanded = tf.expand_dims(x_time, axis=-1)
        x_y_0_expanded = tf.expand_dims(x_y_0, axis=-1)
        u = tf.expand_dims(u_disturbance, axis=1) + self.u_0.reshape((1, 1, -1))

        # call the 'CoreNetwork' to compute the runge_kutta_variables h_hat and then the state prediction x1_hat
        with tf.GradientTape(watch_accessed_variables=True,
                             persistent=True) as grad_t:
            grad_t.watch(x_time)
            runge_kutta_variables = self.CoreNetwork(inputs=[x_time, x_power, delta_0])

            runge_kutta_prediction = x_y_0 + x_time * tf.squeeze(
                runge_kutta_variables @ self.IRK_beta.T, axis=2)

        # calculate the derivative of x1_hat (d/dt x1_hat)
        runge_kutta_prediction_dt = tf.squeeze(grad_t.batch_jacobian(runge_kutta_prediction,
                                                                     x_time,
                                                                     unconnected_gradients='none'), axis=2)
        del grad_t

        runge_kutta_variables = self.CoreNetwork(inputs=[x_time, x_power, delta_0])
        FCX = tf.sin(runge_kutta_prediction @ self.C.T) @ self.D.T

        network_output_physics = runge_kutta_prediction_dt - (runge_kutta_prediction @ self.A.T +
                                                              FCX @
                                                              self.F.T + u_physics @ self.B.T)

        # calculate the approximate states h_hat and transpose it (_T) before applying the physics
        approximate_state = x_y_0_expanded + x_time_expanded * (runge_kutta_variables @ self.IRK_alpha.T)

        approximate_state_T = tf.transpose(approximate_state, perm=[0, 2, 1])

        # apply f() on the approximate state
        f_approximate_state = approximate_state_T @ self.A.T + \
                              tf.sin(approximate_state_T @ self.C.T) @ self.D.T @ self.F.T + \
                              u @ self.B.T

        f_approximate_state_T = tf.transpose(f_approximate_state, perm=[0, 2, 1])

        # evaluate the mismatch of the non-linear Runge-Kutta equations
        runge_kutta_stages_difference = runge_kutta_variables - f_approximate_state_T

        # prepare for loss value calculation
        runge_kutta_stages_difference_flattened = tf.reshape(runge_kutta_stages_difference,
                                                             shape=(-1, self.n_runge_kutta_stages * self.n_states))

        return_split = tf.split(runge_kutta_prediction, num_or_size_splits=self.n_states, axis=1) + \
                       tf.split(network_output_physics, num_or_size_splits=self.n_states, axis=1) + \
                       tf.split(runge_kutta_stages_difference_flattened,
                                num_or_size_splits=self.n_runge_kutta_stages * self.n_states,
                                axis=1)
        return return_split

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        x_time, x_power, x_y_0 = x
        delta_0 = x_y_0[:, 0:1]

        runge_kutta_variables = self.CoreNetwork(inputs=[x_time, x_power, delta_0], training=False)

        runge_kutta_prediction = x_y_0 + x_time * tf.squeeze(runge_kutta_variables @ self.IRK_beta.T, axis=2)

        return runge_kutta_prediction.numpy()

    def loss_term_dimension_check(self, loss_weights_complete, y_training):

        if self.loss_terms != len(loss_weights_complete):
            raise Exception('Loss terms and loss terms weight do not match in length.')

        if self.loss_terms != len(y_training):
            raise Exception('Loss terms and training data target do not match in length.')

        if len(loss_weights_complete) != len(y_training):
            raise Exception('Loss terms weight and training data target do not match in length.')

        pass


class CoreNetwork(tf.keras.models.Model):
    """
    This constitutes the core neural network within the PINN model. It outputs the prediction for the RK-stages based
    on the characteristic inputs.
    """

    def __init__(self,
                 n_states: int,
                 n_runge_kutta_stages: int,
                 neurons_in_hidden_layer: list):

        super(CoreNetwork, self).__init__()

        self.n_hidden_layers = len(neurons_in_hidden_layer)
        if not (self.n_hidden_layers > 0):
            raise Exception('Please specify the number of neurons per layer.')

        # TODO: Make it dataset dependent with .adapt(), currently normalisation hardcoded
        self.input_normalisation = tf.keras.layers.experimental.preprocessing.Normalization(axis=1,
                                                                                            name='input_normalisation',
                                                                                            mean=np.array([5.0,
                                                                                                           0.1, 0.0]),
                                                                                            variance=np.array([
                                                                                                8.5, 0.00367,
                                                                                                0.8554]))
        self.dense_layers = [tf.keras.layers.Dense(units=units,
                                                   activation=tf.keras.activations.tanh,
                                                   use_bias=True,
                                                   kernel_initializer=tf.keras.initializers.glorot_normal,
                                                   bias_initializer=tf.keras.initializers.zeros,
                                                   name=f'dense_layer_{ii}') for (ii, units) in enumerate(
            neurons_in_hidden_layer)]

        self.normalisation_layers = [
            tf.keras.layers.BatchNormalization(name=f'normalisation_layer_{ii + 1}', trainable=True)
            for ii in range(len(neurons_in_hidden_layer))]

        self.dense_output_layer = tf.keras.layers.Dense(units=n_states * n_runge_kutta_stages,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                                        bias_initializer=tf.keras.initializers.zeros,
                                                        name='output_layer')

        self.reshape_output_layer = tf.keras.layers.Reshape(target_shape=(n_states,
                                                                          n_runge_kutta_stages),
                                                            name='reshape_layer')

    def call(self, inputs, training=None, mask=None):
        input_concatenated = tf.keras.layers.concatenate(inputs, axis=1, name='input_concatenation')

        hidden_layer_input = self.input_normalisation(input_concatenated)

        for ii in range(self.n_hidden_layers):
            hidden_layer_output = self.dense_layers[ii](hidden_layer_input)
            hidden_layer_input = self.normalisation_layers[ii](hidden_layer_output)

        output_layer_output = self.dense_output_layer(hidden_layer_input)
        network_output = self.reshape_output_layer(output_layer_output)

        return network_output
