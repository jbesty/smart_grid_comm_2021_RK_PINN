# Learning without Data: Physics-Informed Neural Networks for Fast Time-Domain Simulation

This repository is the official implementation of [Learning without Data: Physics-Informed Neural Networks for Fast Time-Domain Simulation](https://arxiv.org/abs/2106.15987) which we submitted to [SmartGridComm 2021](https://sgc2021.ieee-smartgridcomm.org/). 

## Environment

To install the requirements in your environment run:

```setup
pip install -r requirements.txt
```

## Code structure

The two main functions are 'train_model.py' and 'setup_and_run_parallel_training.py'. The former runs a single training process while the latter sets up a workflow to run multiple neural network trainings in parallel with different parameters.

'global_PATHs.py' defines the paths to which the previous two functions save the results. This file needs to be modified in order for the code to work. Furthermore, the specified folder structure should be created.

'create_data_SMIB.py' implements the dataset creation and 'dataset_handling.py' provides functions to filter, split, and prepare the dataset for the training.

'power_system_functions.py' contains all parameters of the single-machine infinite bus (SMIB) system as well as the governing functions for the differential equations.

'read_runge_kutta_parameters.py' converts the Runge-Kutta parameters stored in the files in 'IRK_parameters' into the correct format. The IRK_parameters stem from https://github.com/maziarraissi/PINNs.

'PINN_RK.py' implements the PinnModel (subclass of a Keras Model).

'timing_comparison.py' provides the functions to report the evaluation time of the RK_PINNs and other classical sovlers.

## Trained models
'weights_example_8_stages.h5' provides a set of trained neural network weights. They can be loaded to a model with 8 Runge-Kutta stages by calling for an initialised model
```setup
model.load_weights('weights_example_8_stages.h5')
```

## References
The implementation of the Physics-Informed Neural Networks is done in [TensorFlow](https://www.tensorflow.org) (Martín Abadi et al., TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.). 
