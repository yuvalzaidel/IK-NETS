import numpy as np
from sklearn.metrics import mean_squared_error
import nengo

from utils import *

def generate(dof, axis, 
             direct_mode=False,
             means=None, scales=None,
             use_scale=False,
             use_intrcepts=False):
    dim = dof

    # NOTE: This function will scale the input so that each dimensions is
    # in the range of -1 to 1. Since we know the operating space of the arm
    # we can set these specifically. This is a hack so that we don't need
    # 100k neurons to be able to simulate accurately generated movement,
    # you can think of it as choosing a relevant part of motor cortex to run.
    # Without this scaling, it would work still, but it would require
    # significantly more neurons to achieve the same level of performance.
    means = np.zeros(dim) if means is None else means
    scales = np.ones(dim) if scales is None else scales
    scale_down, scale_up = generate_scaling_functions(
        np.asarray(means), np.asarray(scales))


    net = nengo.Network('mse')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    net.config[nengo.Ensemble].seed = 0
    with net:
        # create / connect up mse --------------------------------------------------
        if use_intrcepts:
            net.mse = nengo.Ensemble(
                n_neurons=1000, dimensions=axis*2,
         #       radius=np.sqrt(axis*2),
                intercepts=get_intercepts(1000, axis*2))
        else:
            net.mse = nengo.Ensemble(
                n_neurons=1000, dimensions=axis*2,
         #       radius=np.sqrt(axis*2),
                )

        # expecting input in form [axis*2]
        if use_scale:
            net.input = nengo.Node(output=scale_down, size_in=axis*2)
        else:
            net.input = nengo.Node(size_in=axis*2)
        net.output = nengo.Node(size_in=1)

        def mse_func(mse):
            """ calculate the MSE """
            if use_scale:
                mse = scale_up(mse)
            return mean_squared_error(mse[0:3],mse[3:6])
            
        nengo.Connection(net.input, net.mse, synapse=0.05)
        nengo.Connection(
            net.mse, net.output,
            function=mse_func,
            synapse=0.05)

    return net