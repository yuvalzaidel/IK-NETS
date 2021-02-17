import numpy as np

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


    net = nengo.Network('euc')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    net.config[nengo.Ensemble].seed = 0
    with net:
        # create / connect up euc --------------------------------------------------
        if use_intrcepts:           
            net.euc = nengo.Ensemble(
                n_neurons=1000, dimensions=axis,
          #      radius=np.sqrt(axis),
                intercepts=get_intercepts(1000, axis))
        else:
            net.euc = nengo.Ensemble(
                n_neurons=1000, dimensions=axis,
           #     radius=np.sqrt(axis),
                )

        # expecting input in form [axis]
        if use_scale:
            net.input = nengo.Node(output=scale_down, size_in=axis)
        else:
            net.input = nengo.Node(size_in=axis)
        net.output = nengo.Node(size_in=1)

        def euc_func(euc):
            """ calculate the euclidean distance """
            if use_scale:
                euc = scale_up(euc)
            return np.sqrt(np.sum(euc**2))
            
        nengo.Connection(net.input, net.euc ,synapse=0.05)
        nengo.Connection(
            net.euc, net.output,
            function=euc_func,
            synapse=0.05)

    return net