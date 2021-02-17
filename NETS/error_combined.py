import numpy as np

import nengo
from nengo.dists import Choice

from utils import *

def generate(dof, axis, 
             direct_mode=False,
             means=None, scales=None,
             use_scale=False,
             use_intrcepts=False,
            n_scale=100):

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

    current_q = [0.0, 0.0, 0.0, 0.0, 0.0]

    
    net = nengo.Network('error_combined')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    net.config[nengo.Ensemble].seed = 0

    with net:
        # create / connect up error_combined --------------------------------------------------
        if use_intrcepts:
            net.error_combined = nengo.Ensemble(
                n_neurons=2000, dimensions=dim+axis,
                radius=np.sqrt(dim+axis),
                intercepts=get_intercepts(2000, dim+axis),
            )
            
            net.q_c = nengo.Ensemble(
            n_neurons=3000, dimensions=dim,
            radius=np.sqrt(dim),
                neuron_type = nengo.LIF(),
            )
            
        else:
            net.error_combined = nengo.Ensemble(
                n_neurons=3000, dimensions=dim+axis,
                radius=np.sqrt(dim+axis),
                )
            
            net.q_c = nengo.Ensemble(
            n_neurons=3000, dimensions=dim,
            radius=np.sqrt(dim),
                neuron_type = nengo.LIF(),
            )
            
        net.q_in = nengo.Node(current_q)
        
        
        
        nengo.Connection(net.q_in, net.q_c)
        
        net.J = nengo.Node(size_in=dim)
        
        # expecting input in form [dim+axis]
        if use_scale:
            net.input = nengo.Node(output=scale_down, size_in=dim+axis)
        else:
            net.input = nengo.Node(size_in=dim+axis)
        net.output = nengo.Node(size_in=dim)

        def J_func(error_q):
            """ calculate the jacobian matrix according to q_fixed and xyz_t """
            if use_scale:
                error_q = scale_up(error_q)
            q = error_q[0:5]
            J_x = calc_J(q)
            return np.dot(np.linalg.pinv(J_x), error_q[5:])
            
            
        # don't account for synapses twice
        nengo.Connection(net.input, net.error_combined, synapse=0.05)
        nengo.Connection(net.error_combined, net.J,function=J_func,synapse=0.05)              
        net.conn = nengo.Connection(net.q_c, net.output, synapse=0.05)
        net.conn.learning_rule_type = nengo.PES(learning_rate=0.0001)
        nengo.Connection(net.J, net.conn.learning_rule, synapse=0.05)


    return net