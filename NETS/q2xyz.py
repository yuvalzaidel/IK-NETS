import numpy as np

import nengo
from nengo.dists import Choice

from utils import *

def calc_T(q):

    c0 = np.cos(q[0])
    c1 = np.cos(q[1])
    c2 = np.cos(q[2])
    c3 = np.cos(q[3])
    c4 = np.cos(q[4])

    s0 = np.sin(q[0])
    s1 = np.sin(q[1])
    s2 = np.sin(q[2])
    s3 = np.sin(q[3])
    s4 = np.sin(q[4])

    return np.array([[0.208*((-s1*c0*c2 - s2*c0*c1)*c3 + s0*s3)*s4 + 
                      0.208*(-s1*s2*c0 + c0*c1*c2)*c4 - 0.299*s1*s2*c0 - 
                      0.3*s1*c0 + 0.299*c0*c1*c2 + 0.06*c0*c1],
                     [0.208*(-s1*s2 + c1*c2)*s4*c3 + 
                      0.208*(s1*c2 + s2*c1)*c4 + 
                      0.299*s1*c2 + 0.06*s1 + 0.299*s2*c1 + 0.3*c1 + 0.118],
                     [0.208*((s0*s1*c2 + s0*s2*c1)*c3 + s3*c0)*s4 + 
                      0.208*(s0*s1*s2 - s0*c1*c2)*c4 + 0.299*s0*s1*s2 + 
                      0.3*s0*s1 - 0.299*s0*c1*c2 - 0.06*s0*c1]], dtype='float')


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


    net = nengo.Network('q2xyz')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    net.config[nengo.Ensemble].seed = 0

    with net:
        # create / connect up q2xyz --------------------------------------------------
        if use_intrcepts:
            net.q_t = nengo.Ensemble(
                n_neurons=2000, dimensions=dim,
                radius=np.sqrt(dim),
                intercepts=get_intercepts(2000, dim),
        #    encoders = Choice([[1,1,1,1,1]]),
            )
        else:
            net.q_t = nengo.Ensemble(
                n_neurons=2000, dimensions=dim,
                radius=np.sqrt(dim),
        #        encoders = Choice([[1,1,1,1,1]]),
                
            )

        # expecting input in form [dim]
        if use_scale:
            net.input = nengo.Node(output=scale_down, size_in=dim)
        else:
            net.input = nengo.Node(size_in=dim)
        net.output = nengo.Node(size_in=axis)        
       
        
        def q2xyz_func(q):
            """ calculate the xyz according to q_t """  
            if use_scale:
                q = scale_up(q)
            t = calc_T(q)         
            return t[0], t[1], t[2]
                  
        
        nengo.Connection(net.input, net.q_t, synapse=0.05)
        nengo.Connection(net.q_t, net.output,function=q2xyz_func ,synapse=0.05)

  

    return net