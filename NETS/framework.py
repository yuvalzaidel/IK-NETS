import nengo
from nengo.dists import Choice
from utils import *

def generate(target_xyz,net=None,  # define q2xyz, euc, error_combined, mse inside net
             probes_on=True, # set True to record data
             use_intrcepts=False,
             direct_mode=False,
            n_scale=100):  
    """ Connect up the q2xyz, euc, error_combined and mse sub-networks up for
    the Inverse Kinematics model.
    """
             
    config = nengo.Config(nengo.Connection, nengo.Ensemble)
    with net, config:
        
        current_q = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        dim = net.dim  # the number of DOF of the arm
        axis = net.axis  # the number axis
        net.probes_on = probes_on

        net.xyz_in = nengo.Node(target_xyz, label='xyz_in')
        net.xyz_t = nengo.Node(size_in=axis, label='xyz_t')
        net.xyz_diff = nengo.Node(size_in=2*axis, label='xyz_diff')
        net.error_q = nengo.Node(size_in=dim+axis, label='error_q')
        net.q_fixed = nengo.Node(size_in=dim, label='q_fixed')
        
        if use_intrcepts:           
            net.xyz_t = nengo.Ensemble(
                n_neurons=2000, dimensions=axis,
                radius=np.sqrt(axis),
                intercepts=get_intercepts(2000, axis),

            )
        else:
            if direct_mode:
                net.xyz_t = nengo.Ensemble(
                n_neurons=2000, dimensions=axis,
                radius=np.sqrt(axis),
                neuron_type = nengo.Direct(),
                )
            else:    
                net.xyz_t = nengo.Ensemble(
                    n_neurons=2000, dimensions=axis,
                    radius=np.sqrt(axis),

                    )

            
        nengo.Connection(net.xyz_in, net.xyz_t, transform=-1) 
        nengo.Connection(net.xyz_in, net.xyz_diff[0:3]) 
        nengo.Connection(net.xyz_t, net.xyz_diff[3:]) 
        nengo.Connection(net.xyz_t, net.error_q[5:])
        nengo.Connection(net.q_fixed, net.error_q[0:5])
        
        ''' q2xyz '''
        nengo.Connection(net.q_fixed, net.q2xyz.input)
        nengo.Connection(net.q2xyz.output, net.xyz_t)
        
        
        ''' error_combined '''
        nengo.Connection(net.error_q, net.error_combined.input)
        nengo.Connection(net.error_combined.output, net.q_fixed)
            
        ''' euc '''
        nengo.Connection(net.xyz_t, net.euc.input)     
        
        ''' mse '''
        nengo.Connection(net.xyz_diff, net.mse.input)

        
        
        if probes_on:
            net.probe_euc = nengo.Probe(net.euc.output,synapse=0.1)
            net.probe_mse = nengo.Probe(net.mse.output,synapse=0.05)
            net.probe_xyz_in = nengo.Probe(net.xyz_in,synapse=0.05)
            net.probe_xyz_pred = nengo.Probe(net.q2xyz.output,synapse=0.05)
            net.probe_q_fixed = nengo.Probe(net.q_fixed,synapse=0.05)
            net.probe_q_c = nengo.Probe(net.error_combined.q_c,synapse=0.05)
            net.probe_q_in = nengo.Probe(net.error_combined.q_in,synapse=0.05)
            net.probe_error_combined = nengo.Probe(net.error_combined.error_combined,synapse=0.05)
        
        
    return net