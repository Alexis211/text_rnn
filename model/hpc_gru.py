# HPC-GRU : Hierarchical Predictive Coding GRU

import theano
from theano import tensor
import numpy

from blocks.bricks import Softmax, Tanh, Logistic, Linear, MLP, Identity
from blocks.bricks.recurrent import GatedRecurrent

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise


class Model():
    def __init__(self, config):
        inp = tensor.imatrix('bytes')

        embed = theano.shared(config.embedding_matrix.astype(theano.config.floatX),
                              name='embedding_matrix')
        in_repr = embed[inp.flatten(), :].reshape((inp.shape[0], inp.shape[1], config.repr_dim))
        in_repr.name = 'in_repr'

        bricks = []
        states = []

        # Construct predictive GRU hierarchy
        hidden = []
        costs = []
        next_target = in_repr.dimshuffle(1, 0, 2)
        for i, (hdim, cf, q) in enumerate(zip(config.hidden_dims,
                                                   config.cost_factors,
                                                   config.hidden_q)):
            init_state = theano.shared(numpy.zeros((config.num_seqs, hdim)).astype(theano.config.floatX),
                                       name='st0_%d'%i)

            linear = Linear(input_dim=config.repr_dim, output_dim=3*hdim,
                            name="lstm_in_%d"%i)
            lstm = GatedRecurrent(dim=hdim, activation=config.activation_function,
                        name="lstm_rec_%d"%i)
            linear2 = Linear(input_dim=hdim, output_dim=config.repr_dim, name='lstm_out_%d'%i)
            tanh = Tanh('lstm_out_tanh_%d'%i)
            bricks += [linear, lstm, linear2, tanh]
            if i > 0:
                linear1 = Linear(input_dim=config.hidden_dims[i-1], output_dim=3*hdim,
                                 name='lstm_in2_%d'%i)
                bricks += [linear1]

            next_target = tensor.cast(next_target, dtype=theano.config.floatX)
            inter = linear.apply(theano.gradient.disconnected_grad(next_target))
            if i > 0:
                inter += linear1.apply(theano.gradient.disconnected_grad(hidden[-1][:-1,:,:]))
            new_hidden = lstm.apply(inputs=inter[:,:,:hdim],
                                    gate_inputs=inter[:,:,hdim:],
                                    states=init_state)
            states.append((init_state, new_hidden[-1, :, :]))

            hidden += [tensor.concatenate([init_state[None,:,:], new_hidden],axis=0)]
            pred = tanh.apply(linear2.apply(hidden[-1][:-1,:,:]))
            costs += [numpy.float32(cf) * (-next_target * pred).sum(axis=2).mean()]
            costs += [numpy.float32(cf) * q * abs(pred).sum(axis=2).mean()]
            diff = next_target - pred
            next_target = tensor.ge(diff, 0.5) - tensor.le(diff, -0.5)


        # Construct output from hidden states
        hidden = [s.dimshuffle(1, 0, 2) for s in hidden]

        out_parts = []
        out_dims = config.out_hidden + [config.io_dim]
        for i, (dim, state) in enumerate(zip(config.hidden_dims, hidden)):
            pred_linear = Linear(input_dim=dim, output_dim=out_dims[0],
                                name='pred_linear_%d'%i)
            bricks.append(pred_linear)
            lin = theano.gradient.disconnected_grad(state)
            out_parts.append(pred_linear.apply(lin))

        # Do prediction and calculate cost
        out = sum(out_parts)

        if len(out_dims) > 1:
            out = config.out_hidden_act[0](name='out_act0').apply(out)
            mlp = MLP(dims=out_dims,
                      activations=[x(name='out_act%d'%i) for i, x in enumerate(config.out_hidden_act[1:])]
                                 +[Identity()],
                      name='out_mlp')
            bricks.append(mlp)
            out = mlp.apply(out.reshape((inp.shape[0]*(inp.shape[1]+1),-1))
                           ).reshape((inp.shape[0],inp.shape[1]+1,-1))

        pred = out.argmax(axis=2)

        cost = Softmax().categorical_cross_entropy(inp.flatten(),
                                                   out[:,:-1,:].reshape((inp.shape[0]*inp.shape[1],
                                                                config.io_dim))).mean()
        error_rate = tensor.neq(inp.flatten(), pred[:,:-1].flatten()).mean()

        sgd_cost = cost + sum(costs)
            
        # Initialize all bricks
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

        # apply noise
        cg = ComputationGraph([sgd_cost, cost, error_rate]+costs)
        if config.weight_noise > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, config.weight_noise)
        sgd_cost = cg.outputs[0]
        cost = cg.outputs[1]
        error_rate = cg.outputs[2]
        costs = cg.outputs[3:]


        # put stuff into self that is usefull for training or extensions
        self.sgd_cost = sgd_cost

        sgd_cost.name = 'sgd_cost'
        for i in range(len(costs)):
            costs[i].name = 'pred_cost_%d'%i
        cost.name = 'cost'
        error_rate.name = 'error_rate'
        self.monitor_vars = [costs, [cost],
                             [error_rate]]

        self.out = out[:,1:,:]
        self.pred = pred[:,1:]

        self.states = states


# vim: set sts=4 ts=4 sw=4 tw=0 et :
