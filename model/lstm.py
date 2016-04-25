import theano
from theano import tensor
import numpy

from blocks.bricks import Softmax, Linear
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout


class Model():
    def __init__(self, config):
        inp = tensor.imatrix('bytes')

        in_onehot = tensor.eq(tensor.arange(config.io_dim, dtype='int32').reshape((1, 1, config.io_dim)),
                              inp[:, :, None]).astype(theano.config.floatX)
        in_onehot.name = 'in_onehot'

        costs_xreg = []

        # Construct hidden states
        dims = [config.io_dim]
        hidden = [in_onehot.dimshuffle(1, 0, 2)]
        bricks = []
        states = []
        for i in xrange(1, len(config.layers)+1):
            p = config.layers[i-1]

            init_state = theano.shared(numpy.zeros((config.num_seqs, p['dim'])).astype(theano.config.floatX),
                                       name='st0_%d'%i)
            init_cell = theano.shared(numpy.zeros((config.num_seqs, p['dim'])).astype(theano.config.floatX),
                                       name='cell0_%d'%i)

            linear = Linear(input_dim=dims[i-1], output_dim=4*p['dim'],
                            name="lstm_in_%d"%i)
            bricks.append(linear)
            inter = linear.apply(hidden[-1])

            if config.i2h_all and i > 1:
                linear2 = Linear(input_dim=dims[0], output_dim=4*p['dim'],
                                 name="lstm_in0_%d"%i)
                bricks.append(linear2)
                inter = inter + linear2.apply(hidden[0])
                inter.name = 'inter_bis_%d'%i

            lstm = LSTM(dim=p['dim'], activation=config.activation_function,
                        name="lstm_rec_%d"%i)
            bricks.append(lstm)

            new_hidden, new_cells = lstm.apply(inter,
                                               states=init_state,
                                               cells=init_cell)
            states.append((init_state, new_hidden[-1, :, :]))
            states.append((init_cell, new_cells[-1, :, :]))

            if 'xreg' in p and p['xreg'] is not None:
                n, s, w1, w2, w3 = p['xreg']
                cost_x1 = w1 * ((new_hidden.mean(axis=2) - s)**2).mean()
                cost_x2 = w2 * ((new_hidden.mean(axis=(0,1)) - s)**2).mean()
                cost_x3 = -w3 * abs(new_hidden - s).mean()
                cost_x1.name = 'cost_x1_%d'%i
                cost_x2.name = 'cost_x2_%d'%i
                cost_x3.name = 'cost_x3_%d'%i
                costs_xreg += [cost_x1, cost_x2, cost_x3]

            dims.append(p['dim'])
            hidden.append(new_hidden)

        for i, (u, v) in enumerate(states):
            print "****     state", i, u.dtype, v.dtype

        hidden = [s.dimshuffle(1, 0, 2) for s in hidden]

        # Construct output from hidden states
        out = None
        layers = zip(dims, hidden)[1:]
        if not config.h2o_all:
            layers = [layers[-1]]
        for i, (dim, state) in enumerate(layers):
            top_linear = Linear(input_dim=dim, output_dim=config.io_dim,
                                name='top_linear_%d'%i)
            bricks.append(top_linear)
            out_i = top_linear.apply(state)
            print "****         out", i, out_i.dtype
            out = out_i if out is None else out + out_i
            out.name = 'out_part_%d'%i

        # Do prediction and calculate cost
        pred = out.argmax(axis=2).astype('int32')

        print "****         inp", inp.dtype
        print "****         out", out.dtype
        print "****         pred", pred.dtype
        cost0 = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           config.io_dim))).mean()
        cost0.name = 'cost0'
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).astype(theano.config.floatX).mean()
        print "****         cost0", cost0.dtype
        print "****         error_rate", error_rate.dtype

        costs = [cost0] + costs_xreg
        cost = sum(costs)

        # Initialize all bricks
        for brick in bricks:
            brick.weights_init = IsotropicGaussian(0.1)
            brick.biases_init = Constant(0.)
            brick.initialize()

        # Apply noise and dropout
        cg = ComputationGraph([cost, error_rate] + costs)
        if config.w_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, config.w_noise_std)
        if config.i_dropout > 0:
            cg = apply_dropout(cg, hidden[1:], config.i_dropout)
        [cost_reg, error_rate_reg] = cg.outputs[:2]
        costs_reg = cg.outputs[2:]
        print "****         cost_reg", cost_reg.dtype
        print "****         error_rate_reg", error_rate_reg.dtype

        # add l1 regularization
        if config.l1_reg > 0:
            l1pen = sum(abs(st).mean() for st in hidden[1:])
            cost_reg = cost_reg + config.l1_reg * l1pen

        cost_reg += 1e-10           # so that it is not the same Theano variable as cost
        error_rate_reg += 1e-10

        # put stuff into self that is usefull for training or extensions
        self.sgd_cost = cost_reg

        cost.name = 'cost'
        cost_reg.name = 'cost_reg'
        error_rate.name = 'error_rate'
        error_rate_reg.name = 'error_rate_reg'
        self.monitor_vars = [[cost_reg],
                             costs_reg,
                             [error_rate_reg]]

        self.out = out
        self.pred = pred

        self.states = states


#  vim: set sts=4 ts=4 sw=4 tw=0 et :
