import theano
from theano import tensor
import numpy

from blocks.bricks import Softmax, Linear
from blocks.bricks.recurrent import recurrent, LSTM

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout


class CondLSTM(LSTM):
    @recurrent(sequences=['inputs', 'run_mask', 'rst_in_mask', 'rst_out_mask'],
               states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells', 'outputs'])
    def apply_cond(self, inputs, states, cells, run_mask=None, rst_in_mask=None, rst_out_mask=None):
        init_states, init_cells = self.initial_states(states.shape[0])

        if rst_in_mask:
            states = tensor.switch(rst_in_mask[:, None], init_states, states)
            cells = tensor.switch(rst_in_mask[:, None], init_cells, cells)

        states, cells = self.apply(iterate=False,
                                   inputs=inputs, states=states, cells=cells,
                                   mask=run_mask)

        outputs = states

        if rst_out_mask:
            states = tensor.switch(rst_out_mask[:, None], init_states, states)
            cells = tensor.switch(rst_out_mask[:, None], init_cells, cells)

        return states, cells, outputs
            

def compare_matrix(inp, chars):
    chars = numpy.array(map(ord, chars), dtype='int8')
    assert(inp.ndim == 2)
    return tensor.eq(inp[:, :, None], chars[None, None, :]).sum(axis=2).astype(theano.config.floatX)

class Model():
    def __init__(self, config):
        inp = tensor.imatrix('bytes')

        in_onehot = tensor.eq(tensor.arange(config.io_dim, dtype='int32').reshape((1, 1, config.io_dim)),
                              inp[:, :, None]).astype(theano.config.floatX)
        in_onehot.name = 'in_onehot'

        hidden_dim = sum(p['dim'] for p in config.layers)
        recvalues = tensor.concatenate([in_onehot.dimshuffle(1, 0, 2),
                            tensor.zeros((inp.shape[1], inp.shape[0], hidden_dim))],
                        axis=2)
  
        # Construct hidden states
        indim = config.io_dim
        bricks = []
        states = []
        for i in xrange(1, len(config.layers)+1):
            p = config.layers[i-1]

            init_state = theano.shared(numpy.zeros((config.num_seqs, p['dim'])).astype(theano.config.floatX),
                                       name='st0_%d'%i)
            init_cell = theano.shared(numpy.zeros((config.num_seqs, p['dim'])).astype(theano.config.floatX),
                                       name='cell0_%d'%i)

            linear = Linear(input_dim=indim, output_dim=4*p['dim'],
                            name="lstm_in_%d"%i)
            bricks.append(linear)
            inter = linear.apply(recvalues[:, :, :indim])

            lstm = CondLSTM(dim=p['dim'], activation=config.activation_function,
                        name="lstm_rec_%d"%i)
            bricks.append(lstm)

            run_mask = None
            if 'run_on' in p:
                run_mask = compare_matrix(inp.T, p['run_on'])

            rst_in_mask = None
            if 'reset_before' in p:
                rst_in_mask = compare_matrix(inp.T, p['reset_before'])

            rst_out_mask = None
            if 'reset_after' in p:
                rst_out_mask = compare_matrix(inp.T, p['reset_after'])

            new_hidden, new_cells, rec_out = \
                        lstm.apply_cond(inputs=inter,
                                        states=init_state, cells=init_cell,
                                        run_mask=run_mask,
                                        rst_in_mask=rst_in_mask, rst_out_mask=rst_out_mask)
            states.append((init_state, new_hidden[-1, :, :]))
            states.append((init_cell, new_cells[-1, :, :]))

            indim2 = indim + p['dim']
            recvalues = tensor.set_subtensor(recvalues[:, :, indim:indim2],
                                             rec_out)
            indim = indim2


        print "**** recvalues", recvalues.dtype
        for i, (u, v) in enumerate(states):
            print "****     state", i, u.dtype, v.dtype

        recvalues = recvalues.dimshuffle(1, 0, 2)

        # Construct output from hidden states
        top_linear = Linear(input_dim=indim, output_dim=config.io_dim,
                            name="top_linear")
        bricks.append(top_linear)
        out = top_linear.apply(recvalues)
        out.name = 'out'

        # Do prediction and calculate cost
        pred = out.argmax(axis=2).astype('int32')

        print "****         inp", inp.dtype
        print "****         out", out.dtype
        print "****         pred", pred.dtype
        cost = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           config.io_dim))).mean()
        cost.name = 'cost'
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).astype(theano.config.floatX).mean()
        print "****         cost", cost.dtype
        print "****         error_rate", error_rate.dtype

        # Initialize all bricks
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

        # Apply noise and dropout
        cg = ComputationGraph([cost, error_rate])
        if config.w_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, config.w_noise_std)
        if config.i_dropout > 0:
            cg = apply_dropout(cg, hidden[1:], config.i_dropout)
        [cost_reg, error_rate_reg] = cg.outputs
        print "****         cost_reg", cost_reg.dtype
        print "****         error_rate_reg", error_rate_reg.dtype

        # add l1 regularization
        if config.l1_reg > 0:
            l1pen = sum(abs(st).mean() for st in hidden[1:])
            cost_reg = cost_reg + config.l1_reg * l1pen
        if config.l1_reg_weight > 0:
            l1pen_w = sum(abs(w).mean() for w in VariableFilter(roles=[WEIGHT])(cg))
            cost_reg = cost_reg + config.l1_reg_weight * l1pen_w

        cost_reg += 1e-10           # so that it is not the same Theano variable as cost
        error_rate_reg += 1e-10

        # put stuff into self that is usefull for training or extensions
        self.sgd_cost = cost_reg

        cost.name = 'cost'
        cost_reg.name = 'cost_reg'
        error_rate.name = 'error_rate'
        error_rate_reg.name = 'error_rate_reg'
        self.monitor_vars = [[cost],
                             [cost_reg],
                             [error_rate_reg]]

        self.out = out
        self.pred = pred

        self.states = states


#  vim: set sts=4 ts=4 sw=4 tw=0 et :
