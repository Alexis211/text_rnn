import theano
from theano import tensor
import numpy

from blocks.algorithms import Momentum, AdaDelta, RMSProp
from blocks.bricks import Tanh, Softmax, Linear, MLP
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 10
seq_len = 2000
seq_div_size = 100

io_dim = 256

hidden_dims = [512, 512]
activation_function = Tanh()

all_hidden_for_output = False

w_noise_std = 0.01
i_dropout = 0.5

step_rule = 'adadelta'


param_desc = '%s-%sHO-n%s-d%s-%dx%d(%d)-%s' % (
                 repr(hidden_dims),
                 'all' if all_hidden_for_output else 'last',
                 repr(w_noise_std),
                 repr(i_dropout),
                 num_seqs, seq_len, seq_div_size,
                 step_rule
                ) 

if step_rule == 'rmsprop':
    step_rule = RMSProp()
elif step_rule == 'adadelta':
    step_rule = AdaDelta()
else:
    assert(False)

class Model():
    def __init__(self):
        inp = tensor.lmatrix('bytes')

        in_onehot = tensor.eq(tensor.arange(io_dim, dtype='int16').reshape((1, 1, io_dim)),
                              inp[:, :, None])

        dims = [io_dim] + hidden_dims
        states = [in_onehot.dimshuffle(1, 0, 2)]
        bricks = []
        updates = []
        for i in xrange(1, len(dims)):
            init_state = theano.shared(numpy.zeros((num_seqs, dims[i])).astype(theano.config.floatX),
                                       name='st0_%d'%i)
            init_cell = theano.shared(numpy.zeros((num_seqs, dims[i])).astype(theano.config.floatX),
                                       name='cell0_%d'%i)

            linear = Linear(input_dim=dims[i-1], output_dim=4*dims[i],
                            name="lstm_in_%d"%i)
            lstm = LSTM(dim=dims[i], activation=activation_function,
                        name="lstm_rec_%d"%i)

            new_states, new_cells = lstm.apply(linear.apply(states[-1]),
                                               states=init_state,
                                               cells=init_cell)
            updates.append((init_state, new_states[-1, :, :]))
            updates.append((init_cell, new_cells[-1, :, :]))

            states.append(new_states)
            bricks = bricks + [linear, lstm]

        states = [s.dimshuffle(1, 0, 2).reshape((inp.shape[0] * inp.shape[1], dim))
                        for dim, s in zip(dims, states)]

        if all_hidden_for_output:
            top_linear = MLP(dims=[sum(hidden_dims), io_dim],
                             activations=[Softmax()],
                             name="pred_mlp")
            bricks.append(top_linear)

            out = top_linear.apply(tensor.concatenate(states[1:], axis=1))
        else:
            top_linear = MLP(dims=[hidden_dims[-1], io_dim],
                             activations=[None],
                             name="pred_mlp")
            bricks.append(top_linear)

            out = top_linear.apply(states[-1])

        out = out.reshape((inp.shape[0], inp.shape[1], io_dim))

        pred = out.argmax(axis=2)

        cost = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           io_dim)))
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).mean()

        # Initialize
        for brick in bricks:
            brick.weights_init = IsotropicGaussian(0.1)
            brick.biases_init = Constant(0.)
            brick.initialize()

        # apply noise
        cg = ComputationGraph([cost, error_rate])
        if w_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, w_noise_std)
        if i_dropout > 0:
            cg = apply_dropout(cg, states[1:], i_dropout)
        [cost_reg, error_rate_reg] = cg.outputs

        self.cost = cost
        self.error_rate = error_rate
        self.cost_reg = cost_reg
        self.error_rate_reg = error_rate_reg
        self.pred = pred

        self.updates = updates

