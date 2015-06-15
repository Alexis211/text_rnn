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
num_seqs = 20
seq_len = 5000
seq_div_size = 200

io_dim = 256

hidden_dims = [1024, 1024, 1024]
activation_function = Tanh()

i2h_all = True             # input to all hidden layers or only first layer
h2o_all = True             # all hiden layers to output or only last layer

w_noise_std = 0.02
i_dropout = 0.5

l1_reg = 0

step_rule = 'adadelta'
learning_rate = 0.1
momentum = 0.9


param_desc = '%s-%sIH,%sHO-n%s-d%s-l1r%s-%dx%d(%d)-%s' % (
                 repr(hidden_dims),
                 'all' if i2h_all else 'first',
                 'all' if h2o_all else 'last',
                 repr(w_noise_std),
                 repr(i_dropout),
                 repr(l1_reg),
                 num_seqs, seq_len, seq_div_size,
                 step_rule
                ) 

save_freq = 5

# parameters for sample generation
sample_len = 1000
sample_temperature = 0.7 #0.5
sample_freq = 10

if step_rule == 'rmsprop':
    step_rule = RMSProp()
elif step_rule == 'adadelta':
    step_rule = AdaDelta()
elif step_rule == 'momentum':
    step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
else:
    assert(False)

class Model():
    def __init__(self):
        inp = tensor.lmatrix('bytes')

        in_onehot = tensor.eq(tensor.arange(io_dim, dtype='int16').reshape((1, 1, io_dim)),
                              inp[:, :, None])
        in_onehot.name = 'in_onehot'

        # Construct hidden states
        dims = [io_dim] + hidden_dims
        hidden = [in_onehot.dimshuffle(1, 0, 2)]
        bricks = []
        states = []
        for i in xrange(1, len(dims)):
            init_state = theano.shared(numpy.zeros((num_seqs, dims[i])).astype(theano.config.floatX),
                                       name='st0_%d'%i)
            init_cell = theano.shared(numpy.zeros((num_seqs, dims[i])).astype(theano.config.floatX),
                                       name='cell0_%d'%i)

            linear = Linear(input_dim=dims[i-1], output_dim=4*dims[i],
                            name="lstm_in_%d"%i)
            bricks.append(linear)
            inter = linear.apply(hidden[-1])

            if i2h_all and i > 1:
                linear2 = Linear(input_dim=dims[0], output_dim=4*dims[i],
                                 name="lstm_in0_%d"%i)
                bricks.append(linear2)
                inter = inter + linear2.apply(hidden[0])
                inter.name = 'inter_bis_%d'%i

            lstm = LSTM(dim=dims[i], activation=activation_function,
                        name="lstm_rec_%d"%i)
            bricks.append(lstm)

            new_hidden, new_cells = lstm.apply(inter,
                                               states=init_state,
                                               cells=init_cell)
            states.append((init_state, new_hidden[-1, :, :]))
            states.append((init_cell, new_cells[-1, :, :]))

            hidden.append(new_hidden)

        hidden = [s.dimshuffle(1, 0, 2) for s in hidden]

        # Construct output from hidden states
        out = None
        layers = zip(dims, hidden)[1:]
        if not h2o_all:
            layers = [layers[-1]]
        for i, (dim, state) in enumerate(layers):
            top_linear = Linear(input_dim=dim, output_dim=io_dim,
                                name='top_linear_%d'%i)
            bricks.append(top_linear)
            out_i = top_linear.apply(state)
            out = out_i if out is None else out + out_i
            out.name = 'out_part_%d'%i

        # Do prediction and calculate cost
        pred = out.argmax(axis=2)

        cost = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           io_dim)))
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).mean()

        # Initialize all bricks
        for brick in bricks:
            brick.weights_init = IsotropicGaussian(0.1)
            brick.biases_init = Constant(0.)
            brick.initialize()

        # Apply noise and dropout
        cg = ComputationGraph([cost, error_rate])
        if w_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, w_noise_std)
        if i_dropout > 0:
            cg = apply_dropout(cg, hidden[1:], i_dropout)
        [cost_reg, error_rate_reg] = cg.outputs

        # add l1 regularization
        if l1_reg > 0:
            l1pen = sum(abs(st).mean() for st in hidden[1:])
            cost_reg = cost_reg + l1_reg * l1pen

        self.cost = cost
        self.error_rate = error_rate
        self.cost_reg = cost_reg
        self.error_rate_reg = error_rate_reg
        self.out = out
        self.pred = pred

        self.states = states

