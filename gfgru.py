import theano
from theano import tensor
import numpy

from blocks.algorithms import Momentum, AdaDelta, RMSProp
from blocks.bricks import Tanh, Logistic, Softmax, Rectifier, Linear, MLP, Initializable, Identity
from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_zeros

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, INITIAL_STATE, add_role
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 10
seq_len = 2000
seq_div_size = 100

io_dim = 256

recurrent_blocks = [
#            (256, Tanh(), [2048], [Rectifier()]),
            (256, Tanh(), [], []),
            (256, Tanh(), [], []),
            (256, Tanh(), [512], [Rectifier()]),
            (256, Tanh(), [512], [Rectifier()]),
        ]

control_hidden = [512]
control_hidden_activations = [Tanh()]

output_hidden = [512]
output_hidden_activations = [Rectifier()]

weight_noise_std = 0.02
recurrent_dropout = 0.5
control_dropout = 0.5

step_rule = 'adadelta'
learning_rate = 0.1
momentum = 0.9


param_desc = '%s,c%s,o%s-n%s-d%s,%s-%dx%d(%d)-%s' % (
                 repr(map(lambda (a, b, c, d): (a, c), recurrent_blocks)),
                 repr(control_hidden), repr(output_hidden),
                 repr(weight_noise_std),
                 repr(recurrent_dropout), repr(control_dropout),
                 num_seqs, seq_len, seq_div_size,
                 step_rule
                ) 

save_freq = 1

# parameters for sample generation
sample_len = 100
sample_temperature = 0.7 #0.5
sample_freq = 1

if step_rule == 'rmsprop':
    step_rule = RMSProp()
elif step_rule == 'adadelta':
    step_rule = AdaDelta()
elif step_rule == 'momentum':
    step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
else:
    assert(False)


class GFGRU(BaseRecurrent, Initializable):
    @lazy(allocation=['input_dim', 'recurrent_blocks', 'control_hidden', 'control_hidden_activations'])
    def __init__(self, input_dim=None, recurrent_blocks=None, control_hidden=None, control_hidden_activations=None, **kwargs):
        super(GFGRU, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.recurrent_blocks = recurrent_blocks
        self.control_hidden = control_hidden
        self.control_hidden_activations = control_hidden_activations

        self.children = control_hidden_activations

    def _allocate(self):
        for (_, a, _, b) in recurrent_blocks:
            self.children.append(a)
            for c in b:
                self.children.append(c)

        logistic = Logistic()
        self.children.append(logistic)

        self.hidden_total_dim = sum(x for (x, _, _, _) in self.recurrent_blocks)

        control_idim = self.hidden_total_dim + self.input_dim
        control_odim = len(self.recurrent_blocks) * (len(self.recurrent_blocks) + 2)
        self.control = MLP(dims=[control_idim] + self.control_hidden + [control_odim],
                           activations=self.control_hidden_activations + [logistic],
                           name='control')

        self.children.append(self.control)

        self.blocks = []
        self.params = []
        self.initial_states = {}
        for i, (dim, act, hdim, hact) in enumerate(self.recurrent_blocks):
            idim = self.input_dim + self.hidden_total_dim
            if i > 0:
                idim = idim + self.recurrent_blocks[i-1][0]
            rgate = MLP(dims=[self.hidden_total_dim, self.hidden_total_dim],
                        activations=[logistic],
                        name='rgate%d'%i)
            idims = [idim] + hdim
            if hdim == []:
                inter = Identity()
            else:
                inter = MLP(dims=idims, activations=hact, name='inter%d'%i)
            zgate = MLP(dims=[idims[-1], dim], activations=[logistic], name='zgate%d'%i)
            nstate = MLP(dims=[idims[-1], dim], activations=[act], name='nstate%d'%i)
            for brick in [rgate, inter, zgate, nstate]:
                self.children.append(brick)
            self.blocks.append((rgate, inter, zgate, nstate))

        init_states = shared_floatx_zeros((self.hidden_total_dim,), name='initial_states')
        self.params = [init_states]
        add_role(self.params[0], INITIAL_STATE)

    def get_dim(self, name):
        if name == 'states':
            return self.hidden_total_dim
        return super(GFLSTM, self).get_dim(name)

    @recurrent(sequences=['inputs'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs=None, states=None):
        concat_states = states

        states = []
        offset = 0
        for (dim, _, _, _) in self.recurrent_blocks:
            states.append(concat_states[:, offset:offset+dim])
            offset += dim

        concat_input_states = tensor.concatenate([inputs, concat_states], axis=1)

        control = self.control.apply(concat_input_states)

        new_states = []
        for i, (rgate, inter, zgate, nstate) in enumerate(self.blocks):
            controls = control[:, i * (len(self.recurrent_blocks)+2):(i+1) * (len(self.recurrent_blocks)+2)]
            rgate_v = rgate.apply(concat_states)
            r_inputs = tensor.concatenate([s * controls[:, j][:, None] for j, s in enumerate(states)], axis=1)
            r_inputs = r_inputs * (1 - rgate_v * controls[:, -1][:, None])

            more_inputs = [inputs]
            if i > 0:
                more_inputs = more_inputs + [new_states[-1]]
            inter_inputs = tensor.concatenate([r_inputs] + more_inputs, axis=1)

            inter_v = inter.apply(inter_inputs)
            zgate_v = zgate.apply(inter_v)
            nstate_v = nstate.apply(inter_v)

            nstate_v = nstate_v * (1 - zgate_v * controls[:, -2][:, None])
            new_states.append(nstate_v)

        return tensor.concatenate(new_states, axis=1)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return tensor.repeat(self.params[0][None, :], repeats=batch_size, axis=0)
            



class Model():
    def __init__(self):
        inp = tensor.lmatrix('bytes')

        in_onehot = tensor.eq(tensor.arange(io_dim, dtype='int16').reshape((1, 1, io_dim)),
                              inp[:, :, None])
        in_onehot.name = 'in_onehot'

        gfgru = GFGRU(input_dim=io_dim,
                      recurrent_blocks=recurrent_blocks,
                      control_hidden=control_hidden,
                      control_hidden_activations=control_hidden_activations)

        hidden_total_dim = sum(x for (x, _, _, _) in recurrent_blocks)

        prev_states = theano.shared(numpy.zeros((num_seqs, hidden_total_dim)).astype(theano.config.floatX),
                                    name='states_save')
        states = gfgru.apply(in_onehot.dimshuffle(1, 0, 2),
                             states=prev_states).dimshuffle(1, 0, 2)
        new_states = states[:, -1, :]

        out_mlp = MLP(dims=[hidden_total_dim] + output_hidden + [io_dim],
                      activations=output_hidden_activations + [None],
                      name='output_mlp')
        out = out_mlp.apply(states.reshape((inp.shape[0]*inp.shape[1], hidden_total_dim))).reshape((inp.shape[0], inp.shape[1], io_dim))



        # Do prediction and calculate cost
        pred = out.argmax(axis=2)

        cost = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           io_dim)))
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).mean()

        # Initialize all bricks
        for brick in [gfgru, out_mlp]:
            brick.weights_init = IsotropicGaussian(0.1)
            brick.biases_init = Constant(0.)
            brick.initialize()

        # Apply noise and dropout
        cg = ComputationGraph([cost, error_rate])
        if weight_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, weight_noise_std)
        # if i_dropout > 0:
        #     cg = apply_dropout(cg, hidden[1:], i_dropout)
        [cost_reg, error_rate_reg] = cg.outputs


        self.cost = cost
        self.error_rate = error_rate
        self.cost_reg = cost_reg
        self.error_rate_reg = error_rate_reg
        self.out = out
        self.pred = pred

        self.states = [(prev_states, new_states)]

