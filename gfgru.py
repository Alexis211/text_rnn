import theano
from theano import tensor
import numpy

from blocks.algorithms import Momentum, AdaDelta, RMSProp, Adam
from blocks.bricks import Activation, Tanh, Logistic, Softmax, Rectifier, Linear, MLP, Initializable, Identity
from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_zeros

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, INITIAL_STATE, add_role
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

class TRectifier(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.switch(input_ > 1, input_, 0)

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 10
seq_len = 2000
seq_div_size = 100

io_dim = 256

recurrent_blocks = [
#            (256, Tanh(), [2048], [Rectifier()]),
#            (512, Rectifier(), [1024], [Rectifier()]),
            (512, Tanh(), [1024], [Rectifier()]),
            (512, Tanh(), [1024], [Rectifier()]),
#            (2, Tanh(), [2], [Rectifier()]),
#            (2, Tanh(), [], []),
        ]

control_hidden = [1024]
control_hidden_activations = [Rectifier()]

output_hidden = [1024]
output_hidden_activations = [Rectifier()]

weight_noise_std = 0.05

recurrent_h_dropout = 0
control_h_dropout = 0
output_h_dropout = 0.5

step_rule = 'adam'
learning_rate = 0.1
momentum = 0.99


param_desc = '%s,c%s,o%s-n%s-d%s,%s,%s-%s' % (
                 repr(map(lambda (a, b, c, d): (a, c), recurrent_blocks)),
                 repr(control_hidden), repr(output_hidden),
                 repr(weight_noise_std),
                 repr(recurrent_h_dropout), repr(control_h_dropout), repr(output_h_dropout),
                 step_rule
                ) 

save_freq = 5
on_irc = False

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
elif step_rule == 'adam':
    step_rule = Adam()
else:
    assert(False)




class GFGRU(BaseRecurrent, Initializable):
    def __init__(self, input_dim, recurrent_blocks, control_hidden, control_hidden_activations, **kwargs):
        super(GFGRU, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.recurrent_blocks = recurrent_blocks
        self.control_hidden = control_hidden
        self.control_hidden_activations = control_hidden_activations

        # setup children
        self.children = control_hidden_activations
        for (_, a, _, b) in recurrent_blocks:
            self.children.append(a)
            for c in b:
                self.children.append(c)

        logistic = Logistic()
        self.children.append(logistic)

        self.hidden_total_dim = sum(x for (x, _, _, _) in self.recurrent_blocks)

        # control block
        self.cblocklen = len(self.recurrent_blocks) + 2

        control_idim = self.hidden_total_dim + self.input_dim
        control_odim = len(self.recurrent_blocks) * self.cblocklen
        self.control = MLP(dims=[control_idim] + self.control_hidden + [control_odim],
                           activations=self.control_hidden_activations + [logistic],
                           name='control')

        self.children.append(self.control)

        # recurrent blocks
        self.blocks = []
        self.params = []
        for i, (dim, act, hdim, hact) in enumerate(self.recurrent_blocks):
            idim = self.input_dim + self.hidden_total_dim
            if i > 0:
                idim = idim + self.recurrent_blocks[i-1][0]

            idims = [idim] + hdim
            if hdim == []:
                inter = Identity()
            else:
                inter = MLP(dims=idims, activations=hact, name='inter%d'%i)

            rgate = MLP(dims=[idims[-1], dim], activations=[logistic], name='rgate%d'%i)
            nstate = MLP(dims=[idims[-1], dim], activations=[act], name='nstate%d'%i)

            for brick in [inter, rgate, nstate]:
                self.children.append(brick)
            self.blocks.append((inter, rgate, nstate))

        # init state zeros
        self.init_states_names = []
        self.init_states_dict = {}
        self.params = []

        for i, (dim, _, _, _) in enumerate(self.recurrent_blocks):
            name = 'init_state_%d'%i
            svar = shared_floatx_zeros((dim,), name=name)
            add_role(svar, INITIAL_STATE)

            self.init_states_names.append(name)
            self.init_states_dict[name] = svar
            self.params.append(svar)

    def get_dim(self, name):
        if name in self.init_states_dict:
            return self.init_states_dict[name].shape.eval()
        return super(GFGRU, self).get_dim(name)

    def recurrent_h_dropout_vars(self, cg):
        ret = []
        for (inter, rgate, nstate) in self.blocks:
            ret = ret + VariableFilter(name='input_',
                                       bricks=inter.linear_transformations + rgate.linear_transformations + nstate.linear_transformations
                                      )(cg)
        return ret

    def control_h_dropout_vars(self, cg):
        return VariableFilter(name='input_', bricks=self.control.linear_transformations)(cg)

    @recurrent(sequences=['inputs'], contexts=[])
    def apply(self, inputs=None, **kwargs):
        states = [kwargs[i] for i in self.init_states_names]
        concat_states = tensor.concatenate(states, axis=1)

        concat_input_states = tensor.concatenate([inputs, concat_states], axis=1)

        control_v = self.control.apply(concat_input_states)

        new_states = []
        for i, (inter, rgate, nstate) in enumerate(self.blocks):
            controls = control_v[:, i * self.cblocklen:(i+1) * self.cblocklen]
            r_inputs = tensor.concatenate([s * controls[:, j][:, None] for j, s in enumerate(states)], axis=1)

            more_inputs = [inputs]
            if i > 0:
                more_inputs.append(new_states[-1])
            inter_inputs = tensor.concatenate([r_inputs] + more_inputs, axis=1)

            inter_v = inter.apply(inter_inputs)

            rgate_v = rgate.apply(inter_v)
            nstate_v = nstate.apply(inter_v)

            rctl = controls[:, -1][:, None] * rgate_v
            uctl = controls[:, -2][:, None]
            nstate_v = uctl * nstate_v + (1 - rctl) * states[i]

            new_states.append(nstate_v)

        return new_states

    @apply.property('states')
    def apply_states(self):
        return self.init_states_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.init_states_names

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return tensor.repeat(self.init_states_dict[state_name][None, :],
                             repeats=batch_size,
                             axis=0)
            


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

        prev_states_dict = {}
        for i, (dim, _, _, _) in enumerate(recurrent_blocks):
            prev_state = theano.shared(numpy.zeros((num_seqs, dim)).astype(theano.config.floatX),
                                    name='states_save')
            prev_states_dict['init_state_%d'%i] = prev_state

        states = [x.dimshuffle(1, 0, 2) for x in gfgru.apply(in_onehot.dimshuffle(1, 0, 2), **prev_states_dict)]

        self.states = []
        for i, _ in enumerate(recurrent_blocks):
            self.states.append((prev_states_dict['init_state_%d'%i], states[i][:, -1, :]))

        states_concat = tensor.concatenate(states, axis=2)

        out_mlp = MLP(dims=[hidden_total_dim] + output_hidden + [io_dim],
                      activations=output_hidden_activations + [None],
                      name='output_mlp')
        states_sh = states_concat.reshape((inp.shape[0]*inp.shape[1], hidden_total_dim))
        out = out_mlp.apply(states_sh).reshape((inp.shape[0], inp.shape[1], io_dim))



        # Do prediction and calculate cost
        pred = out.argmax(axis=2)

        cost = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           io_dim)))
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).mean()

        # Initialize all bricks
        for brick in [gfgru, out_mlp]:
            brick.weights_init = IsotropicGaussian(0.01)
            brick.biases_init = Constant(0.001)
            brick.initialize()

        # Apply noise and dropout
        cg = ComputationGraph([cost, error_rate])
        if weight_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, weight_noise_std)
        if recurrent_h_dropout > 0:
            dv = gfgru.recurrent_h_dropout_vars(cg)
            print "Recurrent H dropout on", len(dv), "vars"
            cg = apply_dropout(cg, dv, recurrent_h_dropout)
        if control_h_dropout > 0:
            dv = gfgru.control_h_dropout_vars(cg)
            print "Control H dropout on", len(dv), "vars"
            cg = apply_dropout(cg, dv, control_h_dropout)
        if output_h_dropout > 0:
            dv = VariableFilter(name='input_', bricks=out_mlp.linear_transformations)(cg)
            print "Output H dropout on", len(dv), "vars"
            cg = apply_dropout(cg, dv, output_h_dropout)
        [cost_reg, error_rate_reg] = cg.outputs


        self.cost = cost
        self.error_rate = error_rate
        self.cost_reg = cost_reg
        self.error_rate_reg = error_rate_reg
        self.out = out
        self.pred = pred


