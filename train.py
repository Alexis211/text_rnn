#!/usr/bin/env python

import logging
import numpy
import sys
import importlib

import theano
from theano import tensor

from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing, SimpleExtension
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

import datastream
# from apply_model import Apply

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('%s' % model_name)


class GenText(SimpleExtension):
    def __init__(self, model, init_text, max_bytes, **kwargs):
        self.init_text = init_text
        self.max_bytes = max_bytes

        cg = ComputationGraph([model.pred])
        assert(len(cg.inputs) == 1)
        assert(cg.inputs[0].name == 'bytes')
        self.f = theano.function(inputs=cg.inputs, outputs=[model.pred])

        super(GenText, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        v = numpy.array([ord(i) for i in self.init_text],
                        dtype='int16')[None, :].repeat(axis=0, repeats=config.num_seqs)

        while v.shape[1] < self.max_bytes:
            pred, = self.f(v)
            v = numpy.concatenate([v, pred[:, -1:]], axis=1)

        for i in range(v.shape[0]):
            print "Sample:", ''.join([chr(int(v[i, j])) for j in range(v.shape[1])])

def train_model(m, train_stream, load_location=None, save_location=None):

    # Define the model
    model = Model(m.cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(m.cost_reg)
    algorithm = GradientDescent(cost=m.cost_reg,
                                step_rule=config.step_rule,
                                params=cg.parameters)

    algorithm.add_updates(m.updates)

    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            TrainingDataMonitoring(
                [m.cost_reg, m.error_rate_reg, m.cost, m.error_rate],
                prefix='train', every_n_epochs=1),
            Printing(every_n_epochs=1, after_epoch=False),
            Plot(document='tr_'+model_name+'_'+config.param_desc,
                 channels=[['train_cost', 'train_cost_reg'],
                           ['train_error_rate', 'train_error_rate_reg']],
                 every_n_epochs=1, after_epoch=False),
            GenText(m, '\t', 20, every_n_epochs=1, after_epoch=False)
        ]
    )
    main_loop.run()

    # Save the main loop
    if save_location is not None:
        logger.info('Saving the main loop...')
        dump_manager = MainLoopDumpManager(save_location)
        dump_manager.dump(main_loop)
        logger.info('Saved')


if __name__ == "__main__":
    # Build datastream
    train_stream = datastream.setup_datastream('data/logcompil.txt',
                                               config.num_seqs,
                                               config.seq_len,
                                               config.seq_div_size)

    # Build model
    m = config.Model()
    m.cost.name = 'cost'
    m.cost_reg.name = 'cost_reg'
    m.error_rate.name = 'error_rate'
    m.error_rate_reg.name = 'error_rate_reg'
    m.pred.name = 'pred'

    # Train the model
    saveloc = 'model_data/%s' % model_name
    train_model(m, train_stream,
                load_location=None,
                save_location=None)

