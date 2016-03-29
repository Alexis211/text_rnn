#!/usr/bin/env python2

import logging
import sys
import importlib

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

import theano

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring

from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

try:
    from blocks.extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    logger.warning('Plotting extension not available')


import datastream
from paramsaveload import SaveLoadParams
from gentext import GenText


sys.setrecursionlimit(500000)


class ResetStates(SimpleExtension):
    def __init__(self, state_vars, **kwargs):
        super(ResetStates, self).__init__(**kwargs)

        self.f = theano.function(
            inputs=[], outputs=[],
            updates=[(v, v.zeros_like()) for v in state_vars])

    def do(self, which_callback, *args):
        self.f()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s [options] config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[-1]
    config = importlib.import_module('.%s' % model_name, 'config')

    # Build datastream
    train_stream = datastream.setup_datastream(config.dataset,
                                               config.num_seqs,
                                               config.seq_len,
                                               config.seq_div_size)

    # Build model
    m = config.Model(config)

    # Train the model
    cg = Model(m.sgd_cost)
    algorithm = GradientDescent(cost=m.sgd_cost,
                                step_rule=config.step_rule,
                                parameters=cg.parameters)

    algorithm.add_updates(m.states)

    monitor_vars = list(set(v for p in m.monitor_vars for v in p))
    extensions = [
            ProgressBar(),
            TrainingDataMonitoring(
                monitor_vars,
                prefix='train', every_n_batches=config.monitor_freq),
            Printing(every_n_batches=config.monitor_freq, after_epoch=False),

            ResetStates([v for v, _ in m.states], after_epoch=True)
    ]
    if plot_avail:
        plot_channels = [['train_' + v.name for v in p] for p in m.monitor_vars]
        extensions.append(
            Plot(document='text_'+model_name,
                 channels=plot_channels,
                 # server_url='http://localhost:5006',
                 every_n_batches=config.monitor_freq)
        )

    if config.save_freq is not None and not '--nosave' in sys.argv:
        extensions.append(
            SaveLoadParams(path='params/%s.pkl'%model_name,
                           model=cg,
                           before_training=(not '--noload' in sys.argv),
                           after_training=True,
                           every_n_batches=config.save_freq)
        )

    if config.sample_freq is not None:
        extensions.append(
            GenText(m, config.sample_init,
                    config.sample_len, config.sample_temperature,
                    before_training=True,
                    every_n_batches=config.sample_freq)
        )

    main_loop = MainLoop(
        model=cg,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions
    )
    main_loop.run()
    main_loop.profile.report()



#  vim: set sts=4 ts=4 sw=4 tw=0 et :
