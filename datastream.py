import logging
import random
import numpy

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme
from fuel.transformers import Transformer

import sys
import os

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class BinaryFileDataset(Dataset):
    def __init__(self, filename, **kwargs):
        self.provides_sources= ('bytes',)

        self.f = open(filename, "rb")

        super(BinaryFileDataset, self).__init__(**kwargs)

    def get_data(self, state=None, request=None):
        if request is None:
            raise ValueError("Expected a request: begin, length")

        bg, ln = request
        self.f.seek(bg)
        return (self.f.read(ln),)

    def num_examples(self):
        return os.fstat(self.f.fileno()).st_size

class RandomBlockIterator(IterationScheme):
    requests_examples=True
    def __init__(self, item_range, seq_len, num_seqs_per_epoch, **kwargs):
        self.seq_len = seq_len
        self.num_seqs = num_seqs_per_epoch
        self.item_range = item_range

        super(RandomBlockIterator, self).__init__(**kwargs)

    def get_request_iterator(self):
        l = [(random.randrange(0, self.item_range - self.seq_len + 1), self.seq_len)
             for _ in xrange(self.num_seqs)]
        return iter_(l)

class BytesToIndices(Transformer):
    def __init__(self, stream, **kwargs):
        self.sources = ('bytes',)
        super(BytesToIndices, self).__init__(stream, **kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError('Unsupported: request')
        data = next(self.child_epoch_iterator)
        return numpy.array([ord(i) for i in data[0]], dtype='int16'),

class ParallelSequences(Transformer):
    def __init__(self, stream, num_seqs, seq_div_size, **kwargs):
        self.sources = ('bytes',)

        self.num_seqs = num_seqs
        self.div_size = seq_div_size

        self.tmp = None
        self.i = 0

        super(ParallelSequences, self).__init__(stream, **kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError('Unsupported: request')

        if self.tmp is None or self.i >= self.tmp.shape[1]:
            self.tmp = numpy.concatenate([next(self.child_epoch_iterator)[0][None, :]
                                                         for _ in xrange(self.num_seqs)],
                                         axis=0)
            self.i = 0

        ret = self.tmp[:, self.i:self.i + self.div_size]
        self.i += self.div_size

        return ret,

            

def setup_datastream(filename, num_seqs, seq_len, seq_div_size):
    ds = BinaryFileDataset(filename)
    it = RandomBlockIterator(ds.num_examples(), seq_len, num_seqs)
    stream = DataStream(ds, iteration_scheme=it)
    stream = BytesToIndices(stream)
    stream = ParallelSequences(stream, num_seqs, seq_div_size)

    return stream

if __name__ == "__main__":
    # Test
    stream = setup_datastream("data/logcompil.txt", 2, 60, 20)
    it = stream.get_epoch_iterator()
    for d, in stream.get_epoch_iterator():
        print '--'
        for u in range(d.shape[0]):
            print ''.join(chr(i) for i in d[u])

