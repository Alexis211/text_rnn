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

def setup_datastream(filename, seq_len, num_seqs_per_epoch=100):
    ds = BinaryFileDataset(filename)
    it = RandomBlockIterator(ds.num_examples(), seq_len, num_seqs_per_epoch)
    stream = DataStream(ds, iteration_scheme=it)
    stream = BytesToIndices(stream)

    return stream

if __name__ == "__main__":
    # Test
    stream = setup_datastream("data/logcompil.txt", 100)
    print(next(stream.get_epoch_iterator()))

