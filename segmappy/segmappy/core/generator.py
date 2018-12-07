from __future__ import print_function
import numpy as np

from ..tools.classifiertools import to_onehot


class Generator(object):
    def __init__(
        self,
        preprocessor,
        segment_ids,
        n_classes,
        train=True,
        batch_size=16,
        shuffle=False,
    ):
        self.preprocessor = preprocessor
        self.segment_ids = segment_ids
        self.n_classes = n_classes
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_segments = len(self.segment_ids)
        self.n_batches = int(np.ceil(float(self.n_segments) / batch_size))

        self._i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.shuffle and self._i == 0:
            np.random.shuffle(self.segment_ids)

        self.batch_ids = self.segment_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_segments:
            self._i = 0

        batch_segments, batch_classes = self.preprocessor.get_processed(
            self.batch_ids, train=self.train
        )

        batch_segments = batch_segments[:, :, :, :, None]
        batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_segments, batch_classes

class MyGenerator(object):
    def __init__(
        self,
        preprocessor,
        segment_ids,
        n_classes,
        train=True,
        batch_size=16,
        shuffle=False,
    ):
        self.preprocessor = preprocessor
        self.segment_ids = segment_ids
        self.n_classes = n_classes
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_segments = len(self.segment_ids)
        self.n_batches = int(np.ceil(float(self.n_segments) / batch_size))

        self._i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.shuffle and self._i == 0:
            np.random.shuffle(self.segment_ids)

        self.batch_ids = self.segment_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_segments:
            self._i = 0

        batch_segments, batch_classes, batch_labels = self.preprocessor.get_processed_with_labels(
            self.batch_ids, train=self.train
        )

        batch_segments = batch_segments[:, :, :, :, None]
        batch_labels = batch_labels + 1 # index start from 1
        batch_labels = batch_labels[:,None]
        batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_segments, batch_classes, batch_labels
    
    def next_label_onehot(self):
        if self.shuffle and self._i == 0:
            np.random.shuffle(self.segment_ids)

        self.batch_ids = self.segment_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_segments:
            self._i = 0

        batch_segments, batch_classes, batch_labels = self.preprocessor.get_processed_with_labels(
            self.batch_ids, train=self.train
        )

        batch_segments = batch_segments[:, :, :, :, None]
        batch_labels = to_onehot(batch_labels, 3)
        batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_segments, batch_classes, batch_labels


class GeneratorFeatures(object):
    def __init__(self, features, classes, n_classes=2, batch_size=16, shuffle=True):
        self.features = features
        self.classes = np.asarray(classes)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = features.shape[0]
        self.n_batches = int(np.ceil(float(self.n_samples) / batch_size))
        self._i = 0

        self.sample_ids = list(range(self.n_samples))
        if shuffle:
            np.random.shuffle(self.sample_ids)

    def next(self):
        batch_ids = self.sample_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_samples:
            self._i = 0

        batch_features = self.features[batch_ids, :]
        batch_classes = self.classes[batch_ids]
        batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_features, batch_classes

class MyGeneratorFeatures(object):
    def __init__(self, preprocessor, features, classes, segments, n_classes=2, train=True, batch_size=16, shuffle=True):
        self.preprocessor = preprocessor
        self.features = features
        self.classes = np.asarray(classes)
        self.segments = np.asarray(segments)
        self.n_classes = n_classes
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = features.shape[0]
        self.n_batches = int(np.ceil(float(self.n_samples) / batch_size))
        self._i = 0

        self.sample_ids = list(range(self.n_samples))
        if shuffle:
            np.random.shuffle(self.sample_ids)

    def next(self):
        self.batch_ids = self.sample_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_samples:
            self._i = 0

        batch_features = self.features[self.batch_ids, :]
        batch_semantics = self.classes[self.batch_ids]
        batch_semantics = to_onehot(batch_semantics, self.n_classes)

        batch_segments = self.segments[self.batch_ids]
        batch_segments = self.preprocessor.process(batch_segments, batch_semantics, train=True, normalize=True)
        batch_segments = np.expand_dims(batch_segments, axis=-1)

        return batch_features, batch_semantics, batch_segments


