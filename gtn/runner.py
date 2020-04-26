# -*- coding: utf-8 -*-
"""
The main module of the `Guess The Number SNN classifier`.


@author: kalivoda
"""

import argparse
import logging
import logging.config
import os
import time

import numpy as np
from numpy.random import seed
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, ShuffleSplit
from tensorflow.random import set_seed

from ann_model import ANN
from snntoolbox.bin.run import main as convert
from snntoolbox.utils.utils import import_configparser


class Runner:

    def __init__(self, *, path=None, model_name=None, val_ratio=0.25,
                 test_ratio=0.25, cv_iterations=1, model=ANN, snn_tb_config=None):
        self.logger = logging.getLogger(__name__)
        logging.config.fileConfig('log.conf')
        self.logger.debug('Runner started...')

        self.path = path
        self.model_name = model_name
        self.val_ratio = val_ratio
        self.test_size = test_ratio
        self.cv_iterations = cv_iterations
        self.model = model

        if snn_tb_config is None:
            self.snn_tb_config = self.create_config()

    def load_data(self, datapath=None):
        data = loadmat(datapath)
        target = data['allTargetData']
        non_target = data['allNonTargetData']
        self.logger.info('Loaded data from file {}.'.format(datapath))
        return target, non_target

    def __discard_damaged(self, features, labels):
        threshold = 100.0
        accept_features = []
        accept_labels = []
        for i in range(features.shape[0]):
            discard = False
            # check max values of all channels
            if np.max(np.abs(features[i])) > threshold:
                discard = True
            if not discard:
                accept_features.append(features[i])
                accept_labels.append(labels[i])
        self.logger.info('Discarded {}% of samples.'.format((1 - len(accept_features) / features.shape[0]) * 100))
        features = np.array(accept_features)
        labels = np.array(accept_labels)
        return features, labels

    def preprocess(self, target, non_target):
        features = np.concatenate((target, non_target), axis=0)
        target_labels = np.tile(np.array([1, 0]), (target.shape[0], 1))
        non_target_labels = np.tile(np.array([0, 1]), (non_target.shape[0], 1))
        labels = np.concatenate(target_labels, non_target_labels, axis=0)
        (features, labels) = self.__discard_damaged(features, labels)
        features = np.expand_dims(features, -1)
        self.logger.debug('preprocessed features({}): {})'.format(features.shape, features))
        self.logger.debug('preprocessed labels({}): {})'.format(labels.shape, labels))
        return features, labels

    def run(self):
        target, non_target = self.load_data()
        features, labels = self.preprocess(target, non_target)
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=self.test_size, random_state=0, shuffle=True
        )
        # save test data for SNN Toolbox
        self.logger.debug('Saving test data...')
        np.savez_compressed('x_test', x_test)
        np.savez_compressed('y_test', y_test)

        shuffle_split = ShuffleSplit(n_splits=self.cv_iterations, test_size=self.val_ratio, random_state=0)
        for train, validation in shuffle_split.split(x_train):
            ann_model = self.model(channels=x_train.shape[1], interval=x_train.shape[2])
            val_metrics = ann_model.train(x_train[train], y_train[train], x_train[validation], y_train[validation])
            test_metrics = ann_model.evaluate(x_test, y_test)

            self.logger.info('Saving ANN model...')
            ann_model.get_model().save(self.model_name)
            np.savez_compressed('x_norm', x_train[train][::10])
            self.logger.info('Converting ANN to SNN...')
            convert(self.snn_tb_config)

    def create_config(self):
        # Create a config file for SNN Toolbox.
        configparser = import_configparser()
        config = configparser.ConfigParser()
        config['paths'] = {
            'path_wd': self.path,  # Path to model.
            'dataset_path': self.path,  # Path to dataset.
            'filename_ann': self.model_name  # Name of input model.
        }

        config['tools'] = {
            'evaluate_ann': True,  # Test ANN on dataset before conversion.
            'normalize': True  # Normalize weights for full dynamic range.
        }

        config['simulation'] = {
            'simulator': 'INI',  # Chooses execution backend of SNN toolbox.
            'duration': 50,  # Number of time steps to run each sample.
            'num_to_test': 100,  # How many test samples to run.
            'batch_size': 50,  # Batch size for simulation.
        }

        config['output'] = {
            'plot_vars': {
                'spiketrains',
                'spikerates',
                'activations',
                'correlation',
                'v_mem',
                'error_t'}
        }

        # Store config file.
        config_path = os.path.join(self.path, 'config')
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        self.logger.info('SNN-Toolbox config file written to {}'.format(config_path))
        return config_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert, simulate and evaluate a spiking `Guess The Number` classifier.'
    )
    # for reproducibility
    seed(0)
    set_seed(1)
    # set up working directory to store results
    path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '..', 'temp', str(time.time())))
    os.makedirs(path_wd)
    runner = Runner(path=path_wd)
    runner.create_config()
    runner.run()


if __name__ == '__main__':
    main()
