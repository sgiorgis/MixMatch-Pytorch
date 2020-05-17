import os
import sys
import yaml
import torch
import pprint
import argparse


class Config(object):
    def __init__(self, **kwargs):
        self.root = kwargs['dataset'].pop('root', './data')
        self.dataset_name = kwargs['dataset'].pop('name', 'cifar10')
        self.dataset_classes = kwargs['dataset'].pop('classes', 10)

        self.epochs = kwargs['train'].pop('epochs', 1024)
        self.batch_size = kwargs['train'].pop('batch_size', 64)
        self.iterations = kwargs['train'].pop('iterations', 1024)
        self.labeled_data = kwargs['train'].pop('labeled_data', 500)
        self.resume = kwargs['train'].pop('resume', False)
        self.checkpoint_path = kwargs['train'].pop('checkpoint_path', None)
        self.ema = kwargs['train'].pop('ema', False)
        self.mix_up = kwargs['train'].pop('mix_up', 'all')
        self.model = kwargs['train'].pop('model', 'wideresnet')
        self.semi_supervised = kwargs['train'].pop('semi_supervised', 'mix_match')

        self.learning_rate = kwargs['optimizer'].pop('learning_rate', 0.002)
        self.ema_decay = kwargs['optimizer'].pop('ema_decay', 0.999)
        self.lambda_u = kwargs['optimizer'].pop('lambda_u', 100)
        self.t = kwargs['optimizer'].pop('t', 0.5)
        self.k = kwargs['optimizer'].pop('k', 2)
        self.alpha = kwargs['optimizer'].pop('alpha', 0.75)

        self.epoch_step = 0
        self.best_test_accuracy = sys.float_info.min

        self.validate()

    def validate(self):
        if self.model not in ('wideresnet', 'cnn13'):
            raise Exception(f'Model {config.model} is not supported')

        if self.mix_up not in ('all', 'labeled', 'unlabeled', 'separate', 'None'):
            raise Exception(f'Mix up mode {config.mix_up} is not supported')

        if self.semi_supervised not in ('mix_match', 'pseudo_label'):
            raise Exception(f'Semi-supervised method {config.semi_supervised} is not supported')

    def __repr__(self):
        return pprint.pformat(self.__dict__)


def fix_device(config):
    if torch.cuda.is_available():
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
        config.device = torch.device("cuda")
        return

    print('No GPU available, using the CPU instead.')
    config.device = torch.device("cpu")


def load_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./experiments/config.yml',
                        help="Path to config file, (default:./experiments/config.yml)")

    arguments = parser.parse_args()

    return arguments


def load_test_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Path to checkpoints directory, (default:./checkpoints)")

    parser.add_argument('--checkpoint_file', type=str, default=None,
                        help="Path to checkpoint file, (default:None)")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size, (default:32)")

    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help="Dataset name, (default:CIFAR10)")

    return parser.parse_args()


def load_config(path='./experiments/config.yml'):
    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config = Config(**config)
    fix_device(config)

    if config.resume:
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        config.epoch_step = checkpoint['epoch_step']
        config.learning_rate = checkpoint['learning_rate']
        config.lambda_u = checkpoint['lambda_u']
        config.alpha = checkpoint['alpha']
        config.t = checkpoint['t']
        config.k = checkpoint['k']
        config.mix_up = checkpoint['mix_up']
        config.ema = checkpoint['ema']
        config.batch_size = checkpoint['batch_size']
        config.iterations = checkpoint['iterations']
        config.labeled_data = checkpoint['labeled_data']
        config.best_validation_accuracy = checkpoint['best_test_accuracy']

    print(f'Configuration: {config}')

    return config
