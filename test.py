import os
import torch
import torchvision
import numpy as np

from tqdm import tqdm
from operator import itemgetter
from torchvision import transforms
from config import load_test_arguments
from torch.utils.data import DataLoader
from models.wideresnet import WideResNet
from sklearn.metrics import accuracy_score
from datasets.data_loaders import load_test_data
from datasets.data_transformations import validation_transformations


def test_step(model, batch, metrics, device):
    model.eval()

    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(inputs)
        predictions = torch.max(logits, axis=1)[1]

    metrics['test_steps'] += 1
    metrics['test_accuracy'] += accuracy_score(labels.detach().cpu(), predictions.detach().cpu())


def on_test_batch_end(batch_step, batches, metrics, test_progress_bar):
    test_accuracy = metrics['test_accuracy'] / metrics['test_steps']

    test_progress_bar.set_description(
        'Batch:{}/{} | test_error {:.3f}'.format(
            batch_step, batches, (1 - test_accuracy) * 100)
    )


def log_checkpoint(checkpoint):
    keys_to_log = ['labeled_data', 'k', 't']
    keys_values = dict(zip(keys_to_log, itemgetter(*keys_to_log)(checkpoint)))
    keys_values['mix_up'] = 'all' if 'mix_up' not in checkpoint.keys() else checkpoint['mix_up']
    keys_values['model'] = 'wideresnet' if 'model' not in checkpoint.keys() else checkpoint['model']

    print(f'\n Evaluating checkpoint: {keys_values}')


def test():
    device = torch.device('cpu')
    arguments = load_test_arguments()
    checkpoints = {}

    if arguments.checkpoint_file:
        checkpoints[arguments.checkpoint_file] = arguments.checkpoint_file
    else:
        for root, dirs, files in os.walk(arguments.checkpoint_dir):
            for file in files:
                if file.endswith('.bin'):
                    checkpoints[file] = os.path.join(root, file)

    test_dataloader = load_test_data(arguments)

    for checkpoint_name in checkpoints:

        checkpoint = torch.load(checkpoints[checkpoint_name], map_location=device)

        log_checkpoint(checkpoint)

        ema_model = WideResNet(num_classes=10)
        ema_model.load_state_dict(checkpoint['ema_model_state'])

        batches = len(test_dataloader)

        metrics = {
            'test_steps': 0,
            'test_accuracy': 0
        }

        test_progress_bar = tqdm(enumerate(test_dataloader))
        for batch_step, batch in test_progress_bar:
            test_step(ema_model, batch, metrics, device)
            on_test_batch_end(batch_step, batches, metrics, test_progress_bar)


if __name__ == '__main__':
    test()
