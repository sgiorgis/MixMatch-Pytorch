import os
import time
import torch
import shutil
import random
import argparse
import torchvision

import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.optim import Adam
from models.cnn13 import CNN13
from pseudo_label_loss import pseudo_label_loss
from torch.utils.data import Dataset
from models.weight_ema import WeightEMA
from models.wideresnet import WideResNet
from sklearn.metrics import accuracy_score
from config import load_arguments, load_config
from semi_supervised.pseudo_label import PseudoLabel
from datasets.data_loaders import load_train_data

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    arguments = load_arguments()
    config = load_config(arguments.config)

    train_labeled_dataloader, train_unlabeled_dataloader, test_dataloader = load_train_data(config)
    model, ema_model, optimizer, ema_optimizer = load(config)
    pseudo_label = PseudoLabel(config)

    metrics = {
        'train_loss': 0,
        'train_accuracy': 0,
        'train_steps': 0,
        'test_accuracy': 0,
        'test_steps': 0
    }

    best_test_accuracy = config.best_test_accuracy

    for epoch_step in range(config.epoch_step + 1, config.epochs + 1):
        train_labeled_dataloader_iterator = iter(train_labeled_dataloader)
        train_unlabeled_dataloader_iterator = iter(train_unlabeled_dataloader)

        train_progress_bar = tqdm(range(config.iterations))
        for batch_step in train_progress_bar:
            inputs_x, targets_x, ub, train_labeled_dataloader_iterator, train_unlabeled_dataloader_iterator = on_train_batch_start(
                train_labeled_dataloader, train_unlabeled_dataloader, train_labeled_dataloader_iterator,
                train_unlabeled_dataloader_iterator, config)
            train_step(epoch_step, batch_step, inputs_x, targets_x, ub, pseudo_label, model, ema_model,
                       optimizer, ema_optimizer, metrics, config)
            on_train_batch_end(epoch_step, inputs_x, targets_x, ema_model, metrics, train_progress_bar)

        test_progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for batch_step, batch in test_progress_bar:
            validation_step(ema_model, batch, metrics, config)
            on_validation_batch_end(epoch_step, metrics, test_progress_bar)

        best_test_accuracy = on_epoch_end(epoch_step, best_test_accuracy, model, ema_model, optimizer, config, metrics)

        metrics = {
            'train_loss': 0,
            'train_accuracy': 0,
            'train_steps': 0,
            'test_accuracy': 0,
            'test_steps': 0
        }


def train_step(epoch_step, batch_step, inputs_x, targets_x, ub, pseudo_label, model, ema_model,
               optimizer, ema_optimizer, metrics, config):
    x_logits, x_targets, u_logits, u_targets = pseudo_label(inputs_x, targets_x, ub, model, ema_model, config)

    loss = pseudo_label_loss(epoch_step, batch_step, x_logits, x_targets, u_logits, u_targets, config)
    metrics['train_loss'] += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema_optimizer.step()

    apply_weight_decay(model, config)


def apply_weight_decay(model, config):
    weight_decay = 0.02 * config.learning_rate
    for param in model.state_dict().values():
        if param.dtype == torch.float32:
            param.mul_(1 - weight_decay)


def validation_step(model, batch, metrics, config):
    model.eval()

    inputs, labels = batch
    inputs, labels = inputs.to(config.device), labels.to(config.device)

    with torch.no_grad():
        logits = model(inputs)
        labels = labels.detach().cpu()
        predictions = torch.max(logits, axis=1)[1].detach().cpu()

    metrics['test_steps'] += 1
    metrics['test_accuracy'] += accuracy_score(labels, predictions)


def on_train_batch_start(train_labeled_dataloader, train_unlabeled_dataloader, train_labeled_dataloader_iterator,
                         train_unlabeled_dataloader_iterator, config):
    try:
        inputs_x, targets_x = train_labeled_dataloader_iterator.next()
    except StopIteration as e:
        train_labeled_dataloader_iterator = iter(train_labeled_dataloader)
        inputs_x, targets_x = train_labeled_dataloader_iterator.next()

    try:
        ub, _ = train_unlabeled_dataloader_iterator.next()
    except StopIteration as e:
        train_unlabeled_dataloader_iterator = iter(train_unlabeled_dataloader)
        ub, _ = train_unlabeled_dataloader_iterator.next()

    targets_x = torch.zeros(config.batch_size, config.dataset_classes).scatter_(1, targets_x.view(-1, 1), 1)

    inputs_x, targets_x = inputs_x[0].to(config.device), targets_x.to(config.device)
    ub = ub[0].to(config.device)

    return inputs_x, targets_x, ub, train_labeled_dataloader_iterator, train_unlabeled_dataloader_iterator


def on_train_batch_end(epoch_step, x_hat, y, ema_model, metrics, train_progress_bar):
    ema_model.eval()

    ema_logits = ema_model(x_hat)
    labels = np.argmax(y.detach().cpu(), axis=1)
    predictions = torch.max(ema_logits, axis=1)[1].detach().cpu()

    metrics['train_steps'] += 1
    metrics['train_accuracy'] += accuracy_score(labels, predictions)

    train_loss = metrics['train_loss'] / metrics['train_steps']
    train_accuracy = metrics['train_accuracy'] / metrics['train_steps']

    train_progress_bar.set_description(
        'Epoch:{} | train_accuracy {:.3f} | train_loss {:.3f}'.format(
            epoch_step, train_accuracy, train_loss)
    )


def on_validation_batch_end(epoch_step, metrics, test_progress_bar):
    test_accuracy = metrics['test_accuracy'] / metrics['test_steps']

    test_progress_bar.set_description(
        'Epoch:{} | test_accuracy {:.3f}'.format(
            epoch_step, test_accuracy)
    )


def on_epoch_end(epoch_step, best_test_accuracy, model, ema_model, optimizer, config, metrics):
    test_accuracy = metrics['test_accuracy'] / metrics['test_steps']

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy

        checkpoint_to_save = {
            'epoch_step': epoch_step,
            'learning_rate': config.learning_rate,
            'ema_decay': config.ema_decay,
            'lambda_u': config.lambda_u,
            'alpha': config.alpha,
            't': config.t,
            'k': config.k,
            'ema': config.ema,
            'mix_up': config.mix_up,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'iterations': config.iterations,
            'labeled_data': config.labeled_data,
            'model_state': model.state_dict(),
            'ema_model_state': ema_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_test_accuracy': best_test_accuracy
        }

        checkpoint_to_save_path = f'./experiments/checkpoint-{config.dataset_name}-{config.labeled_data}-{config.k}-{config.t}-{config.mix_up}-{config.ema}.bin'
        torch.save(checkpoint_to_save, checkpoint_to_save_path)

        print(f'Checkpoint save in {checkpoint_to_save_path} with test_accuracy:{best_test_accuracy}')

    return best_test_accuracy


def load(config):
    if config.model == 'wideresnet':
        model = WideResNet(num_classes=config.dataset_classes)
        ema_model = WideResNet(num_classes=config.dataset_classes)
    else:
        model = CNN13(num_classes=config.dataset_classes)
        ema_model = CNN13(num_classes=config.dataset_classes)

    for param in ema_model.parameters():
        param.detach_()

    model.to(config.device)
    ema_model.to(config.device)

    torch.backends.cudnn.benchmark = True

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    ema_optimizer = WeightEMA(model, ema_model, alpha=config.ema_decay)

    if config.resume:
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)

        model.load_state_dict(checkpoint['model_state'])
        ema_model.load_state_dict(checkpoint['ema_model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        # optimizer state should be moved to corresponding device
        for optimizer_state in optimizer.state.values():
            for k, v in optimizer_state.items():
                if isinstance(v, torch.Tensor):
                    optimizer_state[k] = v.to(config.device)

    return model, ema_model, optimizer, ema_optimizer


if __name__ == '__main__':
    main()
