import torch
import random

import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from models.cnn13 import CNN13
from loss import mix_match_loss
from optimizers.weight_ema import WeightEMA
from models.wideresnet import WideResNet
from sklearn.metrics import accuracy_score
from config import load_arguments, load_config
from semi_supervised.mix_match import MixMatch
from datasets.data_loaders import load_train_data

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    arguments = load_arguments()
    config = load_config(arguments.config)

    train_labeled_dataloader, train_unlabeled_dataloader, validation_dataloader = load_train_data(config)
    model, ema_model, optimizer, ema_optimizer = load(config)
    mix_match = MixMatch(config)

    metrics = {
        'train_loss': 0,
        'train_accuracy': 0,
        'train_steps': 0,
        'validation_accuracy': 0,
        'validation_steps': 0
    }

    best_validation_accuracy = config.best_test_accuracy

    for epoch_step in range(config.epoch_step + 1, config.epochs + 1):
        train_labeled_dataloader_iterator = iter(train_labeled_dataloader)
        train_unlabeled_dataloader_iterator = iter(train_unlabeled_dataloader) if train_unlabeled_dataloader else None

        train_progress_bar = tqdm(range(config.iterations))
        for batch_step in train_progress_bar:
            inputs_x, targets_x, ub, train_labeled_dataloader_iterator, train_unlabeled_dataloader_iterator = on_train_batch_start(
                train_labeled_dataloader, train_unlabeled_dataloader, train_labeled_dataloader_iterator,
                train_unlabeled_dataloader_iterator, config)
            train_step(epoch_step, batch_step, inputs_x, targets_x, ub,
                       mix_match, model, ema_model, optimizer, ema_optimizer, metrics, config)
            on_train_batch_end(epoch_step, inputs_x, targets_x, ema_model, metrics, train_progress_bar)

        test_progress_bar = tqdm(enumerate(validation_dataloader), total=len(validation_dataloader))
        for batch_step, batch in test_progress_bar:
            validation_step(ema_model, batch, metrics, config)
            on_validation_batch_end(epoch_step, metrics, test_progress_bar)

        best_validation_accuracy = on_epoch_end(epoch_step, best_validation_accuracy, model, ema_model, optimizer, config, metrics)

        metrics = {
            'train_loss': 0,
            'train_accuracy': 0,
            'train_steps': 0,
            'validation_accuracy': 0,
            'validation_steps': 0
        }


def train_step(epoch_step, batch_step, inputs_x, targets_x, ub, mix_match, model, ema_model,
               optimizer, ema_optimizer, metrics, config):
    x_logits, x_targets, u_logits, u_targets = mix_match(inputs_x, targets_x, ub, model, ema_model, config)

    loss = mix_match_loss(epoch_step, batch_step, x_logits, x_targets, u_logits, u_targets, config)
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

    metrics['validation_steps'] += 1
    metrics['validation_accuracy'] += accuracy_score(labels, predictions)


def on_train_batch_start(train_labeled_dataloader, train_unlabeled_dataloader, train_labeled_dataloader_iterator,
                         train_unlabeled_dataloader_iterator, config):
    try:
        inputs_x, targets_x = train_labeled_dataloader_iterator.next()
    except StopIteration as e:
        train_labeled_dataloader_iterator = iter(train_labeled_dataloader)
        inputs_x, targets_x = train_labeled_dataloader_iterator.next()

    try:
        ub, _ = [], [] if not train_unlabeled_dataloader_iterator else train_unlabeled_dataloader_iterator.next()
    except StopIteration as e:
        train_unlabeled_dataloader_iterator = iter(train_unlabeled_dataloader)
        ub, _ = [], [] if not train_unlabeled_dataloader_iterator else train_unlabeled_dataloader_iterator.next()

    targets_x = torch.zeros(config.batch_size, config.dataset_classes).scatter_(1, targets_x.view(-1, 1), 1)

    inputs_x, targets_x = inputs_x[0].to(config.device), targets_x.to(config.device)
    ub = [u_hat.to(config.device) for u_hat in ub]

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
    validation_accuracy = metrics['validation_accuracy'] / metrics['validation_steps']

    test_progress_bar.set_description(
        'Epoch:{} | validation_accuracy {:.3f}'.format(
            epoch_step, validation_accuracy)
    )


def on_epoch_end(epoch_step, best_validation_accuracy, model, ema_model, optimizer, config, metrics):
    validation_accuracy = metrics['validation_accuracy'] / metrics['validation_steps']

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy

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
            'best_test_accuracy': best_validation_accuracy
        }

        checkpoint_to_save_path = f'./experiments/checkpoint-{config.dataset_name}-{config.labeled_data}-{config.k}-{config.t}-{config.mix_up}-{config.ema}.bin'
        torch.save(checkpoint_to_save, checkpoint_to_save_path)

        print(f'Checkpoint save in {checkpoint_to_save_path} with validation_accuracy:{best_validation_accuracy}')

    return best_validation_accuracy


def load(config):
    if config.model == 'wideresnet':
        model = WideResNet(num_classes=config.dataset_classes)
        ema_model = WideResNet(num_classes=config.dataset_classes)
    else:
        model = CNN13(num_classes=config.dataset_classes)
        ema_model = CNN13(num_classes=config.dataset_classes)

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
