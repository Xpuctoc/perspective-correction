import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 2023
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    models = [0, 0, 0]
    optimizers = [0, 0, 0]
    scalers = [0, 0, 0]
    lr_schedulers = [0, 0, 0]

    for i in range(len(models)):
        # build model architecture, then print to console
        models[i] = config.init_obj('arch', module_arch)
        logger.info(models[i])

        models[i] = models[i].to(device)
        if len(device_ids) > 1:
            models[i] = torch.nn.DataParallel(models[i], device_ids=device_ids)

        scalers[i] = torch.cuda.amp.GradScaler()

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, models[i].parameters())
        optimizers[i] = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_schedulers[i] = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizers[i])

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    amp_autocast = torch.cuda.amp.autocast

    trainer = Trainer(models, criterion, metrics, optimizers,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      scalers=scalers,
                      amp_autocast=amp_autocast,
                      valid_data_loader=valid_data_loader,
                      lr_schedulers=lr_schedulers)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Perspective correction (train)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
