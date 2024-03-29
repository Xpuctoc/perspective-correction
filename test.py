import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


# fix random seeds for reproducibility
SEED = 2023
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('test')

    data_loader = config.init_obj('data_loader', module_data)
    data_loader = data_loader.split_validation()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    model = torch.compile(model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    ssim_metric = 0.0

    with torch.no_grad():
        for i, (data, imgs, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            for img, pred_homo, true_homo in zip(imgs.numpy(), output.cpu().numpy(), target.cpu().numpy()):
                pred_homo = pred_homo.reshape(3, 3)
                true_homo = true_homo.reshape(3, 3)
                pred_img = cv2.warpPerspective(img, pred_homo, (512, 512))
                true_img = cv2.warpPerspective(img, true_homo, (512, 512))

                ssim_metric += ssim(true_img, pred_img, channel_axis=2)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples,
           'ssim': ssim_metric / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Perspective correction (test)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
