import argparse
import collections
import warnings

import numpy as np
import torch

import hw_asr.loss as module_loss
import hw_asr.model as module_arch
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model_G = config.init_obj(config["arch_generator"], module_arch)
    disc_MPD = config.init_obj(config["arch_MPD"], module_arch)
    disc_MSD = config.init_obj(config["arch_MSD"], module_arch)

    #logger.info(model_G)
    #logger.info(disc_MSD)
    #logger.info(disc_MPD)


    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model_G = model_G.to(device)
    disc_MSD = disc_MSD.to(device)
    disc_MPD = disc_MPD.to(device)
    if len(device_ids) > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=device_ids)
        disc_MSD = torch.nn.DataParallel(disc_MSD, device_ids=device_ids)
        disc_MPD = torch.nn.DataParallel(disc_MPD, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    model_lst = [model_G, disc_MPD, disc_MSD]
    optimizer_lst = []
    lr_scheduler_lst = []

    for model in model_lst:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
        lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)
        optimizer_lst.append(optimizer)
        lr_scheduler_lst.append(lr_scheduler)

    trainer = Trainer(
        model_lst[1:],
        optimizer_lst[1:],
        lr_scheduler_lst[1:],
        model_lst[0],
        loss_module,
        optimizer_lst[0],
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler_lst[0],
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
