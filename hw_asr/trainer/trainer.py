import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.utils import inf_loop, MetricTracker
from hw_asr.mel_spectrogram import melspec_config, melspec_func


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            disc_model_lst,
            disc_optimizer_lst,
            disc_scheduler_lst,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.disc_model_lst = disc_model_lst
        self.disc_optimizer_lst = disc_optimizer_lst
        self.disc_scheduler_lst = disc_scheduler_lst

        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        for disc in self.disc_model_lst:
            disc.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                #self._log_spectrogram(batch["spectrogram"])
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        batch["wave_pred"] = self.model(batch["spectrogram"])["wave_pred"]
        batch["spectrogram_pred"] = melspec_func(batch["wave_pred"][:, 0, :])

        for disc_opt in self.disc_optimizer_lst:
            disc_opt.zero_grad()

        disc_MPD = self.disc_model_lst[0]
        disc_MSD = self.disc_model_lst[1]

        # discriminator losses
        logits_true, features_true = disc_MPD(batch["audio"])
        logits_pred, features_pred = disc_MPD(batch["wave_pred"].detach())
        MPD_loss = self.criterion.adv_discriminator_loss(logits_true, logits_pred)

        logits_true, features_true = disc_MSD(batch["audio"])
        logits_pred, features_pred = disc_MSD(batch["wave_pred"].detach())
        MSD_loss = self.criterion.adv_discriminator_loss(logits_true, logits_pred)

        disc_loss = MPD_loss + MSD_loss
        disc_loss.backward()
        for disc_opt in self.disc_optimizer_lst:
            disc_opt.step()

        # generator losses
        self.optimizer.zero_grad()

        mel_loss = 45.0 * F.l1_loss(batch["spectrogram_pred"], batch["spectrogram"])

        # MPD part
        logits_true, features_true = disc_MPD(batch["audio"])
        logits_pred, features_pred = disc_MPD(batch["wave_pred"])
        features_loss_MPD = 2.0 * self.criterion.features_loss(features_true, features_pred)
        logits_loss_MPD = self.criterion.adv_generator_loss(logits_pred)

        logits_true, features_true = disc_MSD(batch["audio"])
        logits_pred, features_pred = disc_MSD(batch["wave_pred"])
        features_loss_MSD = 2.0 * self.criterion.features_loss(features_true, features_pred)
        logits_loss_MSD = self.criterion.adv_generator_loss(logits_pred)

        generator_loss = mel_loss + features_loss_MPD + logits_loss_MPD + features_loss_MSD + logits_loss_MSD
        generator_loss.backward()
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.disc_scheduler_lst is not None:
            for disc_scheduler in self.disc_scheduler_lst:
                disc_scheduler.step()

        self.writer.add_scalar("generator loss", generator_loss.item())
        self.writer.add_scalar("discriminator loss", disc_loss.item())
        self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr())

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

