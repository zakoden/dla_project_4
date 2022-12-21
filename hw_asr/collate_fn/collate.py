import torch
import logging
from typing import List
from hw_asr.mel_spectrogram import melspec_config, melspec_func


logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    batch_size = len(dataset_items)
    time_dim = 0
    for cur_item in dataset_items:
        time_dim = max(time_dim, cur_item["audio"].shape[-1])

    # "audio"  torch.tensor
    batch_audio = torch.zeros(batch_size, time_dim)
    # "text"   List[str]
    batch_text = []
    # "audio_path"   List[str]
    batch_audio_path = []

    for ind, cur_item in enumerate(dataset_items):
        batch_audio[ind, :cur_item["audio"].shape[-1]] = cur_item["audio"]

        batch_text.append(cur_item["text"])

        batch_audio_path.append(cur_item["audio_path"])

    batch_spectrogram = melspec_func(batch_audio)

    result_batch = {"audio": batch_audio,
                    "spectrogram": batch_spectrogram,
                    "text": batch_text,
                    "audio_path": batch_audio_path}
    return result_batch
