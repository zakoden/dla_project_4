import torch
import logging
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    batch_size = len(dataset_items)
    time_dim = 0
    text_length_dim = 0
    feature_length_dim = dataset_items[0]["spectrogram"].shape[-2]
    for cur_item in dataset_items:
        time_dim = max(time_dim, cur_item["spectrogram"].shape[-1])
        text_length_dim = max(text_length_dim, cur_item["text_encoded"].shape[-1])

    # "spectrogram"  torch.tensor
    batch_spectrogram = torch.zeros(batch_size, feature_length_dim, time_dim)
    # "text_encoded" [int] torch.tensor
    batch_text_encoded = torch.zeros(batch_size, text_length_dim, dtype=torch.int)
    # "text_encoded_length" [int] torch.tensor
    batch_text_encoded_length = torch.zeros(batch_size, dtype=torch.int)
    # "spectrogram_length" [int] torch.tensor
    batch_spectrogram_length = torch.zeros(batch_size, dtype=torch.int)
    # "text"   List[str]
    batch_text = []
    # "audio_path"   List[str]
    batch_audio_path = []

    for ind, cur_item in enumerate(dataset_items):
        batch_spectrogram[ind, :, :cur_item["spectrogram"].shape[-1]] = cur_item["spectrogram"]

        batch_text_encoded[ind, :cur_item["text_encoded"].shape[-1]] = cur_item["text_encoded"]

        batch_text_encoded_length[ind] = cur_item["text_encoded"].shape[-1]

        batch_spectrogram_length[ind] = cur_item["spectrogram"].shape[-1]

        batch_text.append(cur_item["text"])

        batch_audio_path.append(cur_item["audio_path"])

    result_batch = {"spectrogram": batch_spectrogram,
                    "spectrogram_length": batch_spectrogram_length,
                    "text_encoded": batch_text_encoded,
                    "text_encoded_length": batch_text_encoded_length,
                    "text": batch_text,
                    "audio_path": batch_audio_path}
    return result_batch