import logging

import torchaudio
from torch.utils.data import Dataset

from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class TestAudioDataset(Dataset):
    def __init__(self, data_path_lst, config_parser, *args, **kwargs):

        self.config_parser = config_parser

        print("Test audio paths:", data_path_lst)

        self._index = []
        for cur_path in data_path_lst:
            self._index.append(
                {
                    "path": cur_path,
                    "text": "[No text]",
                    "audio_len": 0,
                }
            )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)

        return {
            "audio": audio,
            "duration": audio.size(1) / self.config_parser["preprocessing"]["sr"],
            "text": data_dict["text"],
            "audio_path": audio_path,
        }

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

