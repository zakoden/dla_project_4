import unittest

from hw_asr.model.deepspeech_model import DeepSpeechModel
from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.parse_config import ConfigParser

class TestDeepSpeech(unittest.TestCase):
    def test_dim(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            ds = LibrispeechDataset(
                "dev-clean", text_encoder=config_parser.get_text_encoder(),
                config_parser=config_parser
            )

            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            model = DeepSpeechModel(batch["spectrogram"].shape[-2], 10, seq_batch_norm=True)
            model(batch["spectrogram"])

