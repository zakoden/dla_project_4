import unittest
import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder()
        probs = torch.ones((10, len(text_encoder.ind2char))) * 0.1
        best_ans = "dog"
        inds = [text_encoder.char2ind[c] for c in best_ans]
        inds_pos = [1, 4, 7] # just random positions for d, o, g
        for i in range(len(inds_pos)):
            probs[inds_pos[i]][inds[i]] = 0.9
        for i in range(10):
            probs[i][text_encoder.char2ind['^']] += 0.1

        hypos = text_encoder.ctc_beam_search(probs, torch.tensor([10]), beam_size=10)
        self.assertEqual(best_ans, hypos[0].text)
        #print(hypos)
