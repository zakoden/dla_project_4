from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        empty_tok_ind = self.char2ind[self.EMPTY_TOK]
        decoded_str = ""
        prev_ind = empty_tok_ind
        for cur_ind in inds:
            if cur_ind == prev_ind:
                continue
            # else cur_ind != prev_ind
            prev_ind = cur_ind
            if cur_ind != empty_tok_ind:
                cur_tok = self.ind2char[cur_ind]
                decoded_str += cur_tok

        return decoded_str

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        hypos_dict = {('', self.EMPTY_TOK): 1.0}

        for column_ind in range(char_length):
            # extend
            new_hypos_dict = {}

            for (hypo_text, last_char), prob in hypos_dict.items():
                for char_ind in range(probs_length.item()):
                    cur_char = self.ind2char[char_ind]
                    new_last_char = cur_char
                    if cur_char == last_char:
                        new_hypo_text = hypo_text
                    else:
                        new_hypo_text = hypo_text
                        if cur_char != self.EMPTY_TOK:
                            new_hypo_text += cur_char

                    # (new_hypo_text, new_last_char)
                    new_prob = prob * probs[column_ind, char_ind]
                    if (new_hypo_text, new_last_char) not in new_hypos_dict:
                        new_hypos_dict[(new_hypo_text, new_last_char)] = 0.0
                    new_hypos_dict[(new_hypo_text, new_last_char)] += new_prob

            # cut
            hypos_dict = dict(list(sorted(new_hypos_dict.items(), key=lambda x: x[1]))[-beam_size:])

        hypos: List[Hypothesis] = []
        for (hypo_text, last_char), prob in hypos_dict.items():
            hypos.append(Hypothesis(text=hypo_text, prob=prob))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
