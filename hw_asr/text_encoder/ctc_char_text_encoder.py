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
        hypos: List[Hypothesis] = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
