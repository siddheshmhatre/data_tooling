# Taken from https://huggingface.co/spaces/edugp/perplexity-lenses/blob/main/perplexity_lenses/perplexity.py

import os
import re
import unicodedata
from typing import Dict
from requests.exceptions import HTTPError

import kenlm
import sentencepiece
from huggingface_hub import cached_download, hf_hub_url

KENLM_MODEL_REPO = "edugp/kenlm"


class SentencePiece:
    def __init__(
        self,
        model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def do(self, text: dict) -> dict:
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    kenlm_model_dir = None
    sentence_piece_model_dir = None

    def __init__(
        self,
        kenlm_model_dir : str,
        sentence_piece_model_dir : str,
    ):
        self.kenlm_model_dir = kenlm_model_dir
        self.sentence_piece_model_dir = sentence_piece_model_dir
        try:
            self.model = kenlm.Model(self.kenlm_model_dir)
            self.tokenizer = SentencePiece(self.sentence_piece_model_dir)
        except OSError:
            os.remove(self.kenlm_model_dir)
            if os.path.exists(self.sentence_piece_model_dir):
                os.remove(self.sentence_piece_model_dir)
            raise OSError(
                "File was corrupt and should have been removed. Please, retry."
            )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str):
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.do(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)