import os
from collections import Counter
from konlpy.tag import Komoran, Mecab, Okt
from typing import Optional, List, Union


class KoNLPyPreTokenizer:
    def __init__(self, konlpy_tagger):
        self.konlpy_tagger = konlpy_tagger

    def __call__(self, sentence):
        return self.pre_tokenize(sentence)

    def pre_tokenize(self, sentence):
        return ' '.join(self.konlpy_tagger.morphs(sentence))


class KoNLPyWordPieceTokenizer:
    def __init__(self, konlpy_tagger, wordpieces_prefix="##", use_tag=False):
        self.konlpy_tagger = konlpy_tagger
        self.use_tag = use_tag
        if use_tag:
            def pretokenize(eojeol):
                return self.konlpy_tagger.pos(eojeol, join=True)
        else:
            def pretokenize(eojeol):
                return self.konlpy_tagger.morphs(eojeol)
        self.pretokenize = pretokenize
        self.prefix = wordpieces_prefix

    def tokenize(self, sent):
        def _tokenize(eojeol):
            split_tokens = self.pretokenize(eojeol)
            if len(split_tokens) <= 1:
                return split_tokens
            return [split_tokens[0]] + [f'{self.prefix}{sub}' for sub in split_tokens[1:]]

        split_tokens = []
        for eojeol in sent.split():
            split_tokens += _tokenize(eojeol)
        return split_tokens

    def token_to_alphabets(self, token):
        if token[:2] == self.prefix:
            token = token[2:]
        if self.use_tag:
            return list(token.rsplit('/')[0])
        return list(token)
