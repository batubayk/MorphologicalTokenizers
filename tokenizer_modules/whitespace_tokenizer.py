from collections import Counter
from pathlib import Path

from tokenizer_modules.tokenizer import Tokenizer
from tokenizer_modules.vocabulary import Vocabulary

class WhiteSpaceTokenizer(Tokenizer):
    def __init__(self, configs):
        self.configs = configs
        self.vocab = Vocabulary(self.configs['params']['special_tokens']) if 'special_tokens' in self.configs[
            'params'] else Vocabulary()
        if Path(self.configs['params']['model']['path']).exists():
            self.load()

    def tokenize(self, text):
        return [token for token in text.split()]

    def encode(self, text):
        return [self.vocab(token) for token in text.split()]

    def decode(self, ids):
        return " ".join([self.vocab.idx2word[id] for id in ids])