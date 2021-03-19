from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from tokenizer_modules.tokenizer import Tokenizer
from tokenizer_modules.vocabulary import Vocabulary


class EnglishStemTokenizer(Tokenizer):
    def __init__(self, configs):
        self.configs = configs
        self.vocab = Vocabulary()
        self.porter_stemmer = PorterStemmer()
        if Path(self.configs['params']['model']['path']).exists():
            self.load()

    def get_affix(self, word, stem):
        parts = word.split(stem)
        if len(parts) == 2 and len(parts[0]) == 0 and len(parts[-1]) > 0:
            return parts[-1]
        else:
            return None

    def tokenize(self, text):
        tokens = []
        for sent in sent_tokenize(text):
            sent_tokens = []
            for word in word_tokenize(sent):
                stem = self.porter_stemmer.stem(word)
                affix = self.get_affix(word, stem)
                sent_tokens.append(stem)
                if affix is not None:
                    sent_tokens.append(self.configs['params']['token_affix'] + affix)

            tokens.extend(sent_tokens)

        return tokens
