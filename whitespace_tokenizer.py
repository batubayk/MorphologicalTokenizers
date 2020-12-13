from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import pickle
from vocabulary import Vocabulary
import configs


class WhiteSpaceTokenizer:
    def __init__(self):
        self.vocab = Vocabulary()

    def train(self,file,vocab_size=None,min_freq=2):
        counter = Counter()
        if type(file) is list:
            file=file[0]
        with open(file, encoding='utf-8') as f:
            for line in f:
                counter.update(line.split())

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = []
        total=0
        for word, cnt in counter.most_common():
            if total>=vocab_size-4:
                break
            if cnt >= min_freq:
                words.append(word)
                total+=1

        # Create a vocab wrapper and add some special tokens.
        self.vocab.add_word(configs.PAD)
        self.vocab.add_word(configs.UNK)
        self.vocab.add_word(configs.SOS)
        self.vocab.add_word(configs.EOS)

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            self.vocab.add_word(word)

        return self.vocab

    def save(self,file_path):
        pickle.dump(self.vocab, open(file_path, "wb" ) )

    def load(self,file_path):
        self.vocab = pickle.load(open(file_path, "rb" ) )

    def encode(self,text):
        return [self.vocab(token) for token in text.split()]

    def decode(self,ids):
        return " ".join([self.vocab.idx2word[id] for id in ids])

    def size(self):
        return len(self.vocab)