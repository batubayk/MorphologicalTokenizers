import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from tokenizer_modules.vocabulary import Vocabulary


class Tokenizer(ABC):
    def __init__(self, configs, **kwargs):
        self.configs = configs
        self.vocab = Vocabulary(self.configs['params']['special_tokens']) if 'special_tokens' in self.configs[
            'params'] else Vocabulary()

    @abstractmethod
    def tokenize(self):
        pass

    def train(self):
        self.vocab = Vocabulary(self.configs['params']['special_tokens']) if 'special_tokens' in self.configs[
            'params'] else Vocabulary()
        begin_time = time.time()
        counter = self.count_morphs()

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = []
        total = 0
        for word, cnt in counter.most_common():
            if total >= self.configs['params']['vocab_size'] - len(self.configs['params']['special_tokens']):
                break
            if cnt >= self.configs['params']['min_freq']:
                words.append(word)
                total += 1

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            self.vocab.add_word(word)

        logging.info("Training elapsed time: ", time.time() - begin_time)
        self.save()
        return self.vocab

    def count_morphs(self):
        counter = Counter()

        input_file_paths = [str(x) for x in Path(self.configs['params']['input_dir']).glob(
            self.configs['params']['input_file_pattern'])]
        file_num = 0
        total_file_count = len(input_file_paths)
        for input_file_path in input_file_paths:
            file_num += 1
            cur_file_total_line_count = self.count_lines(input_file_path)
            with open(input_file_path, encoding='utf-8') as f:
                for line in tqdm(f,
                                 desc="File {}/{} processing. {}".format(file_num, total_file_count, input_file_path),
                                 total=cur_file_total_line_count, position=0, leave=True):
                    line = line.strip()
                    line_tokens = self.tokenize(line)
                    counter.update(line_tokens)
        return counter

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab(token) for token in tokens]

    def save(self):
        output_file_path = Path(self.configs['params']['model']['path'])
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        if output_file_path.exists() and self.configs['params']['model'].get('overwrite', False):
            output_file_path.unlink()
        pickle.dump(self.vocab, open(output_file_path, "wb"))

    def load(self):
        model_file_path = Path(self.configs['params']['model']['path'])
        if model_file_path.exists():
            self.vocab = pickle.load(open(model_file_path, "rb"))
            logging.info("Loading model file: {}".format(model_file_path))
        else:
            logging.error("Model not found at path: {}".format(model_file_path))

    def encode(self, text):
        tokens = self.tokenize(text)
        encode_ids = [self.vocab(token) for token in tokens]
        return encode_ids

    def decode(self, ids):
        text = ""
        for id in ids:
            token = self.vocab.idx2word[id]
            if token.startswith(self.configs['params']['token_affix']):
                text += token[len(self.configs['params']['token_affix']):]
            else:
                text += " " + token
        return text

    def print_ids(self, ids):
        print([self.vocab.idx2word[id] for id in ids])

    def size(self):
        return len(self.vocab)

    def count_lines(self, file_name):
        line_count = 0
        with open(file_name, mode='r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
        return line_count
