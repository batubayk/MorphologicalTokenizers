from tokenizers import (CharBPETokenizer,
                        SentencePieceBPETokenizer,
                        BertWordPieceTokenizer)
import os, re, pickle, sys, math, time
from configs import *
from hungarian_morphological_tokenizer import HungarianMorphologicalTokenizer
from hungarian_root_morphological_tokenizer import HungarianRootMorphologicalTokenizer
from turkish_morphological_tokenizer import TurkishMorphologicalTokenizer
from turkish_root_morphological_tokenizer import TurkishRootMorphologicalTokenizer
from whitespace_tokenizer import WhiteSpaceTokenizer
from sentencepice_tokenizer import SPTokenizer
from english_root_stem_tokenizer import EnglishStemTokenizer
import configs
package_directory = os.path.dirname(os.path.abspath(__file__))

class Tokenizers():
    def __init__(self, tokenizer_type):
        print("Tokenizer type: ", tokenizer_type)
        self.tokenizer_type = tokenizer_type
        self.special_tokens = [configs.PAD, configs.UNK, configs.SOS, configs.EOS]
        if tokenizer_type == 'hf_bpe':
            self.tokenizer = CharBPETokenizer()
        elif tokenizer_type == 'hf_wordpiece':
            self.tokenizer = BertWordPieceTokenizer()
        elif tokenizer_type == 'hf_sentencepiece':
            self.tokenizer = SentencePieceBPETokenizer()
        elif tokenizer_type == 'sp_unigram':
            self.tokenizer = SPTokenizer()
        elif tokenizer_type == 'sp_bpe':
            self.tokenizer = SPTokenizer()
        elif tokenizer_type == 'whitespace':
            self.tokenizer = WhiteSpaceTokenizer()
        elif tokenizer_type == 'tr_morph':
            self.tokenizer = TurkishMorphologicalTokenizer(zemberek_path=os.path.join(package_directory, 'data', 'zemberek-full.jar'))
        elif tokenizer_type == 'tr_root_morph':
            self.tokenizer = TurkishRootMorphologicalTokenizer(zemberek_path=os.path.join(package_directory, 'data', 'zemberek-full.jar'))
        elif tokenizer_type == 'hu_morph':
            self.tokenizer = HungarianMorphologicalTokenizer(pos_tagger_model_path =os.path.join(package_directory, 'data', 'szeged.model'))
        elif tokenizer_type == 'hu_root_morph':
            self.tokenizer = HungarianRootMorphologicalTokenizer(pos_tagger_model_path =os.path.join(package_directory, 'data', 'szeged.model'))
        elif tokenizer_type == 'en_root_stem':
            self.tokenizer = EnglishStemTokenizer()
        else:
            raise Exception("Wrong tokenizer type!")

        # self.tokenizer.add_special_tokens(self.special_tokens)

    def train(self, files, vocab_size, save_path=None, min_freq=configs.TOKENIZER_MIN_FREQ):
        if self.tokenizer_type == 'sp_bpe':
            self.tokenizer.train(files[0], vocab_size=vocab_size, model_type='bpe', save_path=save_path)
        elif self.tokenizer_type == 'sp_unigram':
            self.tokenizer.train(files[0], vocab_size=vocab_size, model_type='unigram', save_path=save_path)
        elif self.tokenizer_type == 'whitespace':
            self.tokenizer.train(files[0], vocab_size, min_freq)
        elif self.tokenizer_type == 'tr_morph':
            self.tokenizer.train(files[0], vocab_size, min_freq)
        elif self.tokenizer_type == 'tr_root_morph':
            self.tokenizer.train(files[0], vocab_size, min_freq)
        elif self.tokenizer_type == 'hu_morph':
            self.tokenizer.train(files[0], vocab_size, min_freq)
        elif self.tokenizer_type == 'hu_root_morph':
            self.tokenizer.train(files[0], vocab_size, min_freq)
        elif self.tokenizer_type == 'en_root_stem':
            self.tokenizer.train(files[0], vocab_size, min_freq)
        else:
            raise ('Cannot train wrong tokenizer value')

    def save(self, folder, file):
        path = folder + '/' + file
        self.tokenizer.save(path)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, folder, file):
        path = folder + '/' + file
        if self.tokenizer_type == 'hf_bpe':
            self.tokenizer = BPETokenizer(vocab_file=path + '-vocab.json', merges_file=path + '-merges.txt',
                                          unk_token=configs.UNK)
        elif self.tokenizer_type == 'hf_wordpiece':
            self.tokenizer = BertWordPieceTokenizer(vocab_file=path + '-vocab.txt', unk_token=configs.UNK)
        elif self.tokenizer_type == 'hf_sentencepiece':
            self.tokenizer = SentencePieceBPETokenizer(vocab_file=path + '-vocab.json',
                                                       merges_file=path + '-merges.txt', unk_token=configs.UNK)
        elif self.tokenizer_type == 'sp_bpe':
            self.tokenizer.load(folder + '/' + file[3:] + '.model')
        elif self.tokenizer_type == 'sp_unigram':
            self.tokenizer.load(folder + '/' + file[3:] + '.model')
        elif self.tokenizer_type == 'whitespace':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'tr_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'tr_root_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'hu_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'hu_root_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'en_root_stem':
            self.tokenizer.load(path)
        else:
            raise Exception("Wrong tokenizer type!")

    def load(self, path):
        if self.tokenizer_type == 'hf_bpe':
            self.tokenizer = BPETokenizer(vocab_file=path + '-vocab.json', merges_file=path + '-merges.txt',
                                          unk_token=configs.UNK)
        elif self.tokenizer_type == 'hf_wordpiece':
            self.tokenizer = BertWordPieceTokenizer(vocab_file=path + '-vocab.txt', unk_token=configs.UNK)
        elif self.tokenizer_type == 'hf_sentencepiece':
            self.tokenizer = SentencePieceBPETokenizer(vocab_file=path + '-vocab.json',
                                                       merges_file=path + '-merges.txt', unk_token=configs.UNK)
        elif self.tokenizer_type == 'sp_bpe':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'sp_unigram':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'whitespace':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'tr_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'tr_root_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'hu_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'hu_root_morph':
            self.tokenizer.load(path)
        elif self.tokenizer_type == 'en_root_stem':
            self.tokenizer.load(path)
        else:
            raise Exception("Wrong tokenizer type!")

    def encode(self, text):
        encoded = None
        if self.tokenizer_type.startswith('sp_'):
            encoded = self.tokenizer.encode(text)
        elif self.tokenizer_type.startswith('whitespace'):
            encoded = self.tokenizer.encode(text)
        elif self.tokenizer_type.startswith('tr_morph'):
            encoded = self.tokenizer.encode(text)
        elif self.tokenizer_type.startswith('tr_root_morph'):
            encoded = self.tokenizer.encode(text)
        elif self.tokenizer_type.startswith('hu_morph'):
            encoded = self.tokenizer.encode(text)
        elif self.tokenizer_type.startswith('hu_root_morph'):
            encoded = self.tokenizer.encode(text)
        elif self.tokenizer_type.startswith('en_root_stem'):
            encoded = self.tokenizer.encode(text)
        else:
            encoded = self.tokenizer.encode(text)
            encoded = encoded.ids

        return encoded

    def decode(self, text):
        decoded = None
        if self.tokenizer_type.startswith('sp_'):
            decoded = self.tokenizer.decode(text)
        elif self.tokenizer_type.startswith('whitespace'):
            decoded = self.tokenizer.decode(text)
        elif self.tokenizer_type.startswith('tr_morph'):
            decoded = self.tokenizer.decode(text)
        elif self.tokenizer_type.startswith('tr_root_morph'):
            decoded = self.tokenizer.decode(text)
        elif self.tokenizer_type.startswith('hu_morph'):
            decoded = self.tokenizer.decode(text)
        elif self.tokenizer_type.startswith('hu_root_morph'):
            decoded = self.tokenizer.decode(text)
        elif self.tokenizer_type.startswith('en_root_stem'):
            decoded = self.tokenizer.decode(text)
        else:
            decoded = self.tokenizer.decode(text)
            decoded = decoded.ids

        return decoded

    def tokenize(self, text):
        tokens = None
        if self.tokenizer_type.startswith('sp_'):
            tokens = self.tokenizer.tokenize(text)
        elif self.tokenizer_type.startswith('whitespace'):
            tokens = self.tokenizer.tokenize(text)
        elif self.tokenizer_type.startswith('tr_morph'):
            tokens = self.tokenizer.tokenize(text)
        elif self.tokenizer_type.startswith('tr_root_morph'):
            tokens = self.tokenizer.tokenize(text)
        elif self.tokenizer_type.startswith('hu_morph'):
            tokens = self.tokenizer.tokenize(text)
        elif self.tokenizer_type.startswith('hu_root_morph'):
            tokens = self.tokenizer.tokenize(text)
        elif self.tokenizer_type.startswith('en_root_stem):
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = self.tokenizer.tokenize(text)
            tokens = decoded.ids

        return tokens

    def convert_tokens_to_ids(self, text):
        ids = None
        if self.tokenizer_type.startswith('sp_'):
            ids = self.tokenizer.convert_tokens_to_ids(text)
        elif self.tokenizer_type.startswith('whitespace'):
            ids = self.tokenizer.convert_tokens_to_ids(text)
        elif self.tokenizer_type.startswith('tr_morph'):
            ids = self.tokenizer.convert_tokens_to_ids(text)
        elif self.tokenizer_type.startswith('tr_root_morph'):
            ids = self.tokenizer.convert_tokens_to_ids(text)
        elif self.tokenizer_type.startswith('hu_morph'):
            ids = self.tokenizer.convert_tokens_to_ids(text)
        elif self.tokenizer_type.startswith('hu_root_morph'):
            ids = self.tokenizer.convert_tokens_to_ids(text)
        elif self.tokenizer_type.startswith('en_root_stem):
            ids = self.tokenizer.convert_tokens_to_ids(text)
        else:
            ids = self.tokenizer.convert_tokens_to_ids(text)
            ids = decoded.ids

        return ids

    def size(self):
        return self.tokenizer.size()

    def print_ids(self,ids):
        return self.tokenizer.print_ids(ids)


if __name__ == "__main__":
    model_type = 'tr_root_morph'
    print(os.getcwd())
    ZEMBEREK_PATH = os.path.join('data', 'zemberek-full.jar')
    TRAIN_FILE_PATH = os.path.join('data', 'test.tsv')
    TOKENIZER_PATH = "files/tr_root_morph_50000"

    tokenizer = Tokenizers(model_type)
    #tokenizer.train([TRAIN_FILE_PATH], vocab_size=50000, min_freq=2)
    #tokenizer.save(TOKENIZER_PATH)
    tokenizer.load(TOKENIZER_PATH)
    text = "eve giderken kitabımı geri verdim."
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(text)
    print(tokens)
    print(ids)
    tokenizer.print_ids(ids)
    print(tokenizer.decode(ids))
