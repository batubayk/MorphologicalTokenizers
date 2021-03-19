import pickle
from pathlib import Path

import sentencepiece as spm

from tokenizer_modules.tokenizer import Tokenizer
from tokenizer_modules.vocabulary import Vocabulary


class SPTokenizer(Tokenizer):
    def __init__(self, configs):
        self.configs = configs
        self.vocab = Vocabulary()
        self.sp = spm.SentencePieceProcessor()
        if Path(self.configs['params']['model']['prefix']).exists():
            self.load()

    def train(self):
        input_file_paths = ",".join([str(x) for x in Path(self.configs['params']['input_dir']).glob(
            self.configs['params']['input_file_pattern'])])
        # model_prefix = "tokenizer/"+model_type
        # params='--input={} --model_prefix={} --vocab_size={} --model_type={} --max_sentence_length=10000 --bos_id=2 --eos_id=3 --unk_id=1 --pad_id=0 --bos_piece=<sos> --eos_piece=<eos> --unk_piece=<unk> --pad_piece=<pad>'.format(text_path, model_prefix,vocab_size,model_type)
        # params = '--input={} --model_prefix={} --vocab_size={} --model_type={} --max_sentence_length=10000 --bos_id=-1 --eos_id=-1 --pad_id=-1 --unk_id=1 --user_defined_symbols={},{},{}'. \
        params = '--input={} --model_prefix={} --vocab_size={} --model_type={} --max_sentence_length=10000 --bos_id=-1 --eos_id=-1 --pad_id=-1 --unk_id=1'. \
            format(input_file_paths, self.configs['params']['model']['prefix'], self.configs['params']['vocab_size'],
                   self.configs['params']['model_type'])

        spm.SentencePieceTrainer.Train(params)
        with open(self.configs['params']['model']['prefix'] + '.vocab', encoding='utf-8') as f:
            for line in f:
                self.vocab.add_word(line.strip().split("\t")[0])
        pickle.dump(self.vocab, open(self.configs['params']['model']['prefix'] + ".pickle", "wb"))

        self.load(self.configs['params']['model']['prefix'])

    def load(self, save_path):
        self.sp.Load(save_path + ".model")
        self.vocab = pickle.load(open(save_path + ".pickle", "rb"))

    def save(self):
        pass

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, id_array):
        return self.sp.DecodeIds(id_array)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    def size(self):
        return self.sp.GetPieceSize()
