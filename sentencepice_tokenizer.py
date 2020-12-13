from vocabulary import Vocabulary
import sentencepiece as spm
import configs
import pickle


class SPTokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.vocab = Vocabulary()

    def train(self, text_path, vocab_size, model_type, save_path):
        # model_prefix = "tokenizer/"+model_type
        # params='--input={} --model_prefix={} --vocab_size={} --model_type={} --max_sentence_length=10000 --bos_id=2 --eos_id=3 --unk_id=1 --pad_id=0 --bos_piece=<sos> --eos_piece=<eos> --unk_piece=<unk> --pad_piece=<pad>'.format(text_path, model_prefix,vocab_size,model_type)
        params = '--input={} --model_prefix={} --vocab_size={} --model_type={} --max_sentence_length=10000 --bos_id=-1 --eos_id=-1 --pad_id=-1 --unk_id=1 --user_defined_symbols={},{},{}'. \
            format(text_path, save_path, vocab_size, model_type, configs.PAD, configs.SOS, configs.EOS)

        print(params)

        spm.SentencePieceTrainer.Train(params)
        with open(save_path + '.vocab', encoding='utf-8') as f:
            for line in f:
                self.vocab.add_word(line.strip().split("\t")[0])
        pickle.dump(self.vocab, open(save_path + ".pickle", "wb"))

        self.load(save_path)

    def load(self, save_path):
        self.sp.Load(save_path + ".model")
        self.vocab = pickle.load(open(save_path + ".pickle", "rb"))

    def save(self, model_path):
        pass

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, id_array):
        return self.sp.DecodeIds(id_array)

    def text2pieces(self, text):
        return self.sp.EncodeAsPieces(text)

    def size(self):
        return self.sp.GetPieceSize()

    def article2ids(self, article):
        return self.sp.EncodeAsIds(article), []

    def article2ids_extra(self, article):
        return self.sp.EncodeAsIds(article), self.sp.EncodeAsIds(article), []

    def abstract2ids(self, abstract, article_oovs):
        return self.sp.EncodeAsIds(abstract)

    def abstract2ids_extra(self, abstract, article_oovs):
        return self.sp.EncodeAsIds(abstract), self.sp.EncodeAsIds(abstract)

