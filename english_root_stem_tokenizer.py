import os, re, pickle, sys, time
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import pickle
from vocabulary import Vocabulary
import configs

class EnglishStemTokenizer:
    def __init__(self):
        self.vocab = Vocabulary()
        self.porter_stemmer= PorterStemmer()

    def count_morphs(self, file, total_line_count, begin, end):
        counter = Counter()
        line_count = 0
        with open(file, encoding='utf-8') as f:
            for line in f:
                if line_count >= begin and line_count < end:
                    line = line.strip()
                    for sent in sent_tokenize(line):
                        sent_tokens = []
                        for word in word_tokenize(sent):
                            stem = self.porter_stemmer.stem(word)
                            affix = self.get_affix(word, stem)
                            sent_tokens.append(stem)
                            if affix is not None:
                                sent_tokens.append('#' + affix)

                        counter.update(sent_tokens)

                if line_count % 100 == 0:
                    sys.stdout.write('\r')
                    sys.stdout.write("{}/{} lines processed".format(line_count, total_line_count))
                    sys.stdout.flush()
                    # print("{}/{} lines processed".format(line_count,total_line_count))
                line_count += 1
        return counter

    def train(self, file, vocab_size=None, min_freq=configs.TOKENIZER_MIN_FREQ):
        begin_time = time.time()

        if type(file) is list:
            file = file[0]
        total_line_count = int(os.popen('wc -l {}'.format(file)).read().split()[0])
        counter = self.count_morphs(file, total_line_count, 0, total_line_count)

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = []
        total = 0
        for word, cnt in counter.most_common():
            if total >= vocab_size - 4:
                break
            if cnt >= min_freq:
                words.append(word)
                total += 1

        # Create a vocab wrapper and add some special tokens.
        self.vocab.add_word(configs.PAD)
        self.vocab.add_word(configs.UNK)
        self.vocab.add_word(configs.SOS)
        self.vocab.add_word(configs.EOS)

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            self.vocab.add_word(word)

        print("Training elapsed time: ", time.time() - begin_time)
        return self.vocab

    def save(self, file_path):
        pickle.dump(self.vocab, open(file_path, "wb"))

    def load(self, file_path):
        self.vocab = pickle.load(open(file_path, "rb"))

    def encode(self,text):
        encoded_ids = []
        for sent in sent_tokenize(text):
            for word in word_tokenize(sent):
                stem = self.porter_stemmer.stem(word)
                affix = self.get_affix(word, stem)
                encoded_ids.append(self.vocab(stem))
                if affix is not None:
                    encoded_ids.append(self.vocab('#' + affix))
        return encoded_ids

    def decode(self, ids):
        text = ""
        for id in ids:
            token = self.vocab.idx2word[id]
            if token.startswith('#'):
                text += token[1:]
            else:
                text += " " + token
        return text

    def print_ids(self,ids):
        print([self.vocab.idx2word[id] for id in ids])

    def size(self):
        return len(self.vocab)

    def get_affix(self, word, stem):
        parts = word.split(stem)
        if len(parts)==2 and len(parts[0])==0 and len(parts[-1])>0:
            return parts[-1]
        else:
            return None

    def article2ids_extra(self, text):
        text = text.strip()
        plain_ids = []
        encoded_ids = []
        oovs = []
        for sent in sent_tokenize(text):
            for word in word_tokenize(sent):
                stem = self.porter_stemmer.stem(word)
                id = self.vocab(stem)
                plain_ids.append(id)
                if id == configs.UNK_IDX:
                    if stem not in oovs:  # Add to list of OOVs
                        oovs.append(stem)
                    oov_num = oovs.index(stem)  # This is 0 for the first article OOV, 1 for the second article OOV...
                    encoded_ids.append(len(
                        self.vocab) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
                else:
                    encoded_ids.append(id)

                affix = self.get_affix(word, stem)
                if affix is not None:
                    id = self.vocab(affix)
                    plain_ids.append(id)
                    if id == configs.UNK_IDX:
                        if affix not in oovs:  # Add to list of OOVs
                            oovs.append(affix)
                        oov_num = oovs.index(
                            affix)  # This is 0 for the first article OOV, 1 for the second article OOV...
                        encoded_ids.append(len(
                            self.vocab) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
                    else:
                        encoded_ids.append(id)

        return plain_ids, encoded_ids, oovs

    def abstract2ids_extra(self, text, article_oovs):
        text = text.strip()
        plain_ids = []
        encoded_ids = []
        for sent in sent_tokenize(text):
            for word in word_tokenize(sent):
                stem = self.porter_stemmer.stem(word)
                id = self.vocab(stem)
                plain_ids.append(id)
                if id == configs.UNK_IDX:  # If w is an OOV word
                    if stem in article_oovs:  # If w is an in-article OOV
                        vocab_idx = len(self.vocab) + article_oovs.index(
                            stem)  # Map to its temporary article OOV number
                        encoded_ids.append(vocab_idx)
                    else:  # If w is an out-of-article OOV
                        encoded_ids.append(configs.UNK_IDX)  # Map to the UNK token id
                else:
                    encoded_ids.append(id)

                affix = self.get_affix(word, stem)
                if affix is not None:
                    id = self.vocab(affix)
                    plain_ids.append(id)
                    if id == configs.UNK_IDX:  # If w is an OOV word
                        if affix in article_oovs:  # If w is an in-article OOV
                            vocab_idx = len(self.vocab) + article_oovs.index(
                                affix)  # Map to its temporary article OOV number
                            encoded_ids.append(vocab_idx)
                        else:  # If w is an out-of-article OOV
                            encoded_ids.append(configs.UNK_IDX)  # Map to the UNK token id
                    else:
                        encoded_ids.append(id)

        return plain_ids, encoded_ids


if __name__=="__main__":
    TRAIN_FILE_PATH = "data/test.tsv"
    TOKENIZER_PATH = "files/cnn_dm_stem_50k"
    en_stem_tokenizer = EnglishStemTokenizer()
    en_stem_tokenizer.train(TRAIN_FILE_PATH, vocab_size=50000, min_freq=2)
    en_stem_tokenizer.save(TOKENIZER_PATH)
    en_stem_tokenizer.load(TOKENIZER_PATH)
    text="a man died after wolves attacked him ."
    ids = en_stem_tokenizer.encode(text)
    print(ids)
    en_stem_tokenizer.print_ids(ids)
    print(en_stem_tokenizer.decode(ids))