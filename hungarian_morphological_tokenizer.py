from collections import Counter
import re
import nltk
import sys
import os
import pickle
import time
from vocabulary import Vocabulary
import configs 
from purepospy.purepospy import PurePOS
from emmorphpy.emmorphpy import EmMorphPy


class HungarianMorphologicalTokenizer:

    def __init__(self, pos_tagger_model_path='purepospy/purepospy/szeged.model'):
        self.vocab = Vocabulary()
        self.pos_tagger = PurePOS(pos_tagger_model_path)
        self.morph_analyzer = EmMorphPy(hfst_lookup='data/hfst/bin/hfst-lookup')

    def count_morphs(self, file, total_line_count, begin, end):
        counter = Counter()
        line_count = 0
        with open(file, encoding='utf-8') as f:
            for line in f:
                if line_count >= begin and line_count < end:
                    line = re.sub(r'(?<=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])[\’\'\'](?=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])', ' \'',
                                  line).strip()

                    for sent in nltk.sent_tokenize(line):
                        sent_tokens = []
                        for token_pos in self.pos_tagger.tag_sentence(sent.split()):
                            morphemes_data = self.morph_analyzer.dstem(token_pos[0])
                            disambiguated_morpheme_data = None
                            for analysis in morphemes_data:
                                if analysis[1] == token_pos[2]:
                                    disambiguated_morpheme_data = analysis[3]

                            idx = 0
                            if disambiguated_morpheme_data is None:
                                sent_tokens.append(token_pos[0])
                            elif token_pos[2] == '[Punct]':
                                sent_tokens.append(token_pos[0])
                            else:
                                morphemes = disambiguated_morpheme_data.split('+')
                                for morpheme in morphemes:
                                    if "=" in morpheme:
                                        morpheme = morpheme[morpheme.index("=") + 1:].strip()
                                    elif "[" in morpheme:
                                        morpheme = morpheme[:morpheme.index("[")].strip()
                                    else:
                                        pass
                                        # print(morpheme)

                                    if morpheme != '' and idx == 0:
                                        sent_tokens.append(morpheme)
                                    elif morpheme != '' and idx > 0:
                                        sent_tokens.append("#" + morpheme)
                                    else:
                                        pass
                                        # print(morpheme)

                                    idx += 1

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

        with open("hun_news_morph_full_vocab.txt", 'w+') as f:
            for k, v in counter.most_common():
                f.write("{} {}\n".format(k, v))

        # Create a vocab wrapper and add some special tokens.
        self.vocab.add_word(configs.PAD)
        self.vocab.add_word(configs.UNK)
        self.vocab.add_word(configs.SOS)
        self.vocab.add_word(configs.EOS)

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            self.vocab.add_word(word)

        print("Training elapsed time: ", time.time() - time.time())
        return self.vocab

    def save(self, file_path):
        pickle.dump(self.vocab, open(file_path, "wb"))

    def load(self, file_path):
        self.vocab = pickle.load(open(file_path, "rb"))

    def encode(self, text):
        text = re.sub(r'(?<=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])[\’\'\'](?=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])', ' \'', text).strip()

        encoded_ids = []
        for sent in nltk.sent_tokenize(text):
            for token_pos in self.pos_tagger.tag_sentence(sent.split()):
                morphemes_data = self.morph_analyzer.dstem(token_pos[0])
                disambiguated_morpheme_data = None
                for analysis in morphemes_data:
                    if analysis[1] == token_pos[2]:
                        disambiguated_morpheme_data = analysis[3]

                idx = 0
                if disambiguated_morpheme_data is None:
                    encoded_ids.append(self.vocab(token_pos[0]))
                elif token_pos[2] == '[Punct]':
                    encoded_ids.append(self.vocab(token_pos[0]))
                else:
                    morphemes = disambiguated_morpheme_data.split('+')

                    for morpheme in morphemes:
                        if "=" in morpheme:
                            morpheme = morpheme[morpheme.index("=") + 1:].strip()
                        elif "[" in morpheme:
                            morpheme = morpheme[:morpheme.index("[")].strip()
                        else:
                            pass

                        if morpheme != '' and idx == 0:
                            encoded_ids.append(self.vocab(morpheme))
                        elif morpheme != '' and idx > 0:
                            encoded_ids.append(self.vocab("#" + morpheme))
                        else:
                            #print(morpheme)
                            pass

                        idx += 1

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

    def print_ids(self, ids):
        print([self.vocab.idx2word[id] for id in ids])

if __name__ == "__main__":
    hun_tokenizer = HungarianMorphologicalTokenizer()
