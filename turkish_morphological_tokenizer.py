import re, pickle, sys, time
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM
from collections import Counter
import configs
from vocabulary import Vocabulary

class TurkishMorphologicalTokenizer:

    def __init__(self,zemberek_path=None):
        self.vocab = Vocabulary()

        if zemberek_path is None:
            self.ZEMBEREK_PATH: str ='zemberek-full.jar'
        else:
            self.ZEMBEREK_PATH=zemberek_path

        startJVM(
            getDefaultJVMPath(),
            '-ea',
            f'-Djava.class.path={self.ZEMBEREK_PATH}',
            convertStrings=False
        )
        TurkishSentenceExtractor: JClass = JClass('zemberek.tokenization.TurkishSentenceExtractor')
        self.sentence_extractor: TurkishSentenceExtractor = TurkishSentenceExtractor.DEFAULT

        TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
        WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')
        SentenceAnalysis: JClass = JClass('zemberek.morphology.analysis.SentenceAnalysis')
        Morpheme: JClass = JClass('zemberek.morphology.morphotactics.Morpheme')
        self.morphological_analyzer: TurkishMorphology = TurkishMorphology.createWithDefaults()

    def count_morphs(self, file, total_line_count, begin, end):
        counter = Counter()
        line_count = 0
        with open(file, encoding='utf-8') as f:
            for line in f:
                if line_count >= begin and line_count < end:
                    line = re.sub(r'(?<=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])[\’\'\'](?=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])', ' \'',
                                  line).strip()

                    for sent in self.sentence_extractor.fromParagraph(line):
                        sent_tokens = []
                        word_analysis_list: WordAnalysis = self.morphological_analyzer.analyzeSentence(JString(sent))
                        disambiguated_word_analysis: SentenceAnalysis = self.morphological_analyzer.disambiguate(
                            JString(sent), word_analysis_list).bestAnalysis()

                        for analysis in disambiguated_word_analysis:
                            morphemes_data = analysis.getMorphemeDataList()
                            idx = 0
                            for morpheme_data_piece in morphemes_data:
                                morpheme_data_piece = str(morpheme_data_piece)
                                if ":" in morpheme_data_piece:
                                    if idx == 0:
                                        sent_tokens.append(morpheme_data_piece.split(":")[0])
                                    else:
                                        sent_tokens.append("#" + morpheme_data_piece.split(":")[0])
                                    idx += 1
                        counter.update(sent_tokens)

                if line_count % 100 == 0:
                    sys.stdout.write('\r')
                    sys.stdout.write("{}/{} lines processed".format(line_count, total_line_count))
                    sys.stdout.flush()
                    # print("{}/{} lines processed".format(line_count,total_line_count))
                line_count += 1
        return counter

    def train(self,file,vocab_size=None,min_freq=configs.TOKENIZER_MIN_FREQ):
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

        print("Training elapsed time: ",time.time()-begin_time)
        return self.vocab

    def save(self,file_path):
        pickle.dump(self.vocab, open(file_path, "wb" ))

    def load(self,file_path):
        self.vocab = pickle.load(open(file_path, "rb" ) )

    def encode(self,text):
        text = re.sub(r'(?<=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])[\’\'\'](?=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])', ' \'', text).strip()

        encoded_ids = []
        for sent in self.sentence_extractor.fromParagraph(text):
            word_analysis_list: WordAnalysis = self.morphological_analyzer.analyzeSentence(JString(sent))
            disambiguated_word_analysis: SentenceAnalysis = self.morphological_analyzer.disambiguate(JString(sent),word_analysis_list).bestAnalysis()

            for analysis in disambiguated_word_analysis:
                morphemes_data = analysis.getMorphemeDataList()
                idx=0
                for morpheme_data_piece in morphemes_data:
                    morpheme_data_piece = str(morpheme_data_piece)
                    if ":"  in morpheme_data_piece:
                        if idx==0:
                            encoded_ids.append(self.vocab(morpheme_data_piece.split(":")[0]))
                        else:
                            encoded_ids.append(self.vocab("#"+morpheme_data_piece.split(":")[0]))
                        idx+=1

        return encoded_ids

    def tokenize(self,text):
        text = re.sub(r'(?<=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])[\’\'\'](?=[a-zA-Z0-9ğĞçCıIiİöÖŞşüÜ])', ' \'', text).strip()

        tokens = []
        for sent in self.sentence_extractor.fromParagraph(text):
            word_analysis_list: WordAnalysis = self.morphological_analyzer.analyzeSentence(JString(sent))
            disambiguated_word_analysis: SentenceAnalysis = self.morphological_analyzer.disambiguate(JString(sent),word_analysis_list).bestAnalysis()

            for analysis in disambiguated_word_analysis:
                morphemes_data = analysis.getMorphemeDataList()
                idx=0
                for morpheme_data_piece in morphemes_data:
                    morpheme_data_piece = str(morpheme_data_piece)
                    if ":"  in morpheme_data_piece:
                        if idx==0:
                            tokens.append(morpheme_data_piece.split(":")[0])
                        else:
                            tokens.append("#"+morpheme_data_piece.split(":")[0])
                        idx+=1

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab(token) for token in tokens]

    def decode(self,ids):
        text=""
        for id in ids:
            token = self.vocab.idx2word[id]
            if token.startswith('#'):
                text+=token[1:]
            else:
                text+=" "+token
        return text

    def print_ids(self,ids):
        print([self.vocab.idx2word[id] for id in ids])

if __name__=="__main__":
    ZEMBEREK_PATH = os.path.join('data', 'zemberek-full.jar')
    TRAIN_FILE_PATH= "data/test.tsv"
    TOKENIZER_PATH="files/tr_morph_50000"
    tr_morph_tokenizer = TurkishRootMorphologicalTokenizer(ZEMBEREK_PATH)
    #tr_morph_tokenizer.train(TRAIN_FILE_PATH,vocab_size=50000,min_freq=2)
    #tr_morph_tokenizer.save(TOKENIZER_PATH)
    tr_morph_tokenizer.load(TOKENIZER_PATH)
    tokens = tr_morph_tokenizer.tokenize("abimle beraber gözlükçüden gözlük aldım")
    ids = tr_morph_tokenizer.convert_tokens_to_ids(tokens)
    print(tokens)
    print(ids)
    tr_morph_tokenizer.print_ids(ids)
