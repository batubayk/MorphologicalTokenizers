import re, pickle, sys, time
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, isJVMStarted
from collections import Counter
from vocabulary import Vocabulary
import os

from pathlib import Path
import codecs
from tqdm import tqdm

class TurkishAutoMorphologicalTokenizer:

    def __init__(self,configs):
        self.configs = configs
        self.zemberek_path = self.configs['params']["zemberek_path"] if 'zemberek_path' in self.configs['params'] else 'data/zemberek-full.jar'
        self.vocab = Vocabulary(self.configs['params']['special_tokens']) if 'special_tokens' in self.configs['params'] else Vocabulary()

        if isJVMStarted() == False:
            startJVM(
                getDefaultJVMPath(),
                '-ea',
                f'-Djava.class.path={self.zemberek_path}',
                convertStrings=False
            )
        TurkishSentenceExtractor: JClass = JClass('zemberek.tokenization.TurkishSentenceExtractor')
        self.sentence_extractor: TurkishSentenceExtractor = TurkishSentenceExtractor.DEFAULT

        TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
        WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')
        SentenceAnalysis: JClass = JClass('zemberek.morphology.analysis.SentenceAnalysis')
        Morpheme: JClass = JClass('zemberek.morphology.morphotactics.Morpheme')
        self.morphological_analyzer: TurkishMorphology = TurkishMorphology.createWithDefaults()
        if Path(self.configs['params']['model']['path']).exists():
            self.load()

    def count_lines(self, file_name):
        line_count = 0
        with codecs.open(file_name, mode='r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
        return line_count

    def count_morphs(self):
        counter = Counter()

        input_file_paths = [str(x) for x in Path(self.configs['params']['input_dir']).glob(self.configs['params']['input_file_pattern'])]
        total_line_count = 0

        file_num = 0
        total_file_count = len(input_file_paths)
        for input_file_path in input_file_paths:
            file_num += 1
            cur_file_total_line_count = self.count_lines(input_file_path)
            sys.stdout.write('\n')
            with codecs.open(input_file_path, encoding='utf-8') as f:

                for line in tqdm(f, desc="File {}/{} processing. {}".format(file_num, total_file_count, input_file_path), total=cur_file_total_line_count, position=0, leave=True):
                    total_line_count += 1
                    line = self.normalize_text(line)

                    sent_tokens = self.tokenize(line)
                    counter.update(sent_tokens)
        sys.stdout.write("\r\n")
        sys.stdout.write("Totally {} lines processed".format(total_line_count))
        return counter

    def train(self):
        self.vocab = Vocabulary(self.configs['params']['special_tokens']) if 'special_tokens' in self.configs['params'] else Vocabulary()
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

        print("Training elapsed time: ",time.time()-begin_time)
        return self.vocab


    def save(self):
        output_file_path = Path(self.configs['params']['model']['path'])
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        if output_file_path.exists() and self.configs['params']['model'].get('overwrite', False):
            output_file_path.unlink()
        pickle.dump(self.vocab, open(output_file_path, "wb" ))

    def load(self):
        model_file_path = Path(self.configs['params']['model']['path'])
        self.vocab = pickle.load(open(model_file_path, "rb" ) )

    def normalize_text(self, text):
        escape_pattern = re.compile(self.configs['params']['escape_pattern'])
        normalized_text = re.sub(escape_pattern, ' \'', text).strip()
        return normalized_text

    def encode(self,text):
        text = self.normalize_text(text)
        tokens = self.tokenize(text)
        encode_ids = [self.vocab(token) for token in tokens]
        return encode_ids

    def tokenize(self, text):
        text = self.normalize_text(text)

        tokens = []
        for sent in self.sentence_extractor.fromParagraph(text):
            word_analysis_list = self.morphological_analyzer.analyzeSentence(JString(sent))
            disambiguated_word_analysis = self.morphological_analyzer.disambiguate(JString(sent),word_analysis_list).bestAnalysis()

            for analysis in disambiguated_word_analysis:
                tmp_tokens = self.get_tokens_by_morphological_analysis(analysis)
                tokens.extend(tmp_tokens)

        return tokens

    def get_tokens_by_morphological_analysis(self, analysis):
        morphemes_data = analysis.getMorphemeDataList()
        dictionary_item = analysis.getDictionaryItem()
        tokens = []
        morpheme_list = [str(morpheme.surface) for morpheme in morphemes_data if morpheme.surface != '']
        if self.configs['params'].get("root",{}).get("use_lemma", False):
            root = str(dictionary_item.lemma)
            if self.configs['params']["root"]["remove_infinitive"] \
                    and str(dictionary_item.primaryPos).lower() == "verb" \
                    and root.lower().endswith("mek") or root.endswith("mak"):
                root = root[:-3]
        else:
            root = morpheme_list[0]

        root_max_len = self.configs['params'].get("root", {}).get("max_len", -1)
        if root_max_len > 0 and len(root)>root_max_len:
            root = root[:root_max_len]

        tokens.append(root)

        suffix_config = self.configs['params'].get("suffixes", None)
        if suffix_config is None:
            # no suffix
            return tokens

        if len(morpheme_list) > 1:
            suffix_list = [suffix for suffix in morpheme_list[1:]]
            if suffix_config.get("combine", False):
                combined_suffix = ''.join(suffix_list)
                tokens.append(self.configs['params']['token_affix'] + combined_suffix)
            elif suffix_config.get("last_only", False) and not suffix_config.get("combine", False) :
                tokens.append(self.configs['params']['token_affix'] + suffix_list[-1])
            else:
                tokens.extend([self.configs['params']['token_affix'] + suffix for suffix in suffix_list])
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab(token) for token in tokens]

    def decode(self,ids):
        text=""
        for id in ids:
            token = self.vocab.idx2word[id]
            if token.startswith(self.configs['params']['token_affix']):
                text+=token[len(self.configs['params']['token_affix']):]
            else:
                text+=" "+token
        return text

    def print_ids(self,ids):
        print([self.vocab.idx2word[id] for id in ids])