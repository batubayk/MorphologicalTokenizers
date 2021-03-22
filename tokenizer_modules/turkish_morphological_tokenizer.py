import os
from pathlib import Path

from jpype import JClass, JString, getDefaultJVMPath, startJVM, isJVMStarted

from tokenizer_modules.tokenizer import Tokenizer
from tokenizer_modules.vocabulary import Vocabulary
package_directory = os.path.dirname(os.path.abspath(__file__))


class TurkishMorphologicalTokenizer(Tokenizer):

    def __init__(self, configs):
        self.configs = configs
        self.zemberek_path = self.configs['params']["zemberek_path"] if 'zemberek_path' in self.configs[
            'params'] else 'data/zemberek-full.jar'

        if not os.path.isabs(self.zemberek_path):
            self.zemberek_path = "/".join(package_directory.split('/')[:-1])+'/'+self.zemberek_path

        self.vocab = Vocabulary(self.configs['params']['special_tokens']) if 'special_tokens' in self.configs[
            'params'] else Vocabulary()

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

    def tokenize(self, text):
        tokens = []
        for sent in self.sentence_extractor.fromParagraph(text):
            word_analysis_list = self.morphological_analyzer.analyzeSentence(JString(sent))
            disambiguated_word_analysis = self.morphological_analyzer.disambiguate(JString(sent),
                                                                                   word_analysis_list).bestAnalysis()

            for analysis in disambiguated_word_analysis:
                tokens.extend(self.get_morphologically_disambiguated_tokens(analysis))

        return tokens

    def get_morphologically_disambiguated_tokens(self, analysis):
        morphemes_data = analysis.getMorphemeDataList()
        dictionary_item = analysis.getDictionaryItem()
        tokens = []
        morpheme_list = [str(morpheme.surface) for morpheme in morphemes_data if morpheme.surface != '']

        if len(morpheme_list) > 0:
            root = morpheme_list[0]
            tokens.append(root)
            suffix_list = morpheme_list[1:]

            if self.configs['params']['suffixes']['root_only']:
                return tokens

            if len(suffix_list) > 0:
                if self.configs['params']['suffixes']['combined']:
                    combined_suffix = ''.join(suffix_list)
                    tokens.append(self.configs['params']['token_affix'] + combined_suffix)
                elif self.configs['params']['suffixes']['last_only'] and not self.configs['params']['suffixes'][
                    'combined']:
                    tokens.append(self.configs['params']['token_affix'] + suffix_list[-1])
                else:
                    tokens.extend(
                        [self.configs['params']['token_affix'] + suffix for suffix in suffix_list])

        return tokens
