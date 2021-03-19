from pathlib import Path

import nltk
from emmorphpy.emmorphpy import EmMorphPy
from purepospy.purepospy import PurePOS

from tokenizer_modules.tokenizer import Tokenizer
from tokenizer_modules.vocabulary import Vocabulary


class HungarianMorphologicalTokenizer(Tokenizer):

    def __init__(self, configs):
        self.configs = configs
        self.vocab = Vocabulary(self.configs['params']['special_tokens']) if 'special_tokens' in self.configs[
            'params'] else Vocabulary()
        self.pos_tagger = PurePOS(self.configs['params']['pos_tagger_model_path'])
        self.morph_analyzer = EmMorphPy(hfst_lookup=self.configs['params']['hfst_lookup'])
        if Path(self.configs['params']['model']['path']).exists():
            self.load()

    def get_morphologically_disambiguated_tokens(self, sent):
        tokens = []
        for token_pos in self.pos_tagger.tag_sentence(sent.split()):
            morphemes_data = self.morph_analyzer.dstem(token_pos[0])
            disambiguated_morpheme_data = None
            for analysis in morphemes_data:
                if analysis[1] == token_pos[2]:
                    disambiguated_morpheme_data = analysis[3]
                    break

            morpheme_list = []
            if disambiguated_morpheme_data is None:
                morpheme_list.append(token_pos[0])
            elif token_pos[2] == '[Punct]':
                morpheme_list.append(token_pos[0])
            else:
                morphemes = disambiguated_morpheme_data.split('+')
                for morpheme in morphemes:
                    if "=" in morpheme:
                        morpheme = morpheme[morpheme.index("=") + 1:].strip()
                    elif "[" in morpheme:
                        morpheme = morpheme[:morpheme.index("[")].strip()
                    else:
                        pass

                    if morpheme != '':
                        morpheme_list.append(morpheme)

            if len(morpheme_list) > 0:
                root = morpheme_list[0]
                tokens.append(root)
                suffix_list = morpheme_list[1:]

                if self.configs['params']['suffixes']['root_only']:
                    continue

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

    def tokenize(self, text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            tokens.extend(self.get_morphologically_disambiguated_tokens(sent))

        return tokens
