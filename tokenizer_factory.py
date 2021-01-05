import os, re, pickle, sys, math, time
package_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,package_directory)
import json
import codecs
from pydoc import locate
from pathlib import Path

class Tokenizers():
    def create(config_file):
        path = Path(config_file)
        with codecs.open(config_file, mode='r', encoding='utf-8') as json_file:
            configs = json.load(json_file)
            file_name_pattern = '{file_name}'
            model_path = configs['params']['model']['path']
            if file_name_pattern in model_path:
                configs['params']['model']['path'] = model_path.format(file_name=path.stem)

        tokenizer_class = locate(configs["tokenizer_class"])
        tokenizer = tokenizer_class(configs)
        return tokenizer
