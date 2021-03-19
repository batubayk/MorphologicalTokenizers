import os
import sys

package_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_directory)
import json
from pydoc import locate
from pathlib import Path


class Tokenizers():
    def create(config_file):
        path = Path(config_file)
        with open(config_file, mode='r', encoding='utf-8') as json_file:
            configs = json.load(json_file)

        tokenizer_class = locate(configs["tokenizer_class"])
        tokenizer = tokenizer_class(configs)
        return tokenizer
