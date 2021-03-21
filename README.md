# MorphologicalTokenizers


## Installation
pip install -r requirements.txt

For Hungarian tokenizers please follow the installation guides for emMorphPy (https://github.com/dlt-rilmta/emmorphpy)
and PurePOSPy (https://github.com/dlt-rilmta/purepospy).
The HFST binaries are included under data folder so there is no need to install it seperately.

Note: If on Mac OS makefile for Purepospy will not work so following the steps below should do the job; <br />
cd purepospy/ <br />
make __extra-deps <br />
pip install -r requirements.txt <br />

## Adding Other Tokenizers
The structure is intented to be flexible so that other tokenizers can be added without many change.
The tokenizer class under tokenizer_modules is an abstract class that has already implemented most of the methods. 
To add your custom tokenizer
1) Extending the tokenizer class and implement necessary methods
2) Add your configuration file under configs
3) The custom tokenizer can then be used as Tokenizer.create('configs/my-custom-tokenizer-config.json')
