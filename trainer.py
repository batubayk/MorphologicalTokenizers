import argparse
from tokenizer_factory import Tokenizers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factory class for morphological tokenizers.')
    parser.add_argument('--config_file', default=None, required=True,
                        help='The path to the config file of the tokenizer')

    args = parser.parse_args()
    tokenizer = Tokenizers.create(args.config_file)
    tokenizer.train()
    tokenizer.save()
    ids = tokenizer.encode("atatürk bu sabah yazı yazıyordu .")
    tokenizer.print_ids(ids)
    print(tokenizer.decode(ids))
    print()