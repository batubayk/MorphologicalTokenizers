import configs

class Vocabulary(object):
    def __init__(self, special_tokens={}):
        self.special_tokens = special_tokens
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        for token, id in special_tokens.items():
            self.word2idx[token] = id
            self.idx2word[id] = token

    def add_word(self, word):
        if word not in self.word2idx:
            #If special token ids collapse, increment current idx
            if self.idx in self.special_tokens.values():
                self.idx+=1
                return self.add_word(word)

            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.special_tokens['[UNK]']]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
