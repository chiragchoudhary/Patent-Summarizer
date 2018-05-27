class Vocab:
    """A simple vocabulary class."""
    def __init__(self):
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self._vocab_size = 0

    def word2idx(self, word):
        """Maps a word to its corresponding index in the Vocabulary."""

        if word in self.vocab_to_int:
            return self.vocab_to_int[word]
        else:
            return self._vocab_size

    def idx2word(self, index):
        """Maps an index to its corresponding word in the Vocabulary."""

        if index < self._vocab_size:
            return self.int_to_vocab[index]
        else:
            raise KeyError('Index larger than Vocabulary size.')

    def add_word(self, word):
        """Adds a new word to the vocabulary."""

        if word not in self.vocab_to_int:
            self.vocab_to_int[word] = self._vocab_size
            self.int_to_vocab[self._vocab_size] = word
            self._vocab_size += 1

    def __len__(self):
        """Return number of words in the Vocabulary"""

        return self._vocab_size
