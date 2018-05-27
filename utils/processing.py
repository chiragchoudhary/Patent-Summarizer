import re
import os
import pandas as pd
from nltk.corpus import stopwords
import pickle


def clean_text(text):
    """Convert words to lowercase, remove special characters and stopwords."""

    text = text.lower()
    text = re.sub(r'[_"\-;%()|+&=*.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\'', '', text)

    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if w not in stops]
    text = " ".join(text)

    return text


def count_words(text, vocab=None):
    """Count the number of occurrences of each word in text."""

    if vocab is None:
        vocab = {}

    for sentence in text:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


def convert_to_ints(text, vocab_to_int, word_count=0, unk_count=0, eos=False):
    """Convert words in text to integers.

    If word is not in vocabulary, replace by <UNK> token.
    Total the number of words and UNKs.
    Add <EOS> token to the end of texts"""
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


def create_lengths(text):
    """Create a data frame of the sentence lengths from a text"""

    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


def unk_counter(sentence, vocab_to_int):
    """Counts the number of unknown tokens in a sentence."""

    unk_count = 0
    unk_id = vocab_to_int["<UNK>"]
    for word in sentence:
        if word == unk_id:
            unk_count += 1
    return unk_count


def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""

    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def text_to_seq(text, vocab_to_int):
    """Prepare the text for the model"""

    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


def sort_text(int_titles, int_abstracts, length_abstracts, vocab_to_int, max_text_length=100, max_summary_length=20,
              min_length=0, unk_text_limit=5, unk_summary_limit=3):
    """Sort the abstracts and titles by the length of the abstract, shortest to longest.

    Limit the length of titles and abstracts based on the min and max ranges.
    Remove abstracts that include too many UNKs."""
    sorted_titles = []
    sorted_abstracts = []
    for length in range(min(length_abstracts.counts), max_text_length):
        for count, words in enumerate(int_titles):
            if (min_length <= len(int_titles[count]) <= max_summary_length and
                    len(int_abstracts[count]) >= min_length and
                    unk_counter(int_titles[count], vocab_to_int) <= unk_summary_limit and
                    unk_counter(int_abstracts[count], vocab_to_int) <= unk_text_limit and
                    length == len(int_abstracts[count])):
                sorted_titles.append(int_titles[count])
                sorted_abstracts.append(int_abstracts[count])

    return sorted_titles, sorted_abstracts


def get_titles_and_abstracts(data):
    """Reads a data file, and returns clean titles and abstracts."""

    patents = pd.read_csv(data, usecols=['title', 'abstract'])

    patent_titles = []
    for title in patents.title:
        patent_titles.append(clean_text(title))

    patent_abstracts = []
    for abstract in patents.abstract:
        patent_abstracts.append(clean_text(abstract))

    return patent_titles, patent_abstracts


def store_vocab(vocab, model):
    """Stores the vocabulary for use during evaluation."""

    path = os.path.join(os.getcwd(), model, "{}.vocab".format(model))
    with open(path, "w+") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


def load_vocab(model):
    """Loads the vocabulary from file."""
    with open(os.path.join(os.getcwd(), model, "{}.vocab".format(model))) as f:
        vocab = pickle.load(f)
    return vocab
