import numpy as np
import sys
from utils.processing import *
from model.rnn import RNNModel
from utils.vocab import Vocab

pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 100)

data_dir = os.path.join(os.getcwd(), 'data')
train_data = os.path.join(data_dir, 'patent_train.csv')
val_data = os.path.join(data_dir, 'patent_val.csv')
test_data = os.path.join(data_dir, 'patent_test.csv')

model_name = sys.argv[1]
patent_titles, patent_abstracts = get_titles_and_abstracts(train_data)

# Find the number of times each word was used and the size of the vocabulary
word_counts = count_words(patent_titles)
word_counts = count_words(patent_abstracts, word_counts)
print('Size of Vocabulary:', len(word_counts))

# Fetch GloVe word embeddings
glove_embeddings = os.path.join(os.getcwd(), 'data', 'glove.6B.300d.txt')
word_embeddings = {}
with open(glove_embeddings) as glove:
    for embedding in glove:
        values = embedding.split(' ')
        word = values[0]
        word_embeddings[word] = np.asarray(values[1:], dtype='float32')

print('Total GloVe Word embeddings:', len(word_embeddings))

# Find the number of words in vocabulary missing from GloVe
missing_words = 0

for word, count in word_counts.items():
    if word not in word_embeddings:
        missing_words += 1

missing_ratio = round((1.0 * missing_words / len(word_counts)) * 100, 4)

print('Number of words missing from GloVe:', missing_words)
print('Percent of words that are missing from vocabulary: {}%'.format(missing_ratio))

# Limit the vocab that we will use to words that appear >= threshold or are in GloVe
vocab = Vocab()

# Dictionary to convert words to integers
threshold = 10

for word, count in word_counts.items():
    if count >= threshold or word in glove_embeddings:
        vocab.add_word(word)

# Special tokens that will be added to our vocab
codes = ["<UNK>", "<EOS>", "<GO>", "<PAD>"]

# Add codes to vocab
for code in codes:
    vocab.add_word(code)

usage_ratio = round(1.0 * len(vocab) / len(word_counts) + 4, 4) * 100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab))
print("Percent of words we will use: {}%".format(usage_ratio))

# save vocabulary

# Need to use 300 for embedding dimensions to match GloVe's vectors.
embedding_dim = 300
num_words = len(vocab)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((num_words, embedding_dim), dtype=np.float32)
for word, i in vocab.vocab_to_int.items():
    if word in glove_embeddings:
        word_embedding_matrix[i] = word_embeddings[word]
    else:
        # If word not in GloVe, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        word_embeddings[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_titles, word_count, unk_count = convert_to_ints(patent_titles, vocab.vocab_to_int, word_count, unk_count)
int_abstracts, word_count, unk_count = convert_to_ints(patent_abstracts, vocab.vocab_to_int, word_count, unk_count, eos=True)

unk_percent = round(1.0 * unk_count / word_count, 4) * 100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

length_titles = create_lengths(int_titles)
length_abstracts = create_lengths(int_abstracts)

print("Patent titles description:")
print(length_titles.describe())
print()
print("Patent abstract description:")
print(length_abstracts.describe())

# Inspect the length of abstracts
print ("Patent abstracts length percentile summary:")
print(np.percentile(length_abstracts.counts, 90))
print(np.percentile(length_abstracts.counts, 95))
print(np.percentile(length_abstracts.counts, 99))

# Inspect the length of titles
print ("Patent titles length percentile summary:")
print(np.percentile(length_titles.counts, 90))
print(np.percentile(length_titles.counts, 95))
print(np.percentile(length_titles.counts, 99))

# Sort the abstracts and titles by the length of the abstract, shortest to longest
# Limit the length of titles and abstracts based on the min and max ranges.
# Remove abstracts that include too many UNKs
sorted_titles, sorted_abstracts = sort_text(int_titles, int_abstracts, length_abstracts, vocab.vocab_to_int,
                                            max_text_length=200,
                                            max_summary_length=20,
                                            min_length=5,
                                            unk_text_limit=5,
                                            unk_summary_limit=3)

# Set the hyper-parameters
epochs = 25
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75
rnn = RNNModel(vocab, word_embedding_matrix, epochs, batch_size, rnn_size, num_layers, learning_rate, keep_probability)

# Build the graph
rnn.build_graph()

print(len(sorted_titles))
print("The shortest abstract length:", len(sorted_abstracts[0]))
print("The longest abstract length:", len(sorted_abstracts[-1]))

# Train the RNN model and store it
rnn.train_model(model_name, sorted_titles, sorted_abstracts)

# store the vocabulary
store_vocab(vocab, model_name)
