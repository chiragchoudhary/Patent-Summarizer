import os
import numpy as np
import tensorflow as tf
from rouge import Rouge
import argparse

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from utils.processing import get_titles_and_abstracts, text_to_seq, load_vocab


def main(model, data, vocab):
    """Evaluates the performance of model on given data."""

    batch_size = 64
    patent_titles, patent_abstracts = get_titles_and_abstracts(data)
    output_patent_titles = []
    checkpoint = "{}.ckpt".format(model)
    metafile = checkpoint+'.meta'
    pad = vocab.word2idx("<PAD>")
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(os.path.join(os.getcwd(), model, metafile))
        loader.restore(sess, os.path.join(os.getcwd(), model, checkpoint))

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        text_length = loaded_graph.get_tensor_by_name('text_length:0')
        summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        # Multiply by batch_size to match the model's input parameters
        for i in range(len(patent_abstracts)):
            text = text_to_seq(patent_abstracts[i], vocab.vocab_to_int)
            answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                              summary_length: [np.random.randint(10, 20)],
                                              text_length: [len(text)] * batch_size,
                                              keep_prob: 1.0})[0]
            temp = " ".join([vocab.int_to_vocab[j] for j in answer_logits if j != pad])
            output_patent_titles.append(temp)
            if (i+1) % 50 == 0:
                print("Current iteration", i)

    rouge = Rouge()
    scores = rouge.get_scores(patent_titles, output_patent_titles, avg=True)
    print("Rouge score: ", scores)

    # create a list of list of expected patent titles, for computing BLEU score
    patent_titles_list = []
    for i in range(len(patent_titles)):
        patent_titles_list.append([patent_titles[i]])

    cc = SmoothingFunction()
    bleu_score = sentence_bleu(patent_titles_list, output_patent_titles, smoothing_function=cc.method4)
    print("BLEU Score:", bleu_score)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="The Model to be evaluated.")
    parser.add_argument("data", help="File containing patent data in CSV format.")
    args = parser.parse_args()
    model = args.model
    data = os.path.join(os.getcwd(), "data", args.data)
    vocab = load_vocab(model)
    main(model, data, vocab)
