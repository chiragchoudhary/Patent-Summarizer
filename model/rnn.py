import tensorflow as tf
from tensorflow.python.layers.core import Dense
import os
import time
import numpy as np
from utils.processing import pad_sentence_batch


class RNNModel:
    def __init__(self, vocab, embeddings, epochs=10, batch_size=128, rnn_size=256, num_layers=2, learning_rate=0.001,
                 keep_probability=1.0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_probability = keep_probability
        self.vocab_to_int = vocab.vocab_to_int
        self.cost = None
        self.train_graph = None
        self.train_op = None
        self.embeddings = embeddings

    def model_inputs(self):
        """Create placeholders for inputs to the model"""

        input = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
        text_length = tf.placeholder(tf.int32, (None,), name='text_length')

        return input, targets, lr, keep_prob, summary_length, max_summary_length, text_length

    def get_batches(self, summaries, texts, batch_size):
        """Batch summaries, texts, and the lengths of their sentences together."""

        for batch_i in range(0, len(texts) // batch_size):
            start_i = batch_i * batch_size
            summaries_batch = summaries[start_i:start_i + batch_size]
            texts_batch = texts[start_i:start_i + batch_size]
            pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch, self.vocab_to_int))
            pad_texts_batch = np.array(pad_sentence_batch(texts_batch, self.vocab_to_int))

            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))

            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))

            yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

    def process_encoding_input(self, target_data, batch_size):
        """Remove the last word id from each batch and concat the <GO> to the beginning of each batch"""

        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], self.vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def encoding_layer(self, rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
        """Create the encoding layer."""

        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer)):
                cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

                cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

                enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length,
                                                                        dtype=tf.float32)

        # Join outputs since we are using a bidirectional RNN
        enc_output = tf.concat(enc_output, 2)

        return enc_output, enc_state

    def training_decoding_layer(self, dec_embed_input, summary_length, dec_cell, initial_state, output_layer,
                                max_summary_length):
        """Create the training logits"""

        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer)

        training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                  output_time_major=False,
                                                                  impute_finished=True,
                                                                  maximum_iterations=max_summary_length)
        return training_logits

    def inference_decoding_layer(self, embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                                 max_summary_length, batch_size):
        """Create the inference logits"""

        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                   output_time_major=False,
                                                                   impute_finished=True,
                                                                   maximum_iterations=max_summary_length)

        return inference_logits

    def decoding_layer(self, dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length,
                       summary_length, max_summary_length, rnn_size, keep_prob, batch_size, num_layers):
        """Create the decoding cell and attention for the training and inference decoding layers."""

        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                gru = tf.contrib.rnn.GRUCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(gru,
                                                         input_keep_prob=keep_prob)

        output_layer = Dense(vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                         enc_output,
                                                         text_length,
                                                         normalize=False,
                                                         name='BahdanauAttention')

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                       attn_mech,
                                                       rnn_size)

        # initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state[0], dec_cell.zero_state(batch_size,tf.float32))

        initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        with tf.variable_scope("decode"):
            training_logits = self.training_decoding_layer(dec_embed_input,
                                                           summary_length,
                                                           dec_cell,
                                                           initial_state,
                                                           output_layer,
                                                           max_summary_length)
        with tf.variable_scope("decode", reuse=True):
            inference_logits = self.inference_decoding_layer(embeddings,
                                                             self.vocab_to_int['<GO>'],
                                                             self.vocab_to_int['<EOS>'],
                                                             dec_cell,
                                                             initial_state,
                                                             output_layer,
                                                             max_summary_length,
                                                             batch_size)

        return training_logits, inference_logits

    def seq2seq_model(self, input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                      vocab_size, rnn_size, num_layers, batch_size):
        """Use the previous functions to create the training and inference logits."""

        # Use GloVe's embeddings and the newly created ones as our embeddings
        embeddings = self.embeddings

        enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
        enc_output, enc_state = self.encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)

        dec_input = self.process_encoding_input(target_data, batch_size)
        dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

        training_logits, inference_logits = self.decoding_layer(dec_embed_input,
                                                                embeddings,
                                                                enc_output,
                                                                enc_state,
                                                                vocab_size,
                                                                text_length,
                                                                summary_length,
                                                                max_summary_length,
                                                                rnn_size,
                                                                keep_prob,
                                                                batch_size,
                                                                num_layers)

        return training_logits, inference_logits

    def build_graph(self):
        """Build tensorflow training graph"""

        self.train_graph = tf.Graph()
        # Set the graph to default to ensure that it is ready for training
        with self.train_graph.as_default():
            # Load the model inputs
            self.input, self.targets, self.lr, self.keep_prob, self.summary_length, self.max_summary_length, self.text_length = self.model_inputs()

            # Create the training and inference logits
            training_logits, inference_logits = self.seq2seq_model(tf.reverse(self.input, [-1]),
                                                                   self.targets,
                                                                   self.keep_prob,
                                                                   self.text_length,
                                                                   self.summary_length,
                                                                   self.max_summary_length,
                                                                   len(self.vocab_to_int) + 1,
                                                                   self.rnn_size,
                                                                   self.num_layers,
                                                                   self.batch_size)

            # Create tensors for the training logits and inference logits
            training_logits = tf.identity(training_logits.rnn_output, 'logits')
            inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

            # Create the weights for sequence_loss
            masks = tf.sequence_mask(self.summary_length, self.max_summary_length, dtype=tf.float32, name='masks')

            with tf.name_scope("optimization"):
                # Loss function
                self.cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    self.targets,
                    masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                # Gradient clipping
                gradients = optimizer.compute_gradients(self.cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gradients)

        print("Graph is built.")

    def train_model(self, model_name, titles, abstracts, load=False):
        """Trains the Model"""

        learning_rate_decay = 0.95
        min_learning_rate = 0.0005
        display_step = 20  # Check training loss after every 20 batches
        stop_early = 0
        stop = 3  # If the update loss does not decrease in 3 consecutive update checks, stop training
        per_epoch = 3  # Make 3 update checks per epoch
        update_check = (len(abstracts) // self.batch_size // per_epoch) - 1
        print(update_check)
        # update_loss = 0
        # batch_loss = 0
        summary_update_loss = []  # Record the update losses for saving improvements in the model
        checkpoint = "{}.ckpt".format(model_name)
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # If we want to continue training a previous session
            if load:
                loader = tf.train.import_meta_graph(os.path.join(os.getcwd(), model_name, '{}.meta'.format(model_name)))
                loader.restore(sess, os.path.join(os.getcwd(), model_name, checkpoint))

            for epoch_i in range(1, self.epochs + 1):
                update_loss = 0
                batch_loss = 0
                for batch_i, (titles_batch, abstracts_batch, titles_lengths, abstracts_lengths) in enumerate(
                        self.get_batches(titles, abstracts, self.batch_size)):
                    start_time = time.time()
                    _, loss = sess.run(
                        [self.train_op, self.cost],
                        {self.input: abstracts_batch,
                         self.targets: titles_batch,
                         self.lr: self.learning_rate,
                         self.summary_length: titles_lengths,
                         self.text_length: abstracts_lengths,
                         self.keep_prob: self.keep_probability})

                    batch_loss += loss
                    update_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time

                    if batch_i % display_step == 0 and batch_i > 0:
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                              .format(epoch_i,
                                      self.epochs,
                                      batch_i,
                                      len(abstracts) // self.batch_size,
                                      batch_loss / display_step,
                                      batch_time * display_step))
                        batch_loss = 0

                    if batch_i % update_check == 0 and batch_i > 0:
                        print("Average loss for this update:", round(update_loss / update_check, 3))
                        summary_update_loss.append(update_loss)

                        # If the update loss is at a new minimum, save the model
                        if update_loss <= min(summary_update_loss):
                            print('New Record!')
                            stop_early = 0
                            saver = tf.train.Saver()
                            saver.save(sess, os.path.join(os.getcwd(), checkpoint))

                        else:
                            print("No Improvement.")
                            stop_early += 1
                            if stop_early == stop:
                                break
                        update_loss = 0

                # Reduce learning rate, but not below its minimum value
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

                if stop_early == stop:
                    print("Stopping Training.")
                    break
