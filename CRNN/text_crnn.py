# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf


class TextCRNN(object):
    """A CRNN for text classification."""

    def __init__(
            self, sequence_length, vocab_size, embedding_type, embedding_size, filter_sizes, num_filters,
            lstm_hidden_size, fc_hidden_size, num_classes, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_front = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_front")
        self.input_x_behind = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_behind")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        def _linear(input_, output_size, initializer=None, scope="SimpleLinear"):
            """
            Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k].

            Args:
                input_: a tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                initializer: The initializer.
                scope: VariableScope for the created subgraph; defaults to "SimpleLinear".
            Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
            """

            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]

            # Now the computation.
            with tf.variable_scope(scope):
                W = tf.get_variable("W", [input_size, output_size], dtype=input_.dtype)
                b = tf.get_variable("b", [output_size], dtype=input_.dtype, initializer=initializer)

            return tf.nn.xw_plus_b(input_, W, b)

        def _highway_layer(input_, size, num_layers=1, bias=-2.0):
            """
            Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wx + b); h = relu(W'x + b')
            z = t * h + (1 - t) * x
            where t is transform gate, and (1 - t) is carry gate.
            """

            for idx in range(num_layers):
                h = tf.nn.relu(_linear(input_, size, scope=("highway_h_{0}".format(idx))))
                t = tf.sigmoid(_linear(input_, size, initializer=tf.constant_initializer(bias),
                                       scope=("highway_t_{0}".format(idx))))
                output = t * h + (1. - t) * input_
                input_ = output

            return output

        # Embedding Layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            self.embedded_sentence_front = tf.nn.embedding_lookup(self.embedding, self.input_x_front)
            self.embedded_sentence_behind = tf.nn.embedding_lookup(self.embedding, self.input_x_behind)
            self.embedded_sentence_expanded_front = tf.expand_dims(self.embedded_sentence_front, axis=-1)
            self.embedded_sentence_expanded_behind = tf.expand_dims(self.embedded_sentence_behind, axis=-1)

        # Create a convolution + max-pool layer for each filter size
        pooled_outputs_front = []
        pooled_outputs_behind = []

        for filter_size in filter_sizes:
            with tf.name_scope("conv-filter{0}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, dtype=tf.float32), name="W")
                b = tf.Variable(tf.constant(value=0.1, shape=[num_filters], dtype=tf.float32), name="b")
                conv_front = tf.nn.conv2d(
                    self.embedded_sentence_expanded_front,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_front")

                conv_behind = tf.nn.conv2d(
                    self.embedded_sentence_expanded_behind,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_behind")

                # Batch Normalization Layer
                conv_bn_front = tf.layers.batch_normalization(tf.nn.bias_add(conv_front, b), training=self.is_training)
                conv_bn_behind = tf.layers.batch_normalization(tf.nn.bias_add(conv_behind, b), training=self.is_training)

                # Apply non-linearity
                conv_out_front = tf.nn.relu(conv_bn_front, name="relu_front")
                conv_out_behind = tf.nn.relu(conv_bn_behind, name="relu_behind")

            with tf.name_scope("pool-filter{0}".format(filter_size)):
                # Max-pooling over the outputs
                pooled_front = tf.nn.max_pool(
                    conv_out_front,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_front")

                pooled_behind = tf.nn.max_pool(
                    conv_out_behind,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_behind")

            pooled_outputs_front.append(pooled_front)
            pooled_outputs_behind.append(pooled_behind)

        # Combine all the pooled features
        pool_flat_outputs_front = []
        pool_flat_outputs_behind = []

        for i in pooled_outputs_front:
            pool_flat = tf.reshape(i, shape=[-1, 1, num_filters])
            pool_flat = tf.nn.dropout(pool_flat, self.dropout_keep_prob)
            pool_flat_outputs_front.append(pool_flat)

        for i in pooled_outputs_behind:
            pool_flat = tf.reshape(i, shape=[-1, 1, num_filters])
            pool_flat = tf.nn.dropout(pool_flat, self.dropout_keep_prob)
            pool_flat_outputs_behind.append(pool_flat)

        lstm_outputs_front = []
        lstm_outputs_behind = []

        # Bi-LSTM Layer
        for i in range(len(pool_flat_outputs_front)):
            with tf.variable_scope("Bi-lstm-{0}".format(i)):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # forward direction cell
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # backward direction cell
                if self.dropout_keep_prob is not None:
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

                # Creates a dynamic bidirectional recurrent neural network
                # shape of `outputs`: tuple -> (outputs_fw, outputs_bw)
                # shape of `outputs_fw`: [batch_size, sequence_length, lstm_hidden_size]

                # shape of `state`: tuple -> (outputs_state_fw, output_state_bw)
                # shape of `outputs_state_fw`: tuple -> (c, h) c: memory cell; h: hidden state

                outputs_front, state_front = tf.nn.bidirectional_dynamic_rnn(
                    lstm_fw_cell, lstm_bw_cell, pool_flat_outputs_front[i], dtype=tf.float32)
                outputs_behind, state_behind = tf.nn.bidirectional_dynamic_rnn(
                    lstm_fw_cell, lstm_bw_cell, pool_flat_outputs_behind[i], dtype=tf.float32)

                # Concat output
                # [batch_size, sequence_length, lstm_hidden_size * 2]
                lstm_concat_front = tf.concat(outputs_front, axis=2)
                lstm_concat_behind = tf.concat(outputs_behind, axis=2)

                # [batch_size, lstm_hidden_size * 2]
                lstm_out_front = tf.reduce_mean(lstm_concat_front, axis=1)
                lstm_out_behind = tf.reduce_mean(lstm_concat_behind, axis=1)

                # shape of `lstm_outputs`: list -> len(filter_sizes) * [batch_size, lstm_hidden_size * 2]
                lstm_outputs_front.append(lstm_out_front)
                lstm_outputs_behind.append(lstm_out_behind)

        # [batch_size, lstm_hidden_size * 2 * len(filter_sizes)]
        self.lstm_out_front = tf.concat(lstm_outputs_front, axis=1)
        self.lstm_out_behind = tf.concat(lstm_outputs_behind, axis=1)

        # [batch_size, lstm_hidden_size * 2 * len(filter_sizes) * 2]
        self.lstm_out_combine = tf.concat([self.lstm_out_front, self.lstm_out_behind], axis=1)

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[lstm_hidden_size * 2 * len(filter_sizes) * 2, fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
            self.fc = tf.nn.xw_plus_b(self.lstm_out_combine, W, b)

            # Batch Normalization Layer
            self.fc_bn = tf.layers.batch_normalization(self.fc, training=self.is_training)

            # Apply non-linearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        with tf.name_scope("highway"):
            self.highway = _highway_layer(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.softmax_scores = tf.nn.softmax(self.logits, name="softmax_scores")
            self.topKPreds = tf.nn.top_k(self.softmax_scores, k=1, sorted=True, name="topKPreds")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_mean(losses, name="softmax_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")