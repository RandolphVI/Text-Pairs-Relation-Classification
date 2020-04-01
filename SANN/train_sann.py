# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging

sys.path.append('../')
logging.getLogger('tensorflow').disabled = True

import numpy as np
import tensorflow as tf

from tensorboard.plugins import projector
from text_sann import TextSANN
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

args = parser.parameter_parser()
OPTION = dh._option(pattern=0)
logger = dh.logger_fn("tflog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))


def train_sann():
    """Training RNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    logger.info("Data processing...")
    train_data = dh.load_data_and_labels(args.train_file, args.embedding_dim)
    validation_data = dh.load_data_and_labels(args.validation_file, args.embedding_dim)

    logger.info("Data padding...")
    x_train_front, x_train_behind, y_train = dh.pad_data(train_data, args.pad_seq_len)
    x_validation_front, x_validation_behind, y_validation = dh.pad_data(validation_data, args.pad_seq_len)

    # Build vocabulary
    VOCAB_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(args.embedding_dim, args.word2vec_file)

    # Build a graph and sann object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            sann = TextSANN(
                sequence_length=args.pad_seq_len,
                vocab_size=VOCAB_SIZE,
                embedding_type=args.embedding_type,
                embedding_size=args.embedding_dim,
                lstm_hidden_size=args.lstm_dim,
                attention_unit_size=args.attention_dim,
                attention_hops_size=args.attention_hops_dim,
                fc_hidden_size=args.fc_dim,
                num_classes=y_train.shape[1],
                l2_reg_lambda=args.l2_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                           global_step=sann.global_step, decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(sann.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=args.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=sann.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            out_dir = dh.get_out_dir(OPTION, logger)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", sann.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if OPTION == 'R':
                # Load sann model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = "embedding"
                embedding_conf.metadata_path = FLAGS.metadata_file

                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(sann.global_step)

            def train_step(x_batch_front, x_batch_behind, y_batch):
                """A single training step"""
                feed_dict = {
                    sann.input_x_front: x_batch_front,
                    sann.input_x_behind: x_batch_behind,
                    sann.input_y: y_batch,
                    sann.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    sann.is_training: True
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, sann.global_step, train_summary_op, sann.loss, sann.accuracy], feed_dict)
                logger.info("step {0}: loss {1:g}, acc {2:g}".format(step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_batch_front, x_batch_behind, y_batch, writer=None):
                """Evaluates model on a validation set"""
                feed_dict = {
                    sann.input_x_front: x_batch_front,
                    sann.input_x_behind: x_batch_behind,
                    sann.input_y: y_batch,
                    sann.dropout_keep_prob: 1.0,
                    sann.is_training: False
                }
                step, summaries, loss, accuracy, recall, precision, f1, auc = sess.run(
                    [sann.global_step, validation_summary_op, sann.loss, sann.accuracy,
                     sann.recall, sann.precision, sann.F1, sann.AUC], feed_dict)
                logger.info("step {0}: loss {1:g}, acc {2:g}, recall {3:g}, precision {4:g}, f1 {5:g}, AUC {6}"
                            .format(step, loss, accuracy, recall, precision, f1, auc))
                if writer:
                    writer.add_summary(summaries, step)

                return accuracy

            # Generate batches
            batches = dh.batch_iter(
                list(zip(x_train_front, x_train_behind, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            num_batches_per_epoch = int((len(x_train_front) - 1) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            for batch in batches:
                x_batch_front, x_batch_behind, y_batch = zip(*batch)
                train_step(x_batch_front, x_batch_behind, y_batch)
                current_step = tf.train.global_step(sess, sann.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    accuracy = validation_step(x_validation_front, x_validation_behind, y_validation,
                                               writer=validation_summary_writer)
                    best_saver.handle(accuracy, sess, current_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("✔︎ Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("✔︎ Epoch {0} has finished!".format(current_epoch))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_sann()
