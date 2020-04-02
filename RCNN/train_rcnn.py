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
from text_rcnn import TextRCNN
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

args = parser.parameter_parser()
OPTION = dh._option(pattern=0)
logger = dh.logger_fn("tflog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))


def train_rcnn():
    """Training RCNN model."""
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

    # Build a graph and rcnn object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rcnn = TextRCNN(
                sequence_length=args.pad_seq_len,
                vocab_size=VOCAB_SIZE,
                embedding_type=args.embedding_type,
                embedding_size=args.embedding_dim,
                lstm_hidden_size=args.lstm_dim,
                filter_sizes=args.filter_sizes,
                num_filters=args.num_filters,
                fc_hidden_size=args.fc_dim,
                num_classes=y_train.shape[1],
                l2_reg_lambda=args.l2_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                           global_step=rcnn.global_step, decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(rcnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=args.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=rcnn.global_step, name="train_op")

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
            loss_summary = tf.summary.scalar("loss", rcnn.loss)

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
                # Load rcnn model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            if OPTION == 'T':
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = "embedding"
                embedding_conf.metadata_path = args.metadata_file

                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(rcnn.global_step)

            def train_step(x_batch_front, x_batch_behind, y_batch):
                """A single training step"""
                feed_dict = {
                    rcnn.input_x_front: x_batch_front,
                    rcnn.input_x_behind: x_batch_behind,
                    rcnn.input_y: y_batch,
                    rcnn.dropout_keep_prob: args.dropout_rate,
                    rcnn.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, rcnn.global_step, train_summary_op, rcnn.loss], feed_dict)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_batch_front, x_batch_behind, y_batch, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = dh.batch_iter(list(zip(x_batch_front, x_batch_behind, y_batch)),
                                                   args.batch_size, 1)
                eval_counter, eval_loss = 0, 0.0
                true_labels = []
                predicted_scores = []
                predicted_labels = []

                for batch_validation in batches_validation:
                    x_batch_val_front, x_batch_val_behind, y_batch_val = zip(*batch_validation)
                    feed_dict = {
                        rcnn.input_x_front: x_batch_val_front,
                        rcnn.input_x_behind: x_batch_val_behind,
                        rcnn.input_y: y_batch_val,
                        rcnn.dropout_keep_prob: 1.0,
                        rcnn.is_training: False
                    }
                    step, summaries, scores, predictions, cur_loss = sess.run(
                        [rcnn.global_step, validation_summary_op,
                         rcnn.topKPreds, rcnn.predictions, rcnn.loss], feed_dict)

                    # Prepare for calculating metrics
                    for i in y_batch_val:
                        true_labels.append(np.argmax(i))
                    for j in scores[0]:
                        predicted_scores.append(j[0])
                    for k in predictions:
                        predicted_labels.append(k)

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                if writer:
                    writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)

                # Calculate Precision & Recall & F1
                eval_acc = accuracy_score(y_true=np.array(true_labels), y_pred=np.array(predicted_labels))
                eval_pre = precision_score(y_true=np.array(true_labels),
                                           y_pred=np.array(predicted_labels), average='micro')
                eval_rec = recall_score(y_true=np.array(true_labels),
                                        y_pred=np.array(predicted_labels), average='micro')
                eval_F1 = f1_score(y_true=np.array(true_labels),
                                   y_pred=np.array(predicted_labels), average='micro')

                # Calculate the average AUC
                eval_auc = roc_auc_score(y_true=np.array(true_labels),
                                         y_score=np.array(predicted_scores), average='micro')

                return eval_loss, eval_acc, eval_pre, eval_rec, eval_F1, eval_auc

            # Generate batches
            batches_train = dh.batch_iter(
                list(zip(x_train_front, x_train_behind, y_train)), args.batch_size, args.epochs)

            num_batches_per_epoch = int((len(x_train_front) - 1) / args.batch_size) + 1

            # Training loop. For each batch...
            for batch_train in batches_train:
                x_batch_front, x_batch_behind, y_batch = zip(*batch_train)
                train_step(x_batch_front, x_batch_behind, y_batch)
                current_step = tf.train.global_step(sess, rcnn.global_step)

                if current_step % args.evaluate_steps == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_acc, eval_pre, eval_rec, eval_F1, eval_auc = \
                        validation_step(x_validation_front, x_validation_behind,
                                        y_validation, writer=validation_summary_writer)
                    logger.info("All Validation set: Loss {0:g} | Acc {1:g} | Precision {2:g} | "
                                "Recall {3:g} | F1 {4:g} | AUC {5:g}"
                                .format(eval_loss, eval_acc, eval_pre, eval_rec, eval_F1, eval_auc))
                    best_saver.handle(eval_acc, sess, current_step)
                if current_step % args.checkpoint_steps == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("Epoch {0} has finished!".format(current_epoch))

    logger.info("All Done.")


if __name__ == '__main__':
    train_rcnn()
