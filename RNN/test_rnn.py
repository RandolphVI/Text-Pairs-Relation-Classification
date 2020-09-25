# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging
import numpy as np

sys.path.append('../')
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("tflog", "logs/Test-{0}.log".format(time.asctime()))

CPT_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_CPT_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'output/' + MODEL


def create_input_data(data: dict):
    return zip(data['f_pad_seqs'], data['b_pad_seqs'], data['onehot_labels'])


def test_rnn():
    """Test RNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args, args.test_file, word2idx)

    # Load rnn model
    OPTION = dh._option(pattern=1)
    if OPTION == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    else:
        logger.info("Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(CPT_DIR)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x_front = graph.get_operation_by_name("input_x_front").outputs[0]
            input_x_behind = graph.get_operation_by_name("input_x_behind").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/topKPreds").outputs[0]
            predictions = graph.get_operation_by_name("output/topKPreds").outputs[1]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "output/topKPreds"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-rnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches_test = dh.batch_iter(list(create_input_data(test_data)), args.batch_size, 1, shuffle=False)

            # Collect the predictions here
            test_counter, test_loss = 0, 0.0
            true_labels = []
            predicted_labels = []
            predicted_scores = []

            for batch_test in batches_test:
                x_f, x_b, y_onehot = zip(*batch_test)
                feed_dict = {
                    input_x_front: x_f,
                    input_x_behind: x_b,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }

                batch_predicted_scores, batch_predicted_labels, batch_loss \
                    = sess.run([scores, predictions, loss], feed_dict)

                for i in y_onehot:
                    true_labels.append(np.argmax(i))
                for j in batch_predicted_scores:
                    predicted_scores.append(j[0])
                for k in batch_predicted_labels:
                    predicted_labels.append(k[0])

                test_loss = test_loss + batch_loss
                test_counter = test_counter + 1

            test_loss = float(test_loss / test_counter)

            # Calculate Precision & Recall & F1
            test_acc = accuracy_score(y_true=np.array(true_labels), y_pred=np.array(predicted_labels))
            test_pre = precision_score(y_true=np.array(true_labels),
                                       y_pred=np.array(predicted_labels), average='micro')
            test_rec = recall_score(y_true=np.array(true_labels),
                                    y_pred=np.array(predicted_labels), average='micro')
            test_F1 = f1_score(y_true=np.array(true_labels),
                               y_pred=np.array(predicted_labels), average='micro')

            # Calculate the average AUC
            test_auc = roc_auc_score(y_true=np.array(true_labels),
                                     y_score=np.array(predicted_scores), average='micro')

            logger.info("All Test Dataset: Loss {0:g} | Acc {1:g} | Precision {2:g} | "
                        "Recall {3:g} | F1 {4:g} | AUC {5:g}"
                        .format(test_loss, test_acc, test_pre, test_rec, test_F1, test_auc))

            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", front_data_id=test_data['f_id'],
                                      behind_data_id=test_data['b_id'], true_labels=true_labels,
                                      predict_labels=predicted_labels, predict_scores=predicted_scores)

    logger.info("All Done.")


if __name__ == '__main__':
    test_rnn()
