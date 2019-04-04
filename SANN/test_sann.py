# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import numpy as np
import tensorflow as tf

from utils import checkmate as cm
from utils import data_helpers as dh

# Parameters
# ==================================================

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()))

MODEL = input("☛ Please input the model file you want to test, it should be like(1490175368): ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Validation.json'
TESTSET_DIR = '../data/Test.json'
MODEL_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.flags.DEFINE_integer("pad_seq_len", 120, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("lstm_hidden_size", 256, "Hidden size for bi-lstm layer(default: 256)")
tf.flags.DEFINE_integer("attention_unit_size", 350, "Attention unit size(default: 350)")
tf.flags.DEFINE_integer("attention_hops_size", 30, "Attention hops size(default: 30)")
tf.flags.DEFINE_boolean("attention_penalization", True, "Use penalization or not(default: True)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Test Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test_sann():
    """Test SANN model."""

    # Load data
    logger.info("✔︎ Loading data...")
    logger.info("Recommended padding Sequence length is: {0}".format(FLAGS.pad_seq_len))

    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data_and_labels(FLAGS.test_data_file, FLAGS.embedding_dim)

    logger.info("✔︎ Test data padding...")
    x_test_front, x_test_behind, y_test = dh.pad_data(test_data, FLAGS.pad_seq_len)
    y_test_labels = test_data.labels

    # Load sann model
    BEST_OR_LATEST = input("☛ Load Best or Latest Model?(B/L): ")

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("✘ The format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("✔︎ Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    else:
        logger.info("✔︎ Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
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
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            topKPreds = graph.get_operation_by_name("output/topKPreds").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "output/logits|output/predictions|output/softmax_scores|output/topKPreds"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-sann-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test_front, x_test_behind, y_test, y_test_labels)),
                                    FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_labels = []
            all_predicted_labels = []
            all_predicted_values = []

            for index, x_test_batch in enumerate(batches):
                x_batch_front, x_batch_behind, y_batch, y_batch_labels = zip(*x_test_batch)
                feed_dict = {
                    input_x_front: x_batch_front,
                    input_x_behind: x_batch_behind,
                    input_y: y_batch,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }

                batch_predicted_labels, batch_predicted_values, batch_acc, batch_loss \
                    = sess.run([predictions, topKPreds, accuracy, loss], feed_dict)

                all_labels = np.append(all_labels, y_batch_labels)
                all_predicted_labels = np.concatenate([all_predicted_labels, batch_predicted_labels])
                all_predicted_values = np.append(all_predicted_values, batch_predicted_values)

                logger.info("✔︎ Test batch {0}: loss {1:g}, accuracy {2:g}.".format((index+1), batch_loss, batch_acc))

            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", front_data_id=test_data.front_testid,
                                      behind_data_id=test_data.behind_testid, all_labels=all_labels,
                                      all_predict_labels=all_predicted_labels, all_predict_values=all_predicted_values)

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    test_sann()
