#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys
from sklearn.metrics import *
from text_cnn import TextCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

config = tf.ConfigProto()
config.gpu_options.allocator_type='BFC'

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        #timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", sys.argv[1]))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
	#saver.restore(sess, "./runs/1461699349/checkpoints/model-10000")
	#print "successfully restored model" 

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, out_labels, y_output = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.out_labels, cnn.output_y],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
	    #predicted_true = np.sum(pred)
            #predicted_false = len(pred) - predicted_true
            #labels = tf.argmax(input_y, 1)
            #actual_true = np.sum(labels)
            #actual_false = len(labels) - actual_true
	    #precision = predicted_true / actual_true
            #recall = predicted_true / actual_true
            precision = 0.0
	    recall = 0.0
            print("{}: step {}, loss {:g}, acc {:g}, P {:g}, R {:g}".format(time_str, step, loss, accuracy, precision, recall))

	    if step % 50 == 0:
	    	pred_1 = np.sum(out_labels)
            	pred_0 = len(out_labels) - pred_1
            	actual_1 = np.sum(y_output)
            	actual_0 = len(y_output) - actual_1
	    	print "Prediction: ", pred_0, pred_1
            	print "Actual:     ", actual_0, actual_1
	    	print classification_report(out_labels, y_output, digits=4)

            train_summary_writer.add_summary(summaries, step)

        def dev_step(dev_data, writer=None):
            """
            Evaluates model on a dev set
            """
	    labels = np.array([])
            actual_output = np.array([])
	    step = 0
	    for dev_batch in dev_data:
	        dev_batch_x, dev_batch_y = zip(*dev_batch)
                feed_dict = {
                  cnn.input_x: dev_batch_x,
                  cnn.input_y: dev_batch_y,
                  cnn.dropout_keep_prob: 1
                }
                s, summaries, loss, accuracy, out_labels, y_output= sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.out_labels, cnn.output_y],
                    feed_dict)
		labels = np.append(labels, out_labels)
                actual_output = np.append(actual_output, y_output)
                step = s

            time_str = datetime.datetime.now().isoformat()
	    pred_1 = np.sum(labels)
            pred_0 = len(labels) - pred_1
            actual_1 = np.sum(actual_output)
            actual_0 = len(actual_output) - actual_1
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#	    print out_labels
#	    print y_output
	    print "Prediction: ", pred_0, pred_1
            print "Actual:     ", actual_0, actual_1
	    print classification_report(labels, actual_output)
	    print confusion_matrix(labels, actual_output)
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_batches = data_helpers.batch_iter(
            		zip(x_dev, y_dev), FLAGS.batch_size, 1)
                dev_step(dev_batches, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
