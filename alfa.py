'''CNN Classifier with custom estimator.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

from architecture import cnn_architecture, scopes_name_map
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='train_csv_path', default_value='../../scotts/datasets/train_set_with_130_labels.csv',
    docstring='Trainset Data Frame')
tf.app.flags.DEFINE_string(
    flag_name='eval_csv_path', default_value='../../scotts/datasets/eval_set_with_130_labels.csv',
    docstring='Evalset Data Frame')
tf.app.flags.DEFINE_string(
    flag_name='test_csv_path', default_value='../../scotts/datasets/test_set_with_130_labels.csv',
    docstring='Testset Data Frame')
tf.app.flags.DEFINE_string(
    flag_name='output_path', default_value='../../scotts/output/',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='checkpoint_path', default_value='../../scotts/checkpoints/inception_v4_2016_09_09.ckpt',
    docstring='Checkpoint path to load pre-trained weights')
tf.app.flags.DEFINE_string(
    flag_name="network_name", default_value="inception_v4",
    docstring="Network architecture to use")
tf.app.flags.DEFINE_integer(
    flag_name='width', default_value=299,
    docstring='Width of the image')
tf.app.flags.DEFINE_integer(
    flag_name='height', default_value=299,
    docstring='Height of the image')
tf.app.flags.DEFINE_integer(
    flag_name='channels', default_value=3,
    docstring='Channels of the image')
tf.app.flags.DEFINE_integer(
    flag_name='num_labels', default_value=5,
    docstring='Number of labels')
tf.app.flags.DEFINE_integer(
    flag_name='learning_rate', default_value=1e-5,
    docstring='Learning rate for the model')
tf.app.flags.DEFINE_integer(
    flag_name='drop_out', default_value=0.5,
    docstring='Drop out for the model')
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value=48,
    docstring='Batch size')
tf.app.flags.DEFINE_integer(
    flag_name='train_steps', default_value=100000,
    docstring='Train steps')
tf.app.flags.DEFINE_integer(
    flag_name='eval_steps', default_value=48,
    docstring='Frequency to perfom evalutaion')
tf.app.flags.DEFINE_integer(
    flag_name='eval_throttle_secs', default_value=1000,
    docstring='Evaluation every X seconds')


def load_and_resize(paths, labels, width, height, channels, num_labels):  
    image = tf.read_file(paths)
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.divide(image, 255)
    image = tf.image.resize_images(image, [height, width])
    label_one_hot = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    return image, label_one_hot


def get_batch_loader(metadata, width, height, channels, num_labels, batch_size):
    """
    Return op to provide batches through training

    Args:
        metadata: pandas Dataframe
        width: int
        height: int
        channels: int
        num_labels: int
        batch_size: int
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (metadata['URIs'].tolist(), metadata['labels'].factorize()[0]))
    dataset = dataset.map(lambda paths, labels: load_and_resize(
        paths, labels, width, height, channels, num_labels))
    dataset = dataset.batch(batch_size)
    batch_iter = dataset.make_initializable_iterator()
    
    return batch_iter


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self): 
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        tf.logging.info("Initialize batch iterator")
        self.iterator_initializer_func(session)


def input_fn(metadata, width, height, channels, num_labels, batch_size):
    iterator_initializer_hook = IteratorInitializerHook()
    
    def _input_fn():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Data_Loader'):
            iterator = get_batch_loader(metadata,
                width, height, channels, num_labels, batch_size)

            next_X, next_Y = iterator.get_next()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(iterator.initializer)

            return next_X, next_Y

    return _input_fn, iterator_initializer_hook


def metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """

    return {
        'Accuracy': tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1),
            predictions=predictions,
            name='accuracy'),
    }


def get_model_fn(network_name, checkpoint_path):

    def model_fn(features, labels, mode, params):
        """Model function used in the estimator.
        Args:
            features (Tensor): Input features to the model.
            labels (Tensor): Labels tensor for training and evaluation.
            mode (ModeKeys): Specifies if training, evaluation or prediction.
            params (HParams): hyperparameters.
        Returns:
            (EstimatorSpec): Model to be run by Estimator.
        """
        is_training = mode == ModeKeys.TRAIN

        # Define model's architecture
        logits = cnn_architecture(features, params.num_labels,
                        is_training=is_training, network_name=network_name)
        if is_training:
            tf.train.init_from_checkpoint(checkpoint_path,
                        {scopes_name_map[network_name]: scopes_name_map[network_name]})
        predictions = tf.argmax(logits, axis=1)

        # Loss, training and eval operations are not needed during inference.
        if mode != ModeKeys.INFER:
            loss = tf.losses.softmax_cross_entropy(labels, logits)
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate,
                beta1=0.9, beta2=0.999, epsilon=1e-8)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
            eval_metric_ops = metric_ops(labels, predictions)
        else:
            loss = None
            train_op = None
            eval_metric_ops = {}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)
    
    return model_fn

        
def main():

    # Read data sets
    train_set = pd.read_csv(FLAGS.train_csv_path)
    eval_set = pd.read_csv(FLAGS.eval_csv_path)
    test_set = pd.read_csv(FLAGS.test_csv_path)

    # Set model params
    params = tf.contrib.training.HParams(
        num_labels=FLAGS.num_labels,
        learning_rate=FLAGS.learning_rate,
        drop_out=FLAGS.drop_out,
        min_eval_frequency=100,
        train_steps=FLAGS.train_steps
    )

    # Set estimator
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.output_path)
    
    model_fn = get_model_fn(FLAGS.network_name, FLAGS.checkpoint_path)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, 
        params=params,  
        config=run_config  
    )

    # Set iput function
    train_input_fn, train_input_hook = input_fn(train_set,
        FLAGS.width, FLAGS.height, FLAGS.channels, FLAGS.num_labels, FLAGS.batch_size)
    eval_input_fn, eval_input_hook = input_fn(eval_set,
        FLAGS.width, FLAGS.height, FLAGS.channels, FLAGS.num_labels, FLAGS.batch_size)
    test_input_fn, test_input_hook = input_fn(test_set,
        FLAGS.width, FLAGS.height, FLAGS.channels, FLAGS.num_labels, FLAGS.batch_size)
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.train_steps, hooks=[train_input_hook])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=FLAGS.eval_steps, hooks=[eval_input_hook], throttle_secs= FLAGS.eval_throttle_secs)
    # estimator.train(train_input_fn,hooks=[train_input_hook],steps=FLAGS.train_steps,max_steps=None,saving_listeners=None)
    
    tf.estimator.train_and_evaluate(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == '__main__':
    main()