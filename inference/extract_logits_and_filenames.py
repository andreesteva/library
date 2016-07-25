"""Extract features and filenames of a sharded dataset (using a specified checkpoint) and write them to disk.

Assumes Inception v3 architecture and corresponding checkpoint is used.

"""

from __future__ import division

import numpy as np
import os
import math
from datetime import datetime
import time
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.skin_data import SkinData


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/archive/esteva/experiments/skindata4/baseline/train',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_string('output_dir', '/archive/esteva/experiments/skindata4/baseline/retrieve',
                           """Directory where to write out the features and filenames.""")

tf.app.flags.DEFINE_string('output_name', 'data',
                           """Prefix to use in naming the written files.""")

tf.app.flags.DEFINE_integer('num_examples', 1942,
                            """Number of examples to run. Note that the connected componenets"""
                            """validation dataset contains 14712 examples.""")

# FLAGS.data_dir (defined in SkinData) must point to a folder containing shards prefixed by
# either 'train' or 'validation'. Which to use is defined below.
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation', 'train'.""")

tf.app.flags.DEFINE_string('features_tensor_name', 'inception_v3/logits/flatten/Reshape:0',
                           """The name of the features tensor to use.""")

tf.app.flags.DEFINE_string('num_classes', '9',
                           """The number of classes used for this graph.""")

def retrieve(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default(), tf.Session() as sess:
    # Get images and labels from the dataset.
    images, labels, filenames_tensor = image_processing.inputs(dataset, return_filenames=True)


    # Build a Graph that computes the features.
    num_classes = dataset.num_classes() + 1
    _, _ = inception.inference(images, num_classes, restore_logits=False)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Restore checkpoint.
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      features_tensor = tf.get_default_graph().get_tensor_by_name(FLAGS.features_tensor_name)
      features = []
      filenames = []
      step = 0
      while step < num_iter and not coord.should_stop():
        features_batch, filenames_batch = sess.run([features_tensor, filenames_tensor])
        features.append(features_batch)
        filenames.extend(filenames_batch)

        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()
      features = features[:FLAGS.num_examples]
      filenames = filenames[:FLAGS.num_examples]

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return np.vstack(features), filenames


def main(unused_argv=None):
  # The number of classes doesn't matter, fix it to 1.
  num_classes = int(FLAGS.num_classes)
  dataset = SkinData(subset=FLAGS.subset, num_classes=num_classes)
  assert dataset.data_files()

  features, filenames = retrieve(dataset)
  write_features = os.path.join(FLAGS.output_dir, '%s-features.npy' % FLAGS.output_name)
  write_filenames = os.path.join(FLAGS.output_dir, '%s-filenames.txt' % FLAGS.output_name)

  np.save(write_features, features)
  with open(write_filenames, 'w') as f:
    prefix = ""
    for fn in filenames:
      f.write(prefix)
      f.write(fn)
      prefix = "\n"
  print 'Features data written to: %s' % write_features
  print 'Corresponding filenames written to: %s ' % write_filenames


if __name__ == '__main__':
  tf.app.run()


