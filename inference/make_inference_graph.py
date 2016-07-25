"""Convert a trained checkpoint into a protobuf inference graph for exporting.

This changes the image preprocessing from sharded inputs to raw image inputsi
(and adds necessary steps)
"""

from __future__ import absolute_import
from __future__ import division


import os.path

import tensorflow as tf

from inception import inception_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('export_graph', '/tmp/skin_inception_export/graph.pb',
                           """Directory where to export inference model protobuf.""")

tf.app.flags.DEFINE_integer('num_classes', '9',
                            """The number of classes (not including the unused background).""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Needs to provide same value as in training.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '',
                           """Directory where to read training checkpoints.""")

tf.app.flags.DEFINE_string('export_type', 'mobile',
                           """The type of export graph: mobile or inference.""")


def inference_input():
    """Returns ops that convert raw image data to a 4D tensor representing a single image.

    Taken from:
    https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_export.py

    The input to the first op can be read using:
    tf.gfile.FastGFile(image_filename, 'r').read()

    """
    # Decode image into float range [0,1]
    jpegs = tf.placeholder(tf.string, shape=(1), name='input')
    image_buffer = tf.squeeze(jpegs, [0])
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
    image = tf.squeeze(image, [0])

    # Rescale the image to [-1,-1]
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    images = tf.expand_dims(image, 0)

    return images, jpegs


def mobile_input():
    """Returns preprocessing ops for iOS mobile input.

    Designed to work with tensorflow's tensorflow/contrib/ios_example/ setup.
    It will accept a 4D float tensor
    """
    # Decode image into float range [0,1]
    jpegs = tf.placeholder(tf.float32, shape=(1, FLAGS.image_size, FLAGS.image_size, 3),
            name='input')
    return jpegs, jpegs
#   image = tf.squeeze(jpegs, [0])
#   image = tf.image.central_crop(image, central_fraction=0.875)
#   image = tf.expand_dims(image, 0)
#   image = tf.image.resize_bilinear(jpegs, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
#   image = tf.squeeze(image, [0])

#   # Rescale the image to [-1,-1]
#   image = tf.sub(image, 0.5)
#   image = tf.mul(image, 2.0)
#   image = tf.expand_dims(image, 0)

#   return image, jpegs


def export():
    with tf.Graph().as_default(), tf.Session() as sess:
        if FLAGS.export_type == 'mobile':
            input_, image_raw = mobile_input()
        elif FLAGS.export_type == 'inference':
            input_, image_raw = inference_input()
        else:
            print('export_type must be mobile or inference, currently %s' %
                    (FLAGS.export_type))
            return

        logits, _ = inception_model.inference(input_, FLAGS.num_classes + 1)
        softmax = tf.nn.softmax(logits, name='softmax')

        variable_averages = tf.train.ExponentialMovingAverage(
            inception_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Load checkpoint
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

            # Write out graph def
            graph_def = sess.graph.as_graph_def()
            tf.train.write_graph(sess.graph.as_graph_def(), os.path.dirname(FLAGS.export_graph),
                    os.path.basename(FLAGS.export_graph))

            print('Successfully converted checkpoint:\n %s/%s\n into proto\n %s\n with inputs of size %d' %
                 (FLAGS.checkpoint_dir, ckpt.model_checkpoint_path, FLAGS.export_graph, FLAGS.image_size))
        else:
            print('No checkpoint file found')



if __name__ == '__main__':
    export()

