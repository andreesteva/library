import tensorflow as tf
from inception import inception_model as inception
import os


def filename2image(image_size=299, central_fraction=0.875):
    """Returns ops that convert a filename to a 4D tensor representing a single image.

    Usage:
        image_op, buffer_op = decode_filename()
        imbuffer = tf.gfile.FastGFile('path/to/file.jpg', 'r').read()
        with tf.Session() as sess:
            image = sess.run([image_op], feed_dict={buffer_op : [imbuffer]})
        plt.imshow(image)


    Taken from:
    https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_export.py

    Args:
        image_size (int): image is resized to this (default 299 for Inception v3)
        central_fraction (float): the central fraction crop to take

    Returns:
        Two ops, one for the image buffer input, the other the tensorflow-ready image.

    """
    image_buffer = tf.placeholder(tf.string, shape=(1))
    image_buffer = tf.squeeze(image_buffer, [0])
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    image = tf.expand_dims(image, 0)

    return image, image_buffer


def load_checkpoint(checkpoint_dir, sess):
    """Loads a checkpoint into the default graph.

    Note: function needs to be called from within a tf.Graph.as_default() context

    Args:
        checkpoint_dir (str) : a checkpoint directory
        sess (tf.Session): Tensorflow session object
    """

    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Load checkpoint
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.join(checkpoint_dir,
                                             ckpt.model_checkpoint_path))

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
        print('No checkpoint file found')
