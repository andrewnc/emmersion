import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from DataLoader import DataLoader


def create_network(inputs, num_classes):
    h1 = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[1, 4410], stride=2205)  # [1, 44100, 32] => [1, 200, 32]
    h2 = slim.conv2d(inputs=h1, num_outputs=32, kernel_size=[1, 20])  # [1, 200, 32] => [1, 200, 32]
    h3 = slim.conv2d(inputs=h2, num_outputs=64, kernel_size=[1, 10], stride=10, padding="VALID")  # [1, 200, 32] => [1, 20, 64]
    flattened = tf.reshape(h3, [-1, 20 * 64])
    h4 = slim.fully_connected(inputs=flattened, num_outputs=200)
    return slim.fully_connected(inputs=h4, num_outputs=num_classes, activation_fn=None)


def main():
    # Constants
    num_seconds = 10
    sample_rate = 44100
    data_width = num_seconds * sample_rate
    num_classes = 200

    # Hyperparameters
    learning_rate = 0.001

    # Load the data
    data_loader = DataLoader(truncate_secs=num_seconds, dtype=np.float32)

    # Create the network
    xs = tf.placeholder(tf.float32, [None, 1, data_width, 1])
    ys = tf.placeholder(tf.float32, [None, num_classes])
    network_output = create_network(xs, num_classes)
    predictions = tf.nn.softmax(network_output)
    loss = tf.reduce_sum((ys - predictions)**2.0)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    test_x, test_y = data_loader.get_batch(test=True)

    for i in range(10000):
        batch_x, batch_y = data_loader.get_batch()
        sess.run(optim, feed_dict={xs: batch_x.reshape([-1, 1, data_width, 1]), ys: batch_y})
        if i % 20 == 0:
            preds, loss_val = sess.run((predictions, loss), feed_dict={xs: test_x.reshape([-1, 1, data_width, 1]), ys: test_y})
            print(i, "test error:", loss_val, np.sum(np.argmax(preds, axis=1) == np.argmax(test_y, axis=1)))


if __name__ == "__main__":
    main()
