import rwFromDirectory
import numpy as np
import tensorflow as tf

class trainer:

    def __init__(self):
        self.rw = rwFromDirectory.rwFromDirectory()

    def __loss__(self, hypothesis, y):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=[1]))

        return cross_entropy

    def __train__(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train = optimizer.minimize(loss)

        return train

    def set_training_data_set(self, trainingDataSet):
        self.x_data = trainingDataSet.get_x_data()
        self.y_data = trainingDataSet.get_y_data()

    def __set_validation__(self):
        x_validation_ = np.zeros((self.x_validation.shape[0] * self.x_validation.shape[1], self.x_validation.shape[2]))
        y_validation_ = np.zeros((self.y_validation.shape[0] * self.y_validation.shape[1], self.y_validation.shape[2]))

        for i in range(self.x_validation.shape[0]):
            count = 0
            for j in range(self.x_validation.shape[1]):
                x_validation_[count] = self.x_validation[i][j]
                y_validation_[count] = self.y_validation[i][j]
                count += 1

        self.x_validation = x_validation_
        self.y_validation = y_validation_

    def set_validation_data_set(self, trainingDataSet):
        self.x_validation = trainingDataSet.get_x_validation()
        self.y_validation = trainingDataSet.get_y_validation()

        self.__set_validation__()

    def __accuracy__(self, hypothesis, y):
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def check_accuracy(self, x, y, epoch, hypothesis, sess):
        accuracy_test = self.__accuracy__(hypothesis, y)

        self.accuracy[epoch] = sess.run(accuracy_test, feed_dict={x:self.x_validation, y:self.y_validation})
        print("accuracy : " + str(self.accuracy[epoch]))

    def training(self, network, data_set, train_Epoch):
        self.set_training_data_set(data_set)
        self.set_validation_data_set(data_set)

        sess = network.Session()

        x= tf.placeholder(tf.float32, shape=[None, 2], name='x')
        y= tf.placeholder(tf.float32, shape=[None, 2], name='y')

        hypothesis = network.algorithm(x)
        loss = self.__loss__(hypothesis, y)
        train = self.__train__(loss)

        epochs = train_Epoch
        self.losses = np.zeros(epochs)
        self.accuracy = np.zeros(epochs)

        loop = self.x_data.shape[0] * self.x_data.shape[1]
        x_data_input = np.zeros((loop, 2))
        y_data_input = np.zeros((loop, 2))

        print("Training Step")
        for step in range(epochs):
            print("Step : " + str(step))
            count = 0

            for i in range(self.x_data.shape[0]):
                for j in range(self.x_data.shape[1]):
                    x_data_input[count] = self.x_data[i][j]
                    y_data_input[count] = self.y_data[i][j]
                    count += 1

            h, _, l = sess.run([hypothesis, train, loss], feed_dict={x:x_data_input, y:y_data_input})

            self.losses[step] = l
            print("Loss : " + str(self.losses[step]))

            self.check_accuracy(x, y, step, hypothesis, sess)

        self.write_result(network)

    def write_result(self, network):
        network.write_data()
        self.rw.write_data('./model_data/losses', self.losses)
        self.rw.write_data('./model_data/accuracy', self.accuracy)
