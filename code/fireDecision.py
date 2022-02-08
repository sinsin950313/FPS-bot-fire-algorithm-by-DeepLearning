import variableDataSet
import numpy as np
import tensorflow as tf

class fireDecision:

    SCALE_X = 1000
    set_test = False
    set_train = False

    def __init__(self):
        self.sess = tf.Session()

    def __read_Data__(self):
        v = variableDataSet.variableDataSet()
        v.read_Datas()

        return v

    def __set_Train__(self):
        v = self.__read_Data__()

        self.W1 = tf.Variable(v.get_W1(), dtype = tf.float32)
        self.W5 = tf.Variable(v.get_W5(), dtype = tf.float32)
        self.b = tf.Variable(v.get_b(), dtype = tf.float32)
        self.b_out = tf.Variable(v.get_b_out(), dtype = tf.float32)

        self.sess.run(tf.global_variables_initializer())

    def __set_Test__(self):
        v = self.__read_Data__()

        self.W1 = v.get_W1()
        self.W5 = v.get_W5()
        self.b = v.get_b()
        self.b_out = v.get_b_out()

    def write_data(self):
        v = variableDataSet.variableDataSet()

        W1 = self.sess.run(self.W1)
        W5 = self.sess.run(self.W5)
        b = np.array([self.sess.run(self.b)])
        b_out = np.array([self.sess.run(self.b_out)])

        v.set_Datas(W1, W5, b, b_out)
        v.write_Datas()

    def Session(self):
        return self.sess

    def algorithm(self, x):
        if(not self.set_train):
            self.set_test = False
            self.set_train = True
            self.__set_Train__()

        x = x / self.SCALE_X

        hidden_layer1 = tf.sigmoid(tf.matmul(x, self.W1) + self.b)
        out_layer = tf.nn.softmax(tf.matmul(hidden_layer1, self.W5) + self.b_out)

        return out_layer

    def decision(self, distance, targetArea):
        self.input = np.array([[distance/self.SCALE_X, targetArea]])

        if(not self.set_test):
            self.set_test = True
            self.set_train = False
            self.__set_Test__()

        return self.sess.run(self.algorithm(self.input))
