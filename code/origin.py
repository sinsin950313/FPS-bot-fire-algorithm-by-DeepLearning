import Posibility
import numpy as np
#import matplotlib.pylab as plt
import tensorflow as tf

class fireAtWhere:

    def __init__(self):
        self.W1 = None
        self.W5 = None
        self.b = None
        self.b_out = None
        self.x_data = None
        self.y_data = None

        self.SCALE = 1000

        self.accuracy = None
        self.efficiency = None
        self.sess = tf.Session()

    def __set_Random_Weight__(self):
        self.W1 = tf.Variable(tf.random_normal([2, 5]), name='weight1')
        self.W5 = tf.Variable(tf.random_normal([5, 2]), name='weight5')

    def __set_Random_bias__(self):
        self.b = tf.Variable(tf.random_normal([1]), name='bias')
        self.b_out = tf.constant(np.array([self.efficiency]), dtype=tf.float32, name='bias_out')

    def setting_Data_Randomly(self, accuracy, efficiency):
        self.accuracy = accuracy
        self.efficiency = efficiency

        self.__set_Random_Weight__()
        self.__set_Random_bias__()

    def __set_x_data__(self):
        #distance
        self.distance = np.arange(50, 390, 1.0)
        #target area
        self.area = np.arange(0.5, 10, 0.5)
        temp_data = np.zeros(shape=(self.area.size, self.distance.size, 2))

        for i in range(self.distance.size):
            for j in  range(self.area.size):
                temp_data[j][i] = np.array([self.distance[i], self.area[j]])

        self.x_data = temp_data

    def __accuracy_rate__(self, inputData, givenAccuracy):
        idealAccuracyWithGivenArea = givenAccuracy * (inputData[1] / self.idealTargetArea)
        coefficient = (self.idealTargetDistance / inputData[0]) ** 2
        seeArea = coefficient * inputData[1]
        ratio = seeArea * inputData[1]

        return idealAccuracyWithGivenArea * ratio
    
    def __set_y_data__(self, x_data, givenAccuracy, efficiency):
        pos = Posibility.Posibility()
        self.y_data = np.zeros(shape=(np.size(x_data, 0), np.size(x_data, 1), 2))

        for i in range(np.size(x_data, 0)):
            for j in range(np.size(x_data, 1)):
                accuracy = self.__accuracy_rate__(x_data[i][j], givenAccuracy)
                correct = pos.result(self.tryCount, accuracy, int(efficiency * self.tryCount))

                if(correct >= efficiency):
                    temp = np.array([0, 1])
                else:
                    temp = np.array([1, 0])

                self.y_data[i][j] = temp

    def set_Datas(self):
        self.__set_x_data__()
        self.__set_y_data__(self.x_data, self.accuracy, self.efficiency)

    def data_Training(self, train_Epoch):
        self.x_data /= self.SCALE

        x= tf.placeholder(tf.float32, shape=[None, 2], name='x')
        y= tf.placeholder(tf.float32, shape=[None, 2], name='y')

        hypothesis = self.__inference__(x)
        loss = self.__loss__(hypothesis, y)
        train = self.__train__(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Shuffle data
        r1 = np.random.permutation(self.x_data.shape[0])
        r2 = np.random.permutation(self.x_data.shape[1])

        input1 = r1[:int(r1.size * 0.8)]
        input2 = r2[:int(r2.size * 0.8)]

        epochs = train_Epoch
        self.losses = np.zeros(epochs)
        self.accuracy_test_result = np.zeros(epochs)

        print("Training Step")
        for step in range(epochs):
            print("Step : " + str(step))
            for i in input1:
                for j in input2:
                    x_data_input = np.array([self.x_data[i][j]])
                    y_data_input = np.array([self.y_data[i][j]])

                    h, _, l = self.sess.run([hypothesis, train, loss], feed_dict={x:x_data_input, y:y_data_input})

                    self.losses[step] += l
            self.losses[step] /= (input1.size * input2.size)
            print("Loss : " + str(self.losses[step]))
            self.__test_Data_Accuracy__(step, x, y, hypothesis, r1, r2)

        self.__write_Data__()

    def train_with_random(self, accuracy, efficiency, train_Epoch):
        self.setting_Data_Randomly(accuracy, efficiency)
        self.set_Datas()
        self.data_Training(train_Epoch)

    def __accuracy__(self, hypothesis, y):
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def __test_Data_Accuracy__(self, step, x, y, hypothesis, r1, r2):
        #make Test data
        test1 = r1[int(r1.size * 0.8):]
        test2 = r2[int(r2.size * 0.8):]

        X_test = np.array([self.x_data[0][0]])
        Y_test = np.array([self.y_data[0][0]])

        accuracy_test = self.__accuracy__(hypothesis, y)

        loop = 0

        #test accuracy
        for i in test1:
            for j in test2:
                X_test = np.append(X_test, [self.x_data[i][j]], axis=0)
                Y_test = np.append(Y_test, [self.x_data[i][j]], axis=0)

                self.accuracy_test_result[step] += self.sess.run(accuracy_test, feed_dict={x:X_test, y:Y_test})
            loop = i * j

        self.accuracy_test_result[step] /= loop
        print("accuracy : " + str(self.accuracy_test_result[step]))

    def __write_Data__(self):
        array = self.sess.run(self.W1)
        np.savetxt('./model_data/W1', array)
        array = self.sess.run(self.W5)
        np.savetxt('./model_data/W5', array)

        array = self.sess.run(self.b)
        np.savetxt('./model_data/b', array)
        array = self.sess.run(self.b_out)
        np.savetxt('./model_data/b_out', array)

        size = np.array([[self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2]], [self.y_data.shape[0], self.y_data.shape[1], self.y_data.shape[2]]])
        np.savetxt('./model_data/xy_size', size)

        temp_Array = self.x_data
        temp_Array = temp_Array.reshape((self.x_data.size))
        np.savetxt('./model_data/x_data', temp_Array)
        temp_Array = self.y_data
        temp_Array = temp_Array.reshape((self.y_data.size))
        np.savetxt('./model_data/y_data', temp_Array)

        np.savetxt('./model_data/losses', self.losses)
        np.savetxt('./model_data/accuracy_test_result', self.accuracy_test_result)

        np.savetxt('./model_data/distance_data', self.distance)
        np.savetxt('./model_data/area_data', self.area)

        array = np.array([self.accuracy, self.efficiency])
        np.savetxt('./model_data/etc', array)

    def __read__(self, file_path):
        return np.loadtxt(file_path)

    def read_Data(self):
        array = self.__read__('./model_data/W1')
        self.W1 = tf.Variable(array, name='weight1', dtype=tf.float32)
        array = self.__read__('./model_data/W5')
        self.W5 = tf.Variable(array, name='weight5', dtype=tf.float32)

        array = self.__read__('./model_data/b')
        self.b = tf.Variable(array, name='bias', dtype=tf.float32)
        array = self.__read__('./model_data/b_out')
        self.b_out = tf.constant(array, dtype=tf.float32, name='bias_out')

        xy_size = self.__read__('./model_data/xy_size')
        
        array = self.__read__('./model_data/x_data')
        self.x_data = array.reshape((int(xy_size[0][0]), int(xy_size[0][1]), int(xy_size[0][2])))
        array = self.__read__('./model_data/y_data')
        self.y_data = array.reshape((int(xy_size[1][0]), int(xy_size[1][1]), int(xy_size[1][2])))

        array = self.__read__('./model_data/losses')
        self.losses = array

        array = self.__read__('./model_data/accuracy_test_result')
        self.accuracy_test_result = array

        array = self.__read__('./model_data/distance_data')
        self.distance = array
        array = self.__read__('./model_data/area_data')
        self.area = array

        array = self.__read__('./model_data/etc')
        self.accuracy = array[0]
        self.efficiency = array[1]

        init = tf.global_variables_initializer()
        self.sess.run(init)

    #Re code from here
    idealTargetArea = 1.0
    idealTargetDistance = 100.0
    tryCount = 5

    def __inference__(self, input):
        hidden_layer1 = tf.sigmoid(tf.matmul(input, self.W1) + self.b)
        out_layer = tf.nn.softmax(tf.matmul(hidden_layer1, self.W5) + self.b_out)

        return out_layer

    def __loss__(self, hypothesis, y):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=[1]))

        return cross_entropy

    def __train__(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train = optimizer.minimize(loss)

        return train

    #def showTrainGraph(self, targetArea):
    #    plt.title("Train Graph")

    #    temp_data = np.zeros(shape=(self.distance.size, 2))

    #    for i in range(self.distance.size):
    #        temp_data[i] = np.array([self.distance[i], targetArea])

    #    x= tf.placeholder(tf.float32, shape=[None, 2], name='x')

    #    plt.plot(self.distance, self.sess.run(self.__inference__(x), feed_dict={x:temp_data}))
    #    plt.show()

    def whereToFire(self, distance, targetArea):
        input_value = np.array([[distance/self.SCALE, targetArea]])
        x = tf.placeholder(tf.float32, shape=[None, 2])
        process = self.__inference__(x)

        return self.sess.run(process, feed_dict={x:input_value})
