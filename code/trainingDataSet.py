import posibility
import numpy as np
import rwFromDirectory

class trainingDataSet:

    idealTargetArea = 1.0
    idealTargetDistance = 100.0
    tryCount = 5

    def __init__(self):
        self.x_data = None
        self.y_data = None

        self.accuracy = None
        self.efficiency = None

        self.rw = rwFromDirectory.rwFromDirectory()

    def __set_x_data__(self):
        #distance
        self.distance = np.arange(50, 1000, 1.0)
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
        pos = posibility.Posibility()
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

    def __classify_Data_Set__(self):
        r1 = np.random.permutation(self.x_data.shape[0])
        r2 = np.random.permutation(self.x_data.shape[1])

        input1 = r1[int(r1.size * 0.7):]
        input2 = r2[int(r2.size * 0.7):]

        x_validation = np.zeros(shape=(input1.size, input2.size, self.x_data.shape[2]))
        y_validation = np.zeros(shape=(input1.size, input2.size, self.y_data.shape[2]))

        for i in range(input1.size):
            for j in range(input2.size):
                x_validation[i][j] = np.array([self.x_data[input1[i]][input2[j]]])
                y_validation[i][j] = np.array([self.y_data[input1[i]][input2[j]]])

        self.x_validation = x_validation
        self.y_validation = y_validation

        input1 = r1[:int(r1.size * 0.7)]
        input2 = r2[:int(r2.size * 0.7)]

        x_data = np.zeros(shape=(input1.size, input2.size, self.x_data.shape[2]))
        y_data = np.zeros(shape=(input1.size, input2.size, self.y_data.shape[2]))

        for i in range(input1.size):
            for j in range(input2.size):
                x_data[i][j] = np.array([self.x_data[input1[i]][input2[j]]])
                y_data[i][j] = np.array([self.y_data[input1[i]][input2[j]]])

        self.x_data = x_data
        self.y_data = y_data

    def __write_Data__(self):
        size = np.array([[self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2]], [self.y_data.shape[0], self.y_data.shape[1], self.y_data.shape[2]]])
        self.rw.write_data('./model_data/xy_size', size)

        temp_Array = self.x_data
        temp_Array = temp_Array.reshape((self.x_data.size))
        self.rw.write_data('./model_data/x_data', temp_Array)

        temp_Array = self.y_data
        temp_Array = temp_Array.reshape((self.y_data.size))
        self.rw.write_data('./model_data/y_data', temp_Array)

        array = np.array([self.accuracy, self.efficiency])
        self.rw.write_data('./model_data/etc', array)

        size = np.array([[self.x_validation.shape[0], self.x_validation.shape[1], self.x_validation.shape[2]], [self.y_validation.shape[0], self.y_validation.shape[1], self.y_validation.shape[2]]])
        self.rw.write_data('./model_data/xy_validation_size', size)

        temp_Array = self.x_validation
        temp_Array = temp_Array.reshape((self.x_validation.size))
        self.rw.write_data('./model_data/x_validation', temp_Array)

        temp_Array = self.y_validation
        temp_Array = temp_Array.reshape((self.y_validation.size))
        self.rw.write_data('./model_data/y_validation', temp_Array)

    def create_Datas(self, accuracy, efficiency):
        self.accuracy = accuracy
        self.efficiency = efficiency

        self.__set_x_data__()
        self.__set_y_data__(self.x_data, self.accuracy, self.efficiency)

        self.__classify_Data_Set__()
        self.__write_Data__()

    def read_Datas(self):
        xy_size = self.rw.read_data('./model_data/xy_size')
        
        array = self.rw.read_data('./model_data/x_data')
        self.x_data = array.reshape((int(xy_size[0][0]), int(xy_size[0][1]), int(xy_size[0][2])))

        array = self.rw.read_data('./model_data/y_data')
        self.y_data = array.reshape((int(xy_size[1][0]), int(xy_size[1][1]), int(xy_size[1][2])))

        array = self.rw.read_data('./model_data/etc')
        self.accuracy = array[0]
        self.efficiency = array[1]

        xy_validation_size = self.rw.read_data('./model_data/xy_validation_size')
        
        array = self.rw.read_data('./model_data/x_validation')
        self.x_validation = array.reshape((int(xy_validation_size[0][0]), int(xy_validation_size[0][1]), int(xy_validation_size[0][2])))
        
        array = self.rw.read_data('./model_data/y_validation')
        self.y_validation = array.reshape((int(xy_validation_size[0][0]), int(xy_validation_size[0][1]), int(xy_validation_size[0][2])))

    def get_x_data(self):
        return self.x_data

    def get_y_data(self):
        return self.y_data

    def get_accuracy(self):
        return self.accuracy

    def get_efficiency(self):
        return self.efficiency

    def get_x_validation(self):
        return self.x_validation

    def get_y_validation(self):
        return self.y_validation
