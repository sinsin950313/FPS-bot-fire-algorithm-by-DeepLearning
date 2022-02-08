import rwFromDirectory
import numpy as np

class variableDataSet:

    def __init__(self):
        self.W1 = None
        self.W5 = None

        self.b = None
        self.b_out = None

        self.rw = rwFromDirectory.rwFromDirectory()

    def __set_Random_Weight__(self):
        self.W1 = np.random.rand(2, 5)
        self.W5 = np.random.rand(5, 2)

    def __set_Random_bias__(self):
        self.b = np.random.rand(5)
        self.b_out = np.random.rand(2)

    def create_Datas(self):
        self.__set_Random_Weight__()
        self.__set_Random_bias__()

        self.write_Datas()

    def set_Datas(self, W1, W5, b, b_out):
        self.W1 = W1
        self.W5 = W5
        self.b = b
        self.b_out = b_out

    def write_Datas(self):
        self.rw.write_data('./model_data/W1', self.W1)
        self.rw.write_data('./model_data/W5', self.W5)

        self.rw.write_data('./model_data/b', self.b)
        self.rw.write_data('./model_data/b_out', self.b_out)

    def read_Datas(self):
        self.W1 = self.rw.read_data('./model_data/W1')
        self.W5 = self.rw.read_data('./model_data/W5')

        self.b = self.rw.read_data('./model_data/b')
        self.b_out = self.rw.read_data('./model_data/b_out')

    def get_W1(self):
        return self.W1

    def get_W5(self):
        return self.W5

    def get_b(self):
        return self.b

    def get_b_out(self):
        return self.b_out
