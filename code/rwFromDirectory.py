import numpy as np

class rwFromDirectory:
    def __init(self):
        pass

    def write_data(self, file_path, write_data):
        np.savetxt(file_path, write_data)

    def read_data(self, file_path):
        return np.loadtxt(file_path).astype(np.float32)
