class Posibility:
    def __init__(self):
        pass

    def __factorial__(self, n):
        result = 1
        for i in range(n):
            result *= (i + 1)

        return result

    def __Combination__(self, tryCount, hopeResult):
        result = self.__factorial__(tryCount) / (self.__factorial__(tryCount - hopeResult) * self.__factorial__(hopeResult))

        return result

    def result(self, tryCount, probability, hopeResult):
        result = self.__Combination__(tryCount, hopeResult) * pow(probability, hopeResult) * pow((1 - probability), (tryCount - hopeResult))

        return result
