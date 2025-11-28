from sklearn.linear_model import LinearRegression
import numpy as np

class MLIndexPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.arr_length = 0

    def fit(self, arr):
        """
        arr — sorted array of numbers.
        Train the model: value → index.
        """
        self.arr_length = len(arr)
        values = np.array(arr).reshape(-1, 1)
        indices = np.arange(len(arr))

        self.model.fit(values, indices)

    def predict(self, target):
        """
        Returns the predicted index of the element.
        """
        pred = self.model.predict([[target]])[0]
        return int(max(0, min(pred, self.arr_length - 1)))
