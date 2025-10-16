from sktime.datasets.forecasting._base import BaseForecastingDataset


class SimpleDataset(BaseForecastingDataset):
    """Random 0 and 1 values.

    If correlated is True, the 0 and 1s appear in groups of 3.
    """

    def __init__(self, correlated: bool = False, size: int = 100):
        self.correlated = correlated
        self.size = size
        super().__init__()

        self._cache = None

    def generate_cache(self):
        import numpy as np
        import pandas as pd

        if self.correlated:
            data = np.arange(self.size) // 3 % 2
        else:
            data = np.random.choice([0, 1], size=self.size)

        index = pd.date_range("2021-01-01", periods=self.size, freq="D")
        y = pd.Series(data, index=index, name="y").to_frame("points")

        self._cache = (y,)

    def load(self, *args):
        if self._cache is None:
            self.generate_cache()

        if "y" not in args:
            raise ValueError("Only 'y' is supported as return argument.")
        return self._cache[0]
