from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataProcessor:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def scale(self) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        return pd.DataFrame(scaled_data, columns=self.data.columns)
