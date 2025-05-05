from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Loads the California housing dataset and splits it into train/test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
