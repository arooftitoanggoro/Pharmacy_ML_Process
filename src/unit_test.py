import preprocessing
import util as utils
import pandas as pd
import numpy as np

def categorical_nan_detector():
    # Arrange
    mock_data = {"Region" : ['Jawa', np.nan, 'Kalimantan', 'Bali Nusa']}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"Region" : ['Jawa', 'UNKNOWN', 'Kalimantan', 'Bali Nusa']}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.categorical_nan_detector(mock_data)

    # Assert
    assert processed_data.equals(expected_data)

def test_le_transform():
    # Arrange
    config = utils.load_config()
    mock_data = {"Ship Mode" : ["First Class", "Standard Class", "Second Class"]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"Ship Mode" : [1, 2, 3]}
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(int)

    # Act
    processed_data = preprocessing.le_transform(mock_data["Ship Mode"], config)
    processed_data = pd.DataFrame({"Ship Mode": processed_data})

    # Assert
    assert processed_data.equals(expected_data)