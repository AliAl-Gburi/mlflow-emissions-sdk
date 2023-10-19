# mlflow_emissions_sdk

This package logs the carbon emissions of machine learning models.

To get started:

`pip install mlflow-emissions-package`

Make sure that you have an mlflow client running:

`mlflow ui`

By default, the tracking uri is http://127.0.0.1:5000

To log your emissions you need to provide the tracker with the uri and the name of the experiment. If the experiment does not exist then the tracker will create a new experiment with the given name. Example

```python
from pprint import pprint

import numpy as np
from sklearn.linear_model import LinearRegression

from mlflow_emissions_sdk.experiment_tracking_training import EmissionsTrackerMlflow




tracker_info = {
    "tracking_uri" : "http://127.0.0.1:5000",
    "experiment_name": "test_name"
}
# prepare training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# train a model
model = LinearRegression()

# Instatiates the tracker
runner = EmissionsTrackerMlflow()
runner.read_params(tracker_info)

# Starts the emissions tracking
runner.start_training_job()

history = model.fit(X, y)

# Ends the tracking
runner.end_training_job()

```


