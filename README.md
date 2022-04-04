# Neural Networks

Module 13 Challenge

[![Screen-Shot-2022-04-03-at-8-25-10-PM.png](https://i.postimg.cc/FzPqn5S6/Screen-Shot-2022-04-03-at-8-25-10-PM.png)](https://postimg.cc/bSt3s530)

# Background

For this challenge I work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked me to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

---

## Technologies

The data we're analyzing comes from a jupyter notebook that we'll create and import files to. We'll be using Python to run and read our data. 

* [jupyter] - (https://github.com/jupyter/notebook) - Helps us run our code and get the information we need from the data listed in csv files.

---

## Installation Guide

In order for us to get the data we need we must import pandas, plots and the csv files we want to observe.

```python
# Imports
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```


## Usage
---
## Define at least three new deep neural network models.

* With each, try to improve on your first model’s predictive accuracy.


```python
# First alternative layer
nn_A1.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Output layer
nn.add(Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn_A1.summary()
```


---
```python
# Second alternative layer
nn_A2.add(Dense(units=hidden_nodes_layer1_A2, input_dim=number_input_features, activation="relu"))

# Output layer
nn_A2.add(Dense(units=number_output_neurons, activation="linear"))

# Check the structure of the model
nn_A2.summary()
```

---
```python
# Third alternative layer
nn_A3.add(Dense(units=hidden_nodes_layer1_A3, input_dim=number_input_features, activation="relu"))

# Output layer
nn_A3.add(Dense(units=number_output_neurons, activation="linear"))

# Check the structure of the model
nn_A3.summary()
```

---
## Contributors

Brought to you by Elgin Braggs Jr.

---

## License

MIT