# RACER
Unofficial Python implementation of the RACER classification algorithm described by [Basiri et. al, 2019](https://link.springer.com/article/10.1007/s00521-017-3117-2).

## Installation
[![PyPI version](https://badge.fury.io/py/pyracer.svg)](https://badge.fury.io/py/pyracer)
```bash
$ pip install pyracer
```

## Usage
RACER is designed to be consistent with Scikit-learn estimator API which makes it very easy to use.


The following example demonstrates the use of RACER on the Zoo dataset.
### Data Obtention and Cleaning
```python
from RACER import RACER, RACERPreprocessor
from sklearn.model_selection import train_test_split
import pandas as pd

# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/
df = pd.read_csv(
    "datasets/zoo.data",
    names=[
        "animal_name",
        "hair",
        "feathers",
        "eggs",
        "milk",
        "airborne",
        "aquatic",
        "predator",
        "toothed",
        "backbone",
        "breathes",
        "venomous",
        "fins",
        "legs",
        "tail",
        "domestic",
        "catsize",
        "type",
    ],
)

X = df.drop(columns=['animal_name', 'type']).astype('category')
Y = df[['type']].astype('category')
```

### RACER Preprocessing Step
```python
X, Y = RACERPreprocessor().fit_transform(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1, test_size=0.3)
```

### Fitting RACER on the Dataset
```python
racer = RACER(alpha=0.95, benchmark=True)
racer.fit(X_train, Y_train)
```

Now you may access the public methods available within the `racer` object such as `score` and `display_rules`. For example:
```python
>>> racer.score(X_test, Y_test)
0.8709677419354839
>>> racer.display_rules()
Algorithm Parameters:
	- Alpha: 0.95
	- Time to fit: 0.008133015999987947s

Final Rules (8 total): (if --> then (label) | fitness)
	[111011011111111101011011111000111111] --> [1000000] (0) | 0.9685714285714285
	[100101101111111001011010010000011111] --> [0100000] (1) | 0.9607142857142856
	[101001101001110101101101100000011111] --> [0001000] (3) | 0.9571428571428571
	[101011101011011010111110101011111011] --> [0000001] (6) | 0.9542857142857143
	[111001101110101010011110000010101010] --> [0000010] (5) | 0.9535714285714285
	[101011101011111101111110101000011011] --> [0010000] (2) | 0.9528571428571428
	[101001101001110101011110001000101010] --> [0000100] (4) | 0.9521428571428571
	[101001101010101010011010100000101010] --> [0000001] (6) | 0.9507142857142856
```


## Official Paper
```bibtex
@Article{Basiri2019,
  author="Basiri, Javad
  and Taghiyareh, Fattaneh
  and Faili, Heshaam",
  title="RACER: accurate and efficient classification based on rule aggregation approach",
  journal="Neural Computing and Applications",
  year="2019",
  month="Mar",
  day="01",
  volume="31",
  number="3",
  pages="895--908",
  issn="1433-3058",
  doi="10.1007/s00521-017-3117-2",
  url="https://doi.org/10.1007/s00521-017-3117-2"
}
```
