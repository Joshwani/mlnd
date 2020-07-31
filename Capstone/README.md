# Fitting the LPPLS model to empirical data via Keras and TensorFlow

## Setup
All work towards this objective is currently on the warp_speed branch of the BiT lppls repo. You'll need to clone that repo to your local machine, checkout the warp_speed branch and import it as a package.
```bash
git clone https://github.com/Boulder-Investment-Technologies/lppls.git lppls_local
cd lppls_local
git checkout warp_speed
```
Then you should be good to import the local package as long as it is in your sys.path.

```python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adagrad, SGD
from sklearn.preprocessing import MinMaxScaler

import multiprocessing
import time

# importing local version of cloned lppls package 
# (https://github.com/Boulder-Investment-Technologies/lppls.git)
if '/Users/joshnielsen/projects/lppls_local' not in sys.path:
    sys.path.append('/Users/joshnielsen/projects/lppls_local')

from lppls import lppls_layer
```
## Fitting the Model
```python
# read the downloaded data (same dataset from medium article: 
df = pd.read_csv('https://finance.yahoo.com/quote/SPY/history?period1=1235862000&period2=1577401200&interval=1d&filter=history&frequency=1d ', 
                 index_col='Date', 
                 parse_dates=True)

df.fillna(method='ffill', inplace=True)

# LPPLS works better with log prices
x = np.log(df['Close'].values)

# fits should be more stable if we always use the same domain
x = MinMaxScaler().fit_transform(x.reshape(-1, 1))

# reshape data into keras batch format 
x = x.reshape(1, -1)
```

```python
model = Sequential([lppls_layer.LPPLSLayer()])
model.compile(loss='mse', optimizer=Adagrad(0.011))
hist = model.fit(x, x, epochs=8000, verbose=0)

plt.plot(hist.history["loss"])
plt.show()

res = pd.DataFrame({"close": x[0], 
        "lppls": model.predict_on_batch(x)
       })
res.plot()
```

![]('/Joshwani/mlnd/blob/master/Capstone/imgs/loss.png?raw=true')

![]('/Joshwani/mlnd/blob/master/Capstone/imgs/fit.png?raw=true')