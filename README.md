```import keras            # High Level ML Package
import tensorflow       # Low Level ML Package
import numpy            # Helps us do fancy math
import matplotlib

numpy.set_printoptions(linewidth=200, threshold=1000)
# Keras -> Tensorflow -> PC 

# Step 1: Data Pre-Processing 
dataset = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()
print(x_train[19])
print(y_train[19])
# matplotlib.pyplot.imshow(x_train[19], cmap=matplotlib.pyplot.cm.binary)
# matplotlib.pyplot.show()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)
print(x_train[19])
print(y_train[19])

# Step 2: Training -> Model
model = keras.models.Sequential()

# Input Layer
model.add(keras.layers.Flatten()) # 2D -> 1D: 728 neurons 

# Hidden Layers
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))

# Output Layers

model.add(keras.layers.Dense(10, activation="softmax"))

model.compile() # Setting it up
model.fit()     # Training! 

# Step 3: Inferencing (Using It) 
```
