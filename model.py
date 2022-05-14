import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-110, 110, 3) 
y = np.arange(-100, 120, 3)

X_train = X[:60]
y_train = y[:60]
X_test = X[60:]
y_test = y[60:]

tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
                               tf.keras.layers.Dense(1)
])
model_1.compile( loss = tf.keras.losses.mae,
                optimizer = tf.keras.optimizers.SGD(),
                metrics = ['mae'])
model_1.fit( X_train, y_train, epochs = 100, verbose = 0)

preds = model_1.predict(X_test)

mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=preds.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=preds.squeeze()).numpy()
print(f"MAE : {mae} :: MSE : {mse}")
