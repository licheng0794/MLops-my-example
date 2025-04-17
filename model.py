# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions and procedures
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(6, 5))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  
  plt.legend(shadow=True)
  # Set grids
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  # Some text
  plt.title('Model Results', family='Arial', fontsize=14)
  plt.xlabel('X axis values', family='Arial', fontsize=11)
  plt.ylabel('Y axis values', family='Arial', fontsize=11)
  # Show
  plt.savefig('model_results.png', dpi=120)


def mae(y_test, y_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.reduce_mean(tf.abs(y_test - y_pred))
  

def mse(y_test, y_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.reduce_mean(tf.square(y_test - y_pred))


# Check Tensorflow version
print(f"TensorFlow version: {tf.__version__}")


# Create features
X = np.arange(-100, 100, 4)
print(f"X shape: {X.shape}")

# Create labels
y = np.arange(-90, 110, 4)
print(f"y shape: {y.shape}")


# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

print(f"X_train shape before reshape: {X_train.shape}")
print(f"y_train shape before reshape: {y_train.shape}")


# Reshape the data for TensorFlow
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (40, 1)
X_test = np.expand_dims(X_test, axis=-1)    # Shape: (10, 1)
y_train = np.expand_dims(y_train, axis=-1)  # Shape: (40, 1)
y_test = np.expand_dims(y_test, axis=-1)    # Shape: (10, 1)

print(f"X_train shape after reshape: {X_train.shape}")
print(f"y_train shape after reshape: {y_train.shape}")
print(f"X_train sample: {X_train[:5]}")
print(f"y_train sample: {y_train[:5]}")


# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Print model summary
model.summary()

# Compile the model
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.SGD(),
              metrics = ['mae'])

# Fit the model
print("Starting model training...")
model.fit(X_train, y_train, epochs=100, verbose=1)


# Make and plot predictions for model_1
y_preds = model.predict(X_test)
plot_predictions(train_data=X_train.squeeze(), 
                train_labels=y_train.squeeze(),  
                test_data=X_test.squeeze(), 
                test_labels=y_test.squeeze(),  
                predictions=y_preds.squeeze())


# Calculate model_1 metrics
mae_1 = np.round(float(mae(y_test, y_preds)), 2)
mse_1 = np.round(float(mse(y_test, y_preds)), 2)
print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
