# Import all necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Define path to dataset file
original_data_path = "C:\\Users\\Shounak\\Code\\BE Project\\Project Docs\\Dataset\\PricePredictionDataset_80k.csv"

# Load the dataset into a dataframe using pandas
original_data = pd.read_csv(original_data_path)

#visualize the data
print(original_data)

# Here we have multiple columns with categorical values
# For the model to process these values we need to convert them to 
# one hot encoded values
# one hot encode all categorical values using pandas
one_hot_type = pd.get_dummies(original_data['Type'])
original_data.pop('Type')
original_data = original_data.join(one_hot_type)


one_hot_name = pd.get_dummies(original_data['Name'])
original_data.pop('Name')
original_data = original_data.join(one_hot_name)


one_hot_day = pd.get_dummies(original_data['Day'])
original_data.pop('Day')
original_data = original_data.join(one_hot_day)

# Visualise the one hot encoded dataset
print(original_data)

# Clean the dataset by removing any cloumns that are not relevant 
# to train the model
original_data.pop('Is_Countered_Low')
original_data.pop('Is_Countered_High')
original_data.pop('Low_Five')
original_data.pop('Up_Five')
original_data.pop('Is_Accepted')
original_data.pop('Counter_Offer')

# Shuffle the dataset
dataset = shuffle(original_data)

# Split the dataset into train and test samples
train_dataset = dataset.sample(frac=0.8 , random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Extract labels from the train and test dataset
# Here the label is the Final Price we want to predict
train_labels = train_dataset.pop('Final_Price')
test_labels = test_dataset.pop('Final_Price')

# Visualize the description of the dataset
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats)
print(train_dataset.shape)

# Build the tensorflow model
def build_model():
    model = keras.Sequential([
        layers.Dense(20 , activation=tf.nn.relu , 
                     input_shape=[len(train_dataset.keys())], name='inputs'),
        layers.Dense(64 , activation=tf.nn.relu),
        layers.Dense(1 , name='outputs')
    ])
    
    optimizer = tf.train.AdamOptimizer(0.001)

    
    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse' , 'acc'])
    
    return model

model = build_model()
print(model.summary())

# Train the model
# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 150

history = model.fit(
    train_dataset, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()]
)

# Visualize the tail of history (error)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Plot graph for mean squared error and mean absolute error
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 200])
   
plot_history(history)

# Plot graph for error count for test dataset
test_predictions = model.predict(test_dataset).flatten()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")

# Get prediction on test dataset and compare to labeles
example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
print(example_result)
print(train_labels[:10])

# Create function to format new data for prediction
def format_for_prediction(dataset_path):
    raw_dataset = pd.read_csv(dataset_path)
    dataset = raw_dataset.copy()
    df = pd.DataFrame(dataset)
    one_hot_type = pd.get_dummies(df['Type'])
    df1=df.drop('Type', axis=1)
    df1=df1.join(one_hot_type)
    one_hot_name = pd.get_dummies(df1['Name'])
    df1 = df1.drop('Name', axis=1)
    df1 = df1.join(one_hot_name)
    df1.pop('Is_Countered_Low')
    df1.pop('Is_Countered_High')
    df1.pop('Low_Five')
    df1.pop('Up_Five')
    df1.pop('Is_Accepted')
    df1.pop('Counter_Offer')
    one_hot_days = pd.get_dummies(df1['Day'])
    df1 = df1.drop('Day', axis=1)
    df1 = df1.join(one_hot_days)
    dataset = shuffle(df1)
    labels = dataset.pop('Final_Price')
    print(labels)

    return dataset

# New data for prediction
new_dataset_path = "C:\\Users\\Shounak\\Code\\BE Project\\Project Docs\\Dataset\\Predict_v1.csv"
prediction_dataset = format_for_prediction(new_dataset_path)

# Get prediction on new dataset
new_prediction = model.predict(prediction_dataset)
print(new_prediction)

# Save the train model as Tensorflow Saved_Model
tf.contrib.saved_model.save_keras_model(model , 'price_prediction_model_v1')