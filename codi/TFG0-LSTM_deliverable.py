import pandas as pd
import numpy as np

import numpy as np

import os

os.environ['JAX_ENABLE_X64'] = '1'

import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


'''
    Dataset obtained from
    https://zenodo.org/records/10017718
'''
'''
    STEP 1: Load the dataset
'''
df = pd.read_csv('./processed_asimow_dataset.csv')

'''
    STEP 2: We remove those instances that do not have labels assinged (labels = -1)
    
'''
df = df.drop(df[df['labels'] == -1].index)

df.shape

'''
    STEP 3: Get Voltage + Current time series (+ labels)
'''
v_columns = [col for col in df.columns if 'V' in col]
c_columns = [col for col in df.columns if 'I' in col]

voltages = np.asarray(df.loc[:, v_columns].values)
currents = np.asarray(df.loc[:, c_columns].values)

voltages = np.reshape(voltages, (voltages.shape[0], voltages.shape[1], 1))
currents = np.reshape(currents, (currents.shape[0], currents.shape[1], 1))

labels = np.asarray(df.iloc[:, 2].values)

predictors = np.concatenate((voltages, currents), axis=2)

'''
    STEP 4: Shuffling the dataset
'''
np.random.seed(42)

idx = np.arange(predictors.shape[0])
np.random.shuffle(idx)
predictors, labels = predictors[idx], labels[idx]

labels = pd.Series(labels)
labels = pd.get_dummies(labels).values

print(labels.shape)
print(np.unique(labels, return_counts=True))

'''
    STEP 5: Splitting the dataset into training and testing
'''
X, x, Y_, y = train_test_split(predictors, labels, test_size=0.3)

model = keras.Sequential(
    [
        keras.Input(shape=X.shape[1:]),  
        keras.layers.LSTM(64, return_sequences=True, activation="relu"),
        keras.layers.LSTM(32, return_sequences=True, activation="relu"),
        keras.layers.LSTM(16, activation="relu"),
        keras.layers.Dense(32, activation="relu", name="dense"),
        #Dropout(0.5),
        keras.layers.Dense(16, activation="relu", name="dense2"),
        keras.layers.Dense(2, activation="softmax", name="dense3"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 5
epochs = 10

history = model.fit(
    X,
    Y_,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_split=0.2,
)

#accuracy = accuracy_score(y, model.predict(x_voltage))

#pd.DataFrame([accuracy]).to_csv('./Results/accuracy_FNN_Voltage.csv', index=False, header=False)
pd.DataFrame(model.predict(x)).to_csv('./Results/predictions_LSTM.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_LSTM.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_LSTM.csv', index=False)