import pandas as pd
import numpy as np

import numpy as np

import os

os.environ['JAX_ENABLE_X64'] = '1'

import keras as kr

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

labels = np.asarray(df.iloc[:, 2].values)

voltages.shape, currents.shape, labels.shape

'''
    STEP 4: Shuffling the dataset
'''
np.random.seed(42)

idx = np.arange(voltages.shape[0])
np.random.shuffle(idx)
voltages, currents, labels = voltages[idx], currents[idx], labels[idx]

labels = pd.Series(labels)
labels = pd.get_dummies(labels).values

print(labels.shape)
print(np.unique(labels, return_counts=True))

'''
    STEP 5: Splitting the dataset into training and testing
'''
X_voltage, x_voltage, X_current, x_current, Y_, y = train_test_split(voltages, currents, labels, test_size=0.3)

model = kr.Sequential(
    [
        kr.layers.Input(shape=X_voltage.shape[1:]),
        kr.layers.BatchNormalization(),
        kr.layers.Dense(30, activation="relu", name="dense1"),
        kr.layers.BatchNormalization(),
        kr.layers.Dense(15, activation="relu"),
        kr.layers.Dense(15, activation="relu", name="dense2"),
        kr.layers.Dense(15, activation="relu", name="dense3"),
        kr.layers.Dense(2, activation="softmax", name="dense4"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 5
epochs = 100

history = model.fit(
    X_voltage,
    Y_,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_split=0.2,
)

#accuracy = accuracy_score(y, model.predict(x_voltage))

#pd.DataFrame([accuracy]).to_csv('./Results/accuracy_FNN_Voltage.csv', index=False, header=False)
pd.DataFrame(model.predict(x_voltage)).to_csv('./Results/predictions_FNN_Voltage.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_FNN_Voltage.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_FNN_Voltage.csv', index=False)

model = kr.Sequential(
    [
        kr.layers.Input(shape=X_voltage.shape[1:]),
        kr.layers.BatchNormalization(),
        kr.layers.Dense(30, activation="relu", name="dense1"),
        kr.layers.BatchNormalization(),
        kr.layers.Dense(15, activation="relu"),
        kr.layers.Dense(15, activation="relu", name="dense2"),
        kr.layers.Dense(15, activation="relu", name="dense3"),
        kr.layers.Dense(2, activation="softmax", name="dense4"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 5
epochs = 100

history = model.fit(
    X_current,
    Y_,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_split=0.2,
)

#accuracy = accuracy_score(y, model.predict(x_voltage))

#pd.DataFrame([accuracy]).to_csv('./Results/accuracy_FNN_Current.csv', index=False, header=False)
pd.DataFrame(model.predict(x_voltage)).to_csv('./Results/predictions_FNN_Current.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_FNN_Current.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_FNN_Current.csv', index=False)