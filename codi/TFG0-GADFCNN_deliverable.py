import pandas as pd
import numpy as np

import numpy as np

import os

os.environ['JAX_ENABLE_X64'] = '1'

import keras

import markov as mk

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
from pyts.multivariate.image import JointRecurrencePlot

from sklearn.preprocessing import MinMaxScaler

'''
    Dataset obtained from
    https://zenodo.org/records/10017718
'''
df = pd.read_csv('./processed_asimow_dataset.csv')

'''
    We remove those instances that do not have labels assinged (labels = 1)
    
'''
df = df.drop(df[df['labels'] == -1].index)

'''
    Get Voltage + Current time series (+ labels)
'''
v_columns = [col for col in df.columns if 'V' in col]
c_columns = [col for col in df.columns if 'I' in col]

voltages = np.asarray(df.loc[:, v_columns].values)
currents = np.asarray(df.loc[:, c_columns].values)

labels = np.asarray(df.iloc[:, 2].values)
labels = pd.get_dummies(labels)

'''
    We apply downsampling
'''
downsampling_factor = 4

df_voltages = pd.DataFrame(voltages).T
df_voltages = df_voltages.groupby(df_voltages.index // downsampling_factor).last()
voltages = np.asarray(df_voltages.T.values)

df_currents = pd.DataFrame(currents).T
df_currents = df_currents.groupby(df_currents.index // downsampling_factor).last()
currents = np.asarray(df_currents.T.values)

'''
    Train Test Split
'''
X_voltage, x_voltage, X_current, x_current, Y, y = train_test_split(voltages, currents, labels, test_size=0.2, random_state=42)

'''
    Normalisation
'''
scaler = MinMaxScaler()

X_voltage = scaler.fit_transform(X_voltage)
x_voltage = scaler.transform(x_voltage)

X_current = scaler.fit_transform(X_current)
x_current = scaler.fit_transform(x_current)

'''
    GADF
'''
X_gadf_voltage = GramianAngularField(method='difference').fit_transform(X_voltage)
x_gadf_voltage = GramianAngularField(method='difference').fit_transform(x_voltage)

'''
    Obtain All Images (Voltage)
'''
#X_images = [X_gadf_voltage]           
X_images = np.asarray(X_gadf_voltage)
print(X_images.shape)
X_images = np.reshape(X_images, (X_images.shape[0], X_images.shape[1], X_images.shape[2], 1))
print(X_images.shape)

#x_images = [x_gadf_voltage]
x_images = np.asarray(x_gadf_voltage)
x_images = np.reshape(x_images, (x_images.shape[0], x_images.shape[1], x_images.shape[2], 1))


model = keras.Sequential(
    [
        keras.layers.Input(shape=X_images.shape[1:]),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation="relu", name="dense"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation="relu", name="dense2"),
        keras.layers.Dense(2, activation="softmax", name="dense3"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 5
epochs = 100

history = model.fit(
    X_images,
    Y,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_split=0.2,
)

pd.DataFrame(model.predict(x_images)).to_csv('./Results/predictions_GADFVoltage2CNN.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_GADFVoltage2CNN.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_GADFVoltage2CNN.csv', index=False)