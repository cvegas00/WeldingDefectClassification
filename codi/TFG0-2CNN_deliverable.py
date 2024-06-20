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
    Obtain Recurrence Images
'''
X_recurrence_voltage = RecurrencePlot().transform(X_voltage)
x_recurrence_voltage = RecurrencePlot().transform(x_voltage)

X_recurrence_current = RecurrencePlot().transform(X_current)
x_recurrence_current = RecurrencePlot().transform(x_current)

'''
    Obtain GASF
'''
X_gasf_voltage = GramianAngularField(method='summation').fit_transform(X_voltage)
x_gasf_voltage = GramianAngularField(method='summation').fit_transform(x_voltage)

X_gasf_current = GramianAngularField(method='summation').fit_transform(X_current)
x_gasf_current = GramianAngularField(method='summation').fit_transform(x_current)

'''
    Obtain GADF
'''
X_gadf_voltage = GramianAngularField(method='difference').fit_transform(X_voltage)
x_gadf_voltage = GramianAngularField(method='difference').fit_transform(x_voltage)

X_gadf_current = GramianAngularField(method='difference').fit_transform(X_current)
x_gadf_current = GramianAngularField(method='difference').fit_transform(x_current)

'''
    Obtain MTF
'''
X_mtf_voltage = MarkovTransitionField(n_bins=20).fit_transform(X_voltage)
x_mtf_voltage = MarkovTransitionField(n_bins=20).fit_transform(x_voltage)

X_mtf_current = MarkovTransitionField(n_bins=20).fit_transform(X_current)
x_mtf_current = MarkovTransitionField(n_bins=20).fit_transform(x_current)

'''
    Obtain Joint Recurrence Images
'''
multivariate = np.asarray([X_voltage, X_current])
multivariate = np.reshape(multivariate, (multivariate.shape[1], multivariate.shape[0], multivariate.shape[2]))

X_multivariate = JointRecurrencePlot().transform(multivariate)

multivariate = np.asarray([x_voltage, x_current])
multivariate = np.reshape(multivariate, (multivariate.shape[1], multivariate.shape[0], multivariate.shape[2]))

x_multivariate = JointRecurrencePlot().transform(multivariate)

n_states = 50
X_markov, _ = mk.get_MTM_images(pd.DataFrame(X_voltage), n_states, channels=1)
x_markov, test_states = mk.get_MTM_images(pd.DataFrame(x_voltage), n_states, channels=1)

'''
    Obtain All Images
'''
X_images = [X_recurrence_voltage, X_recurrence_current, X_gasf_voltage, X_gasf_current, X_gadf_voltage, 
                X_gadf_current, X_mtf_voltage, X_mtf_current, X_multivariate, X_markov]           
X_images = np.asarray(X_images)
print(X_images.shape)
X_images = np.reshape(X_images, (X_images.shape[1], X_images.shape[2], X_images.shape[3], X_images.shape[0]))
print(X_images.shape)

x_images = [x_recurrence_voltage, x_recurrence_current, x_gasf_voltage, x_gasf_current, x_gadf_voltage, 
                x_gadf_current, x_mtf_voltage, x_mtf_current, x_multivariate, x_markov]
x_images = [x_images]
x_images = np.asarray(x_images)
x_images = np.reshape(x_images, (x_images.shape[1], x_images.shape[2], x_images.shape[3], x_images.shape[0]))


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
epochs = 10

history = model.fit(
    X_images,
    Y,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_split=0.2,
)

pd.DataFrame(model.predict(x_images)).to_csv('./Results/predictions_all2CNN.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_all2CNN.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_all2CNN.csv', index=False)