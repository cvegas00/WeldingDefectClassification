import pandas as pd
import numpy as np

import markov as mk

from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
from pyts.multivariate.image import JointRecurrencePlot

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import itertools
import numpy as np

import os

os.environ['JAX_ENABLE_X64'] = '1'

import keras
from keras import layers
from keras import ops

from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler

def get_all_combinations(l):
    subsets = []

    for i in range(1, len(l)+1):
        subsets.extend(itertools.combinations(l, i))

    perms = []

    for subset in subsets:
        current_perms = itertools.permutations(subset)

        for p in current_perms:
            perms.append(p)

    return perms


def train_vit(X, Y):
    print(X.shape[3])
    '''
        Implementation of ViT
    '''
    num_classes = 2
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 1280
    num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
    patch_size = 4  # Size of the patches to be extract from the input images
    num_patches = (X.shape[2] // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [
        2048,
        1024,
    ] 

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=keras.activations.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size

        def call(self, images):
            input_shape = ops.shape(images)
            batch_size = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            channels = input_shape[3]
            num_patches_h = height // self.patch_size
            num_patches_w = width // self.patch_size
            patches = keras.ops.image.extract_patches(images, size=self.patch_size)
            patches = ops.reshape(
                patches,
                (
                    batch_size,
                    num_patches_h * num_patches_w,
                    self.patch_size * self.patch_size * channels,
                ),
            )
            return patches

        def get_config(self):
            config = super().get_config()
            config.update({"patch_size": self.patch_size})
            return config

    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = ops.expand_dims(
                ops.arange(start=0, stop=self.num_patches, step=1), axis=0
            )
            projected_patches = self.projection(patch)
            encoded = projected_patches + self.position_embedding(positions)
            return encoded

        def get_config(self):
            config = super().get_config()
            config.update({"num_patches": self.num_patches})
            return config
        
    def create_vit_classifier():
        inputs = keras.Input(shape=input_shape)
        # Create patches.
        patches = Patches(patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def run_experiment(model):
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
            ],
        )

        checkpoint_filepath = "/tmp/checkpoint.weights.h5"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        history = model.fit(
            x=X,
            y=Y,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )

        return history, model

    vit_classifier = create_vit_classifier()
    history, model = run_experiment(vit_classifier)

    return history, model

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
downsampling_factor = 5

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
    Obtain Joint Recurrence Images
'''
multivariate = np.asarray([X_voltage, X_current])
multivariate = np.reshape(multivariate, (multivariate.shape[1], multivariate.shape[0], multivariate.shape[2]))

X_multivariate = JointRecurrencePlot().transform(multivariate)

multivariate = np.asarray([x_voltage, x_current])
multivariate = np.reshape(multivariate, (multivariate.shape[1], multivariate.shape[0], multivariate.shape[2]))

x_multivariate = JointRecurrencePlot().transform(multivariate)

'''
    Obtain All Images (RP Voltatge)
'''
X_images = [X_recurrence_voltage]
X_images = np.asarray(X_images)
X_images = np.reshape(X_images, (X_images.shape[1], X_images.shape[2], X_images.shape[3], X_images.shape[0]))

x_images = [x_recurrence_voltage]
x_images = np.asarray(x_images)
x_images = np.reshape(x_images, (x_images.shape[1], x_images.shape[2], x_images.shape[3], x_images.shape[0]))

'''
    Train ViT
'''
history, model = train_vit(X_images, Y)

pd.DataFrame(model.predict(x_images)).to_csv('./Results/predictions_RPVoltatgeViT.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_RPVoltatgeViT.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_RPVoltatgeViT.csv', index=False)

'''
    Obtain All Images (RP Current)
'''
X_images = [X_recurrence_current]
X_images = np.asarray(X_images)
X_images = np.reshape(X_images, (X_images.shape[1], X_images.shape[2], X_images.shape[3], X_images.shape[0]))

x_images = [x_recurrence_current]
x_images = np.asarray(x_images)
x_images = np.reshape(x_images, (x_images.shape[1], x_images.shape[2], x_images.shape[3], x_images.shape[0]))

'''
    Train ViT
'''
history, model = train_vit(X_images, Y)

pd.DataFrame(model.predict(x_images)).to_csv('./Results/predictions_RPCurrentViT.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_RPCurrentViT.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_RPCurrentViT.csv', index=False)

'''
    Obtain All Images (JRP Current)
'''
X_images = [X_multivariate]
X_images = np.asarray(X_images)
X_images = np.reshape(X_images, (X_images.shape[1], X_images.shape[2], X_images.shape[3], X_images.shape[0]))

x_images = [x_multivariate]
x_images = np.asarray(x_images)
x_images = np.reshape(x_images, (x_images.shape[1], x_images.shape[2], x_images.shape[3], x_images.shape[0]))

'''
    Train ViT
'''
history, model = train_vit(X_images, Y)

pd.DataFrame(model.predict(x_images)).to_csv('./Results/predictions_RPMultivariateViT.csv', index=False, header=False)
pd.DataFrame(y).to_csv('./Results/true_labels_RPMultivariateViT.csv', index=False, header=False)
pd.DataFrame(history.history).to_csv('./Results/history_RPMultivariateViT.csv', index=False)