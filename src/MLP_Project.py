import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#Embeddings as dataset (instead of straight to pickle)
x = np.load("embeddings.npy")
y = np.load("labels.npy")

#Shuffle due to overfitting
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Pickle file moved to here instead of the training
with open("face_model_tflite.pickle", "wb") as f:
    pickle.dump(le,f)

#MLP

model = Sequential([
    Dense(256, activation='relu', input_shape=(x.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(
    optimizer = Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x, y_encoded,
    epochs = 32,
    batch_size = 32,
    validation_split = 0.2,
    shuffle = True
)

model.save("mlp_classifier.h5")
