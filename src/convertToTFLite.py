import tensorflow as tf
model_convert = tf.keras.models.load_model("mlp_classifier.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model_convert)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("mlp_classifier.tflite", "wb") as f:
    f.write(tflite_model)