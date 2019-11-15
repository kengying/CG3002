import numpy as np

from tensorflow.keras.models import load_model

def cnn_load():
    model = load_model('cnn_pretrained_model_256.h5')
    return model

def cnn_predict(model, x_test):
    y_pred = model.predict_classes(np.array([x_test]), batch_size=32, verbose=0)
    return y_pred
