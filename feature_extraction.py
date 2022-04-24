from tensorflow import keras
from keras.models import Sequential
import json

layers = ['conv2d_input', 'conv2d', 'batch_normalization', 'conv2d_1', 'batch_normalization_1', 'max_pooling2d',
'conv2d_2', 'dropout', 'batch_normalization_2', 'conv2d_3', 'batch_normalization_3', 'max_pooling2d_1', 'flatten']

with open('./artifacts/cnn_model.json', 'r') as json_file:
    json_saved_model = json_file.read()
    CNN = keras.models.model_from_json(json_saved_model)
    CNN.load_weights('./artifacts/weights1.hdf5')
    CNN.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        
extractor = Sequential()
for layer in layers:
    cnn_layer = CNN.get_layer(layer)
    extractor.add(cnn_layer)
