import keras.backend as K
import numpy as np

from config import channel
from keras.applications.resnet50 import ResNet50


def migrate_model(new_model):
    model_path = 'models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    old_model = ResNet50(include_top=False,weights= None, input_tensor=None, input_shape=(224,224,3), pooling=None)
    old_model.load_weights(model_path)

    # print(old_model.summary())
    old_layers = [l for l in old_model.layers]
    new_layers = [l for l in new_model.layers]

    old_conv1_1 = old_model.get_layer('conv1')
    old_weights = old_conv1_1.get_weights()[0]
    old_biases = old_conv1_1.get_weights()[1]
    new_weights = np.zeros((7, 7, channel, 64), dtype=np.float32)
    new_weights[:, :, 0:3, :] = old_weights
    new_weights[:, :, 3:channel, :] = 0.0
    new_conv1_1 = new_model.get_layer('conv1')
    new_conv1_1.set_weights([new_weights, old_biases])

    for i in range(3, len(old_layers)):
        old_layer = old_layers[i]
        new_layer = new_layers[i]
        new_layer.set_weights(old_layer.get_weights())

    del old_model


if __name__ == '__main__':
    new_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(224, 224, 3), pooling=None)
    print(new_model.summary())
    # migrate_model()

    K.clear_session()
