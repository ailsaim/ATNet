import keras.backend as K
import numpy as np

from config import channel
from keras.applications.vgg16 import VGG16


def migrate_model(new_model):
    model_path = 'models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    old_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(320, 320, 3), pooling=None)
    old_model.load_weights(model_path)
    # print(old_model.summary())
    old_layers = [l for l in old_model.layers]
    new_layers = [l for l in new_model.layers]

    old_conv1_1 = old_model.get_layer('block1_conv1')
    old_weights = old_conv1_1.get_weights()[0]
    old_biases = old_conv1_1.get_weights()[1]
    new_weights = np.zeros((3, 3, channel, 64), dtype=np.float32)
    new_weights[:, :, 0:3, :] = old_weights
    new_weights[:, :, 3:channel, :] = 0.0
    new_conv1_1 = new_model.get_layer('block1_conv1')
    new_conv1_1.set_weights([new_weights, old_biases])

    for i in range(2, 19):
        old_layer = old_layers[i]
        new_layer = new_layers[i]
        new_layer.set_weights(old_layer.get_weights())



    del old_model


if __name__ == '__main__':
    # model = build_encoder_decoder(320,320)
    # migrate_model(model)
    # print(model.summary())
    # model.save_weights('models/model_weights.h5')
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None)
    # migrate_model(base_model)
    print(base_model.summary())
    K.clear_session()
