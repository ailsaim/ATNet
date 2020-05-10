import keras.backend as K
import numpy as np

from config import channel
from keras.applications.mobilenet_v2 import MobileNetV2
from ATNet_MobileNetV2 import build_encoder_decoder, build_refinement

def migrate_model(new_model):
    model_path = 'models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    old_model =MobileNetV2(include_top=False,weights= None, input_tensor=None, input_shape=(224,224,3), pooling=None)
    old_model.load_weights(model_path)

    old_layers = [l for l in old_model.layers]
    new_layers = [l for l in new_model.layers]

    old_conv1_1 = old_model.get_layer('Conv1')
    old_weights = old_conv1_1.get_weights()[0]
    print(np.shape(old_weights))
    # old_biases = old_conv1_1.get_weights()[1]
    new_weights = np.zeros((3, 3, channel, 32), dtype=np.float32)
    new_weights[:, :, 0:3, :] = old_weights
    new_weights[:, :, 3:channel, :] = 0.0
    new_conv1_1 = new_model.get_layer('Conv1')
    new_conv1_1.set_weights([new_weights])

    for i in range(3, 152):
        old_layer = old_layers[i]
        new_layer = new_layers[i]
        new_layer.set_weights(old_layer.get_weights())

    del old_model


if __name__ == '__main__':
    # old_model = MobileNetV2(include_top=False, weights=None, input_tensor=None, input_shape=(320, 320, 4), pooling=None)
    # print(old_model.summary())
    # old_layers = [l for l in old_model.layers]
    # print(len(old_layers))
    # model = MobileNetv2(224,224)
    # migrate_model(model)
    encoder_decoder = build_encoder_decoder(320,320)
    migrate_model(encoder_decoder)
    final = build_refinement(encoder_decoder)
    print(final.summary())

    # K.clear_session()
