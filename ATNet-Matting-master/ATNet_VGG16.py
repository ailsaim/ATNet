import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Concatenate, \
    Reshape, Lambda,GlobalAveragePooling2D,multiply,Add,Conv1D
from keras.models import Model
from sub_pixel_convolution import pixel_shuffler

from custom_layers.unpooling_layer import Unpooling
from keras.applications.vgg16 import VGG16
from migrate_VGG16 import migrate_model
def build_encoder_decoder(hh,ww):
    # Encoder
    base_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(hh, ww, 4), pooling=None)

    migrate_model(base_model)
    x = base_model.output
    orig_5 = base_model.get_layer('block5_pool').output

    orig_4 = base_model.get_layer('block4_pool').output

    orig_3 = base_model.get_layer('block3_pool').output

    orig_2 = base_model.get_layer('block2_pool').output

    orig_1 = base_model.get_layer('block1_pool').output



    # Decoder

    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,512))(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    # low_feature = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',bias_initializer='zeros')(orig_5)
    attention_feature = multiply([orig_5, x])

    #concate
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    # x = UpSampling2D(size=(2, 2))(x)

    #add scale transfer
    x = Conv2D(2048, (1,1),padding='same',use_bias=False)(x)
    x = pixel_shuffler((2,2))(x)

    #block_2
    x = Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 512))(x)
    x = Conv2D(512, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    # low_feature = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                      bias_initializer='zeros')(orig_4)
    attention_feature = multiply([orig_4, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    # x = UpSampling2D(size=(2, 2))(x)

    x = pixel_shuffler((2, 2))(x)

    # block_3
    x = Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 256))(x)
    x = Conv2D(256, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    # low_feature = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                      bias_initializer='zeros')(orig_3)
    attention_feature = multiply([orig_3, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    # x = UpSampling2D(size=(2, 2))(x)

    x = pixel_shuffler((2, 2))(x)

    # block_4
    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 128))(x)
    x = Conv2D(128, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    # low_feature = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                      bias_initializer='zeros')(orig_2)
    attention_feature = multiply([orig_2, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    # x = UpSampling2D(size=(2, 2))(x)

    x = pixel_shuffler((2, 2))(x)

    # block_5
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 64))(x)
    x = Conv2D(64, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    # low_feature = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                      bias_initializer='zeros')(orig_1)
    attention_feature = multiply([orig_1, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    # x = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(256, (1,1),padding='same',use_bias=False)(x)
    x = pixel_shuffler((2, 2))(x)
    # block_6
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def build_refinement(encoder_decoder):
    input_tensor = encoder_decoder.input

    input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)

    x = Concatenate(axis=3)([input, encoder_decoder.output])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='refinement_pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)

    model = Model(inputs=input_tensor, outputs=x)
    return model

