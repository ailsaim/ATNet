import keras.backend as K
from keras.layers import Input, Conv2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Concatenate, \
    Reshape, Lambda,GlobalAveragePooling2D,multiply
from keras.models import Model
from sub_pixel_convolution import pixel_shuffler
from custom_layers.unpooling_layer import Unpooling
from keras.applications.resnet50 import ResNet50
from migrate_Resnet50 import migrate_model

def build_encoder_decoder(hh,ww):
    # encoder
    resNet = ResNet50(include_top=False,weights= None, input_tensor=None, input_shape=(hh,ww,4), pooling=None)
    # migrate_model(resNet)
    x = resNet.output
    orig_5 = resNet.get_layer('activation_49').output
    orig_5 = Conv2D(512, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(orig_5)
    orig_4 = resNet.get_layer('activation_40').output
    orig_4 = Conv2D(512, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(orig_4)
    orig_3 = resNet.get_layer('activation_22').output
    orig_3 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(orig_3)
    orig_2 = resNet.get_layer('activation_10').output
    orig_2 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(orig_2)
    orig_1 = resNet.get_layer('activation_1').output
    orig_1 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(orig_1)

    #decoder
    #stage_one
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,512))(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    attention_feature = multiply([orig_5, x])
    #concate
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)

    #add scale transfer
    x = Conv2D(2048, (1,1),padding='same',use_bias=False)(x)
    x = pixel_shuffler((2,2))(x)

    #stage_two
    x = Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 512))(x)
    x = Conv2D(512, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    attention_feature = multiply([orig_4, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)

    x = pixel_shuffler((2, 2))(x)

    # stage_three
    x = Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 256))(x)
    x = Conv2D(256, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    attention_feature = multiply([orig_3, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)

    x = pixel_shuffler((2, 2))(x)

    # stage_four
    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 128))(x)
    x = Conv2D(128, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    attention_feature = multiply([orig_2, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)

    x = pixel_shuffler((2, 2))(x)

    # stage_five
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    high_feature = x
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 64))(x)
    x = Conv2D(64, (1,1), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    attention_feature = multiply([orig_1, x])
    the_shape = K.int_shape(attention_feature)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(attention_feature)
    xReshaped = Reshape(shape)(high_feature)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)

    x = Conv2D(256, (1,1),padding='same',use_bias=False)(x)
    x = pixel_shuffler((2, 2))(x)

    # stage_six
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    #prediction layer
    x = Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)

    model = Model(inputs=resNet.input, outputs=x)
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

if __name__ == '__main__':

    encoder_decoder = build_encoder_decoder(320,320)
    refinement = build_refinement(encoder_decoder)
    print(refinement.summary())

    K.clear_session()


