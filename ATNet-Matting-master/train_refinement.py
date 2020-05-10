import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD

from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator_modify import train_gen, valid_gen
from ATNet_MobileNetV2 import build_encoder_decoder, build_refinement
from utils import alpha_prediction_loss, get_available_cpus
from keras import optimizers
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':
    checkpoint_models_path = 'models_deepImageMatting__mobile_net_v2_pretrain/'

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'refinement.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    pretrained_path = 'models_deepImageMatting__mobile_net_v2_pretrain/model.96-0.0743.hdf5'
    encoder_decoder = build_encoder_decoder(320,320)
    encoder_decoder.load_weights(pretrained_path)
    # fix encoder-decoder part parameters and then update the refinement part.
    for layer in encoder_decoder.layers:
        layer.trainable = False

    refinement = build_refinement(encoder_decoder)

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #nadam = optimizers.Nadam(lr = 0.0002)
    refinement.compile(optimizer='nadam', loss=alpha_prediction_loss)

    print(refinement.summary())

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 2))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    cc = refinement.fit_generator(train_gen(),
                             steps_per_epoch=num_train_samples // batch_size,
                             validation_data=valid_gen(),
                             validation_steps=num_valid_samples // batch_size,
                             epochs=epochs,
                             verbose=1,
                             callbacks=callbacks,
                             use_multiprocessing=True,
                             workers=workers
                             )
    flops = tf.profiler.profile(cc, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))
