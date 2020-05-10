import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator_modify import train_gen, valid_gen
from ATNet_MobileNetV2 import build_encoder_decoder
from utils import overall_loss, get_available_cpus, get_available_gpus,compositional_loss,alpha_prediction_loss,over_all_add_per_loss,perceptual_loss
from keras import optimizers

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", help="path to save checkpoint model files")
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    args = vars(ap.parse_args())
    checkpoint_path = args["checkpoint"]
    pretrained_path = args["pretrained"]
    if checkpoint_path is None:
        checkpoint_models_path = 'models_ATNet_VGG16_pretrain/'
    else:
        # python train_encoder_decoder.py -c /mnt/Deep-Image-Matting/models/
        checkpoint_models_path = '{}/'.format(checkpoint_path)

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_loss']))

    # Load our model, added support for Multi-GPUs
    pretrained_path = 'models_ATNet_VGG16_pretrain/model.94-0.0479.hdf5'
    if pretrained_path is not None:
        new_model = build_encoder_decoder(320,320)
        new_model.load_weights(pretrained_path)
    else:
        new_model = build_encoder_decoder(320,320)
        migrate.migrate_model(new_model)

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    nadam = optimizers.Nadam(lr = 0.0002)
    new_model.compile(optimizer=nadam,loss = over_all_add_per_loss)

    print(new_model.summary())


    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 2))
    #workers = 4
    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    new_model.fit_generator(train_gen(),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=workers
                            )
