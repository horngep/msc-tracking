
from goturn import goturn
from batch_generator import batch_generator, batch_generator_imagenet
from visualize_video import visualize_video, visualise_calipsa, evaluate
import keras
import time
import pdb
from keras import backend as K
import os


if __name__ == "__main__":

    # Training param
    batchsize = 64
    ep = 100

    # DATA: alov300 90/10/10
    num_alov_train_data = 12380
    num_alov_val_data = 1778
    num_alov_test_data = 1719

    num_imagenet_train_data = 1952*200
    num_imangenet_val_data = 281*200



    # Tensorboard callback $ tensorboard --logdir log/
    tb = keras.callbacks.TensorBoard(log_dir='./log/imagenet-temp',
                                    histogram_freq=0,
                                    write_graph=True)
    # Model checkpoint callback
    cp = keras.callbacks.ModelCheckpoint(filepath='../imagenet-best.h5',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True, # saving only best model
                                        save_weights_only=True,
                                        mode='min',
                                        period=1)
    # Early stopping callback
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=10,
                                        verbose=0,
                                        mode='min')


    # ==================================
    # training on ImageNet video
    model = goturn(0.001)
    history = model.fit_generator(
                    generator=batch_generator_imagenet(batchsize, 'train'),
                    steps_per_epoch=int(num_imagenet_train_data/batchsize),
                    epochs = ep,
                    validation_data=batch_generator_imagenet(batchsize, 'val'),
                    validation_steps=int(num_imangenet_val_data/batchsize),
                    callbacks=[tb, cp, es], # NOTE: tb, cp, es
                    )
    model.save_weights('../imagenet-final.h5')
    val_loss =  min(history.history['val_loss'])
    # ==================================


    # ==================================
    # Train with train only (THIS IS ONLY FOR HYPER PARAM TUNING)
    # model = goturn(0.001)
    # history = model.fit_generator(
    #                     generator=batch_generator(batchsize, 'train+val'),
    #                     steps_per_epoch=int((num_alov_train_data+num_alov_val_data)/(batchsize)),
    #                     epochs=ep,
    #                     validation_data=batch_generator(batchsize, 'test'),
    #                     validation_steps=int(num_alov_test_data/(batchsize)),
    #                     callbacks=[tb, cp, es], # NOTE: tb, cp, es
    #                     )
    # # val_loss =  min(history.history['val_loss'])
    # model.save_weights('../adam_alov_final.h5')
    # # ==================================

    # train with train+val
    # =================================
    # #  TRANING
    # lr = 0.005 # [0.001,0.005, 0.01]
    #
    # model = goturn(lr)
    # history = model.fit_generator(
    #                 generator=batch_generator_imagenet(batchsize),
    #                 steps_per_epoch=30,
    #                 epochs=ep,
    #                 callbacks=[],
    #                 )
    # model.save_weights('../imagenet-tmp.h5')
    # ==================================

    # # ==================================
    # # LOAD
    # model = goturn(0.005)
    # model.load_weights('../baseline_new_final005.h5')
    # evaluate(model)
    # ==================================


    # Test
    # test_gen = batch_generator(1,'test')
    # Xtest, Ytest = next(test_gen)
    # t0 = time.time()
    # predict = model.predict(Xtest)
    # tt = time.time()
    # print('prediction: ', predict)
    # print('groundtruth: ', Ytest)
    # print('inference time per image,', (tt-t0))
    # #


    # Apply on 1 video to visualised
    # t1 = time.time()
    #
    # # num_frame = visualize_video(model)
    # num_frame = visualise_calipsa(model)
    #
    # t2 = time.time()
    # print('num_frames = ', num_frame)
    # print('runtime (sec) = ', (t2-t1))
    # print('frame_rate (fps) = ', (num_frame/(t2-t1)))
    #
    #































#
