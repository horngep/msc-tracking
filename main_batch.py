
from goturn import goturn
from batch_generator import batch_generator
from visualize_video import visualize_video, visualise_calipsa, evaluate
import keras
import time
import pdb
from keras import backend as K

if __name__ == "__main__":


    # Training param
    batchsize = 64
    ep = 20

    # Data (60/20/20)
    num_train_data = 9431
    num_val_data = 3288
    num_test_data = 3158

    # Tensorboard callback $ tensorboard --logdir log/
    tb = keras.callbacks.TensorBoard(log_dir='./log/baseline',
                                    histogram_freq=0,
                                    write_graph=True)
    # Model checkpoint callback
    # cp = keras.callbacks.ModelCheckpoint(filepath='baseline.h5',
    #                                     monitor='val_loss',
    #                                     verbose=0,
    #                                     save_best_only=True, # saving only best model
    #                                     save_weights_only=True,
    #                                     mode='min',
    #                                     period=1)
    # Early stopping callback
    # es = keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                     min_delta=0,
    #                                     patience=5,
    #                                     verbose=0,
    #                                     mode='min')


    # ==================================
    # Train with train only (THIS IS ONLY FOR HYPER PARAM TUNING)
    # history = model.fit_generator(
    #                     generator=batch_generator(batchsize, 'train'),
    #                     steps_per_epoch=int(num_train_data/(batchsize)),
    #                     epochs=ep,
    #                     validation_data=batch_generator(batchsize, 'val'),
    #                     validation_steps=int(num_val_data/(batchsize)),
    #                     callbacks=[tb, cp], # NOTE: tb, cp, es
    #                     )
    # val_loss =  min(history.history['val_loss'])
    # ==================================

    # train with train+val
    lr = 0.005 # [0.001,0.005, 0.01]

    model = goturn(lr)

    # =================================
    #  TRANING
    # history = model.fit_generator(
    #                 generator=batch_generator(batchsize, 'train+val'),
    #                 steps_per_epoch=int(num_train_data/(batchsize)),
    #                 epochs=ep,
    #                 callbacks=[tb],
    #                 )
    # model.save_weights('../temp.h5')
    # ==================================

    # ==================================
    # LOAD
    model.load_weights('../baseline.h5')
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

    evaluate(model)






























#