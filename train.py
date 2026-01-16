from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, Adadelta, Nadam, RMSprop

import os
import pandas as pd
import numpy as np
import csv

import time

from __init__ import get_train_test_data
from Factory import factory
from metrics import rmse, mape, mae, get_model_save_path
import argparse

from keras import backend as K
import tensorflow as tf
from config import Config

from keras.callbacks import TensorBoard
from keras.utils import set_random_seed

def main():
    conf = Config("config_fig.yaml")

    set_random_seed(0)

    tensorboard = TensorBoard(log_dir="logs/{}".format(conf.model_name))

    print(conf.observe_length)

    data, arm_shape, train_xs, train_ys, train_arms, train_xp, train_xt, train_xe, \
        train_vehicle_type, train_engine_config, train_gen_weight, \
        test_xs, test_ys, test_arms, test_xp, test_xt, test_xe, \
        test_vehicle_type, test_engine_config, test_gen_weight = \
        get_train_test_data(conf, need_road_network_structure_matrix=True)

    print('************** Train - Predict **********')
    print(
        'train_xs:', train_xs.shape,
        'test_xs:', test_xs.shape,
        'train_xp:', train_xp.shape,
        'test_xp:', test_xp.shape,
        'test_xe:', test_xe.shape,
        'train_ys:', train_ys.shape,
        'test_ys:', test_ys.shape,
        'train_vehicle_type: ', train_vehicle_type.shape,
        'train_engine_config: ', train_engine_config.shape,
        'train_gen_weight: ', train_gen_weight.shape
    )
    
    if conf.use_loopup:
        train_xs = [train_xs, train_arms]
        test_xs = [test_xs, test_arms]

    if conf.ablate_vehicle:
        if not isinstance(train_xs, list):
            train_xs = [train_xs]
            test_xs = [test_xs]

        train_xs += [train_vehicle_type, train_engine_config, train_gen_weight]
        test_xs  += [test_vehicle_type, test_engine_config, test_gen_weight]

    if conf.use_externel:
        if conf.observe_p != 0:
            if isinstance(train_xs, list):
                train_xs += [train_xp]
                test_xs += [test_xp]
            else:
                train_xs = [train_xs, train_xp]
                test_xs = [test_xs, test_xp]

        if conf.observe_t != 0:
            if isinstance(train_xs, list):
                train_xs += [train_xt]
                test_xs += [test_xt]
            else:
                train_xs = [train_xs, train_xt]
                test_xs = [test_xs, test_xt]

        if conf.observe_p != 0 or conf.observe_t != 0:
            train_xs += [train_xe]
            test_xs += [test_xe]

    model_save_path = get_model_save_path(conf)
    print('arm shape', arm_shape)

    start = time.time()

    model = factory.get_model(conf, arm_shape)
    print(model)
    
    if conf.use_cache_model and os.path.exists(model_save_path):
        model.load_weights(model_save_path)
    else:
        adam = Adam(lr=conf.learning_rate)
        model.compile(
            adam,
            loss="mean_squared_error",
            metrics=["mae", "mape", "accuracy"]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=conf.early_stopping
        )

        check_points = ModelCheckpoint(
            model_save_path,
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False
        )

        model.summary()

        with open('model_sumamry.txt', 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        if conf.use_externel:
            train_ys = train_ys[:, :-1, :]
            test_ys = test_ys[:, :-1, :]
        else:
            train_ys = train_ys[:, :-1, :]
            test_ys = test_ys[:, :-1, :]

        print('train ys', test_ys.shape)

        history = model.fit(
            train_xs,
            train_ys,
            verbose=2,
            epochs=conf.epochs,
            batch_size=conf.batch_size,
            callbacks=[early_stopping, check_points, tensorboard],
            validation_data=[test_xs, test_ys]
        )

        model.load_weights(model_save_path)

    print(len(test_xs))
    predict = model.predict(test_xs, batch_size=conf.batch_size)

    predict = data.min_max_scala.inverse_transform(predict)
    y_true = data.min_max_scala.inverse_transform(test_ys)

    end = time.time()
    print("TRAINING TIME COST", (end - start))

    print("test")
    print(" predict")
    print(predict.shape)
    print(" real")
    print(y_true.shape)

    v_rmse = rmse(predict, y_true)
    v_mae = mae(predict, y_true)
    v_mape = mape(predict, y_true)

    print("RMSE:", v_rmse)
    print("MAE:", v_mae)
    print("MAPE:", v_mape)

    with open('predicted.csv', 'w') as outfile:
        for data_slice in predict:
            outfile.write('# New slice\n')
            np.savetxt(outfile, data_slice)
    
    with open('real.csv', 'w') as outfile:
        for data_slice in y_true:
            outfile.write('# New slice\n')
            np.savetxt(outfile, data_slice)


if __name__ == '__main__':
    main()
