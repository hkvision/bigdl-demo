#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# --epochs 1 --backend spark --executor_cores 32 --batch_size 1280

import math
import argparse
import pandas as pd

from bigdl.dllib.utils.log4Error import *
from bigdl.dllib.feature.dataset import movielens
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator
from bigdl.friesian.feature import FeatureTable


def build_model(num_users, num_items, class_num, layers=[20, 10], include_mf=True, mf_embed=20):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, concatenate, multiply

    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    mlp_embed_user = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                               input_length=1)(user_input)
    mlp_embed_item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                               input_length=1)(item_input)

    user_latent = Flatten()(mlp_embed_user)
    item_latent = Flatten()(mlp_embed_item)

    mlp_latent = concatenate([user_latent, item_latent], axis=1)
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu',
                      name='layer%d' % idx)
        mlp_latent = layer(mlp_latent)

    if include_mf:
        mf_embed_user = Embedding(input_dim=num_users,
                                  output_dim=mf_embed,
                                  input_length=1)(user_input)
        mf_embed_item = Embedding(input_dim=num_items,
                                  output_dim=mf_embed,
                                  input_length=1)(item_input)
        mf_user_flatten = Flatten()(mf_embed_user)
        mf_item_flatten = Flatten()(mf_embed_item)

        mf_latent = multiply([mf_user_flatten, mf_item_flatten])
        concated_model = concatenate([mlp_latent, mf_latent], axis=1)
        prediction = Dense(class_num, activation='softmax', name='prediction')(concated_model)
    else:
        prediction = Dense(class_num, activation='softmax', name='prediction')(mlp_latent)

    model = tf.keras.Model([user_input, item_input], prediction)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCF Training')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The number of cores to use on each executor.')
    parser.add_argument('--executor_memory', type=str, default="4g",
                        help='The amount of memory to allocate on each executor.')
    parser.add_argument('--num_executors', type=int, default=2,
                        help='The number of executors to use in the cluster.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The number of cores to use for the driver.')
    parser.add_argument('--driver_memory', type=str, default="4g",
                        help='The amount of memory to allocate for the driver.')
    parser.add_argument('--backend', type=str, default="ray",
                        help='The backend of TF2 Estimator, either ray or spark.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='The learning rate to train the model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The number of epochs to train the model.')
    parser.add_argument('--batch_size', default=8000, type=int,
                        help='The batch size to train the model.')
    parser.add_argument('--model_dir', default='./', type=str,
                        help='The directory to save the trained model.')
    parser.add_argument('--data_dir', type=str, default="./movielens",
                        help='The directory for the movielens data.')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores,
                               memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               object_store_memory="10g")
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    if args.backend == "ray":
        save_path = args.model_dir + "ncf.ckpt"
    elif args.backend == "spark":
        save_path = args.model_dir + "ncf.h5"
    else:
        invalidInputError(False,
                          "backend should be either 'ray' or 'spark', but got " + args.backend)

    from bigdl.orca.data.pandas import read_csv
    import numpy as np
    full_data = read_csv("./movielens/ml-1m/ratings.dat", sep='::', header=None,
                         names=['user', 'item', 'label'], usecols=[0, 1, 2],
                         dtype={0: np.int32, 1: np.int32, 2: np.int32})

    user_set = set(full_data['user'].unique())
    item_set = set(full_data['item'].unique())

    num_users = max(user_set) + 1
    num_items = max(item_set) + 1


    def update_label(df):
        print("updating label")
        df['label'] = df['label'] - 1
        return df

    # run Python codes on each partition in a data-parallel fashion using `XShards.transform_shard`
    full_data = full_data.transform_shard(update_label)


    def split_train_test(data):
        print("splitting")
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=0.2, random_state=100)
        return train, test


    train_data, test_data = full_data.transform_shard(split_train_test).split()

    config = {"lr": args.lr, "inter_op_parallelism": 4, "intra_op_parallelism": args.executor_cores}

    def model_creator(config):
        import tensorflow as tf

        model = build_model(num_users, num_items, 5)
        optimizer = tf.keras.optimizers.Adam(config["lr"])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])
        return model

    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    val_steps = math.ceil(len(test_data) / args.batch_size)

    estimator = Estimator.from_keras(model_creator=model_creator,
                                     verbose=True,
                                     config=config,
                                     backend=args.backend,
                                     model_dir=args.model_dir)
    estimator.fit(train_data,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  feature_cols=['user', 'item'],
                  label_cols=['label'],
                  steps_per_epoch=steps_per_epoch,
                  validation_data=test_data,
                  validation_steps=val_steps)
    result = estimator.evaluate(train_data,
                                batch_size=args.batch_size,
                                feature_cols=['user', 'item'],
                                label_cols=['label'],
                                num_steps=val_steps)
    print('Evaluation results:')
    for r in result:
        print(r, ":", result[r])

    predictions = estimator.predict(test_data,
                                    batch_size=args.batch_size,
                                    feature_cols=['user', 'item'])
    print("Predictions on validation dataset:")
    print(predictions.rdd.take(5))

    print("Saving model to: ", save_path)
    estimator.save(save_path)

    # load with estimator.load(args.save_path)

    stop_orca_context()
