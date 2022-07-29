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

import math
import argparse

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator


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
        mf_embed_item = Embedding(input_dim=num_users,
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
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="4g",
                        help='The executor memory.')
    parser.add_argument('--num_executors', type=int, default=2,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="4g",
                        help='The driver memory.')
    parser.add_argument('--backend', type=str, default="tf2",
                        help='The backend of TF2 Estimator, either ray or spark')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
    parser.add_argument('--model_dir', default='./', type=str,
                        help='The directory to save the trained model')
    parser.add_argument('--data_dir', type=str, default="./movielens", help='data directory')
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
                               object_store_memory="10g", extra_params={"dashboard-port": "11281", "min-worker-port": "30000", "max-worker-port": "33333", "metrics-export-port": "10010"}, conf={"spark.executorEnv.ARROW_LIBHDFS_DIR": "/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64/"})
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit", extra_params={"dashboard-port": "11281", "min-worker-port": "30000", "max-worker-port": "33333", "metrics-export-port": "10010"})
    else:
        raise Exception("cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                        " 'spark-submit', but got " + args.cluster_mode)

    if args.backend == "tf2":
        save_path = args.model_dir + "ncf.ckpt"
    elif args.backend == "spark":
        save_path = args.model_dir + "ncf.h5"
    else:
        raise Exception("backend should be either 'tf2' or 'spark', but got " + args.backend)

    import random
    from pyspark.sql.types import StructType, StructField, IntegerType
    from bigdl.orca import OrcaContext
    spark = OrcaContext.get_spark_session()

    num_users, num_items = 6000, 3000
    rdd = sc.range(0, 50000).map(
        lambda x: [random.randint(0, num_users-1), random.randint(0, num_items-1), random.randint(0, 4)])
    schema = StructType([StructField("user", IntegerType(), False),
                         StructField("item", IntegerType(), False),
                         StructField("label", IntegerType(), False)])
    data = spark.createDataFrame(rdd, schema)
    train, test = data.randomSplit([0.8, 0.2], seed=1)

    config = {"lr": args.lr, "inter_op_parallelism": 4, "intra_op_parallelism": args.executor_cores}

    def model_creator(config):
        import tensorflow as tf

        model = build_model(num_users, num_items, 5)
        print(model.summary())
        optimizer = tf.keras.optimizers.Adam(config["lr"])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])
        return model

    steps_per_epoch = math.ceil(train.count() / args.batch_size)
    val_steps = math.ceil(test.count() / args.batch_size)

    estimator = Estimator.from_keras(model_creator=model_creator,
                                     verbose=False,
                                     config=config,
                                     backend=args.backend,
                                     model_dir=args.model_dir)
    estimator.fit(train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  feature_cols=['user', 'item'],
                  label_cols=['label'],
                  steps_per_epoch=steps_per_epoch,
                  validation_data=test,
                  validation_steps=val_steps)

    predictions = estimator.predict(test,
                                    batch_size=args.batch_size,
                                    feature_cols=['user', 'item'],
                                    steps=val_steps)
    print("Predictions on validation dataset:")
    predictions.show(5, truncate=False)

    print("Saving model to: ", save_path)
    estimator.save(save_path)

    # load with estimator.load(args.save_path)

    stop_orca_context()
