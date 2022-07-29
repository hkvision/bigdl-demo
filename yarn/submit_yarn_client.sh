export BIGDL_HOME=/home/kai/hk/bigdl-2.0/
export SPARK_HOME=/home/kai/hk/spark-2.4.6-bin-hadoop2.7
export SPARK_VERSION=2.4.6
export BIGDL_VERSION=2.0.0

${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip \
    --conf spark.pyspark.driver.python=/home/kai/anaconda3/envs/orca-hk/bin/python \
    --conf spark.pyspark.python=environment/bin/python \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64 \
    ncf_train.py --cluster_mode spark-submit --backend spark --model_dir hdfs://172.16.0.105:8020/yushan/
