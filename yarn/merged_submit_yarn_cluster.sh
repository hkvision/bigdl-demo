export BIGDL_HOME=/home/kai/hk/bigdl-2.0/
export SPARK_HOME=/home/kai/hk/spark-2.4.6-bin-hadoop2.7
export SPARK_VERSION=2.4.6
export BIGDL_VERSION=2.0.0

${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --driver-cores 4 \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.yarn.appMasterEnv.http_proxy=http://child-prc.intel.com:913 \
    --conf spark.yarn.appMasterEnv.https_proxy=http://child-prc.intel.com:913 \
    --conf spark.yarn.appMasterEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64 \
    --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64 \
    --jars ${BIGDL_HOME}/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar,${BIGDL_HOME}/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar,${BIGDL_HOME}/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    ncf_train.py --cluster_mode spark-submit --backend spark --model_dir hdfs://172.16.0.105:8020/yushan/
