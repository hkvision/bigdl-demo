## Environment Preparation
- Create a conda environment and install the packages in requirements.txt. (pyarrow is used for saving to hdfs for spark backend)
- Pack the current conda environment: `conda pack -o environment.tar.gz`

## Download BigDL 2.0
- Download BigDL all-in-one package: https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.0.0/bigdl-assembly-spark_2.4.6-2.0.0.zip
- `unzip bigdl-assembly-spark_2.4.6-2.0.0.zip -d bigdl-2.0`
- For yarn cluster mode, additionally download the separate jars and *put them into bigdl-2.0 directory*:
  - https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-dllib-spark_2.4.6/2.0.0/bigdl-dllib-spark_2.4.6-2.0.0-jar-with-dependencies.jar
  - https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-orca-spark_2.4.6/2.0.0/bigdl-orca-spark_2.4.6-2.0.0-jar-with-dependencies.jar

## Download Spark 2.4.6
- https://archive.apache.org/dist/spark/spark-2.4.6/spark-2.4.6-bin-hadoop2.7.tgz
- `tar -xzvf spark-2.4.6-bin-hadoop2.7.tgz`

## Scripts
- `run_python.sh`: init_orca_context with yarn-client cluster mode
- `submit_yarn_client.sh`: spark-submit cluster mode for yarn client
- `submit_yarn_cluster.sh`: spark-submit cluster mode for yarn cluster

Remarks:
- Can change backend to tf2 or spark in the scripts.
- Need to change the environment variables (BIGDL_HOME, SPARK_HOME) in the spark-submit scripts. 
- Need to change `spark.pyspark.driver.python` to the Python interpreter on the client node for spark-submit yarn client mode. Can get this path using `which python` for the active conda environment.
- Need to change or remove `spark.executorEnv.ARROW_LIBHDFS_DIR` according to your cluster settings.
- Need to change `--model_dir` for the training script.
- You can remove `ARROW_LIBHDFS_DIR` related environment variables in the script if you are not using CDH. See the issue below for more details.

## Issue Solutions
- https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/known_issues.html#oserror-unable-to-load-libhdfs-libhdfs-so-cannot-open-shared-object-file-no-such-file-or-directory
