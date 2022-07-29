#python ncf_train.py --cluster_mode yarn --num_executor 2 --executor_cores 4 --executor_memory 4g --backend tf2 --model_dir hdfs://172.16.0.105:8020/user/kai/

python ncf_train.py --cluster_mode yarn --num_executor 2 --executor_cores 4 --executor_memory 4g --backend spark --model_dir hdfs://172.16.0.105:8020/yushan/
