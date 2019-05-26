# JNUNN
JNU Neural Network Course Intel Image Classification

[https://www.kaggle.com/puneet6060/intel-image-classification](https://www.kaggle.com/puneet6060/intel-image-classification)

## 网络训练与测试

    ./cnn.py [-h] [-r model_path] [-i iteration_times] [-l logging_step] [-b batch_size] [-p predict_record_path]


### 参数

    -r --restore                    要加载的模型存档。如果不指定，则从头开始训练模型
    -i --iteration                  训练的迭代次数。默认为10次
    -l --logging_iteration_step     记录数据的间隔。默认为10步
    -b --batch_size                 每批的大小。默认为32
    -p --predict_path               使用模型对指定的数据集进行预测。如果不指定，则表示训练模型。注意此参数需要配合--restore参数加载模型存档
    
    
### 示例

以默认配置开始训练模型：

    ./cnn.py
    
从头开始训练模型，迭代1000次，每批32：

    ./cnn.py -i 1000 -b 32
    
从记录model.ckpt加载模型存档训练模型，迭代1000次，每批64：

    ./cnn.py -r model.ckpt -i 1000 -b 64
    
使用模型model.ckpt对数据集all.tfrecords进行预测，每批64：

    ./cnn.py -r model.ckpt -b 64 -p ./all.tfrecords

### 注意事项

1. 载入模型存档时，存档需放置在model目录下，传入参数时只需要传入文件名即可。
2. 载入要预测的数据集时，需要指定路径。

## 数据集制作

    ./data_preprocessing/tfrecord.py [-h] [-r tfrecord_path] [-c training_set_path] [-v validation_set_path] [-t testing_set_path] [-a] [data_set_path1 data_set_path2...]

### 参数
    -r --tfrecord_path          指定输出数据集的目录路径。默认为../data/record
    -c --train_set              指定训练数据的目录。默认为../data/seg_train
    -v --val_set                指定校验数据的目录。默认为../data/seg_val
    -t --test_set               指定测试数据的目录。默认为../data/test_val
    -a --all_set                任意打包数据，参数后跟若干目录

### 示例
    
制作数据集：

    ./data_preprocessing/tfrecord.py
    
自定义各数据集的目录：

    ./data_preprocessing/tfrecord.py -r "../data/record/" -c "../data/train/" -v "../data/val/" -t "../data/test"
    
打包若干数据：

    ./data_preprocessing/tfrecord.py -r "../data/record/" -a "../data/train/" "../data/val/" "../data/test/"
    