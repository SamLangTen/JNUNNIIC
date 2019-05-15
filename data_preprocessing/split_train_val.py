'''将训练集分为37分为训练验证集'''

import os
import shutil
import random

def clear_dir(dir):
    shutil.rmtree(dir)
    os.mkdir(dir)

train_path = '../data/seg_train/'
val_path = '../data/seg_val/'
seed = 0.3

if not os.path.exists(val_path):
    os.mkdir(val_path)

classes = os.listdir(train_path)

# 每一类分30%到验证集
for class_name in classes:
    # 当前处理类别的路径
    cur_path = train_path + class_name + '/'

    img_num = len(os.listdir(cur_path))
    val_img_num = int(img_num*seed)

    # 目的文件夹
    dst_path = val_path + class_name + '/'
    # 源文件
    src_path = cur_path
    clear_dir(dst_path)
    print("======================spliting class {}======================".format(class_name))
    for i in range(0,val_img_num):
        index = random.randint(0, len(os.listdir(cur_path))-1)
        imgname = os.listdir(cur_path)[index]
        print("===========moving image {}===========".format(imgname))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        shutil.move(src_path + imgname, dst_path + imgname)

print("======================split over======================")
