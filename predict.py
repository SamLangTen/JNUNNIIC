from cnn import predict_model



if __name__ == '__main__':
    model_path = './model/model-320.ckpt'
    test_dir = './data/seg_test/'  # 要测试的图片放在这个路径下
    label_list = ['sea','forest','street','mountain','buildings','glacier']
    predict_model(test_dir, model_path, label_list)