import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  # 读取训练集数据
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 训练集特征 (m_train（209）,num_px, num_px, 3)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 训练集标签 (m_train(209),1)

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")  # 读取测试集数据
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 测试集特征 (m_test(50),num_px, num_px, 3)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 测试集标签 (m_test(50),1)

    classes = np.array(test_dataset["list_classes"][:])  # 字符串numpy数组，包含'cat'和'noncat'

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 维度变为(1,m_train(209))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # 维度变为(1,m_test(50))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
