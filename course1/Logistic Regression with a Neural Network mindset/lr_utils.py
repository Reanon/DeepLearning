import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features,保存的是训练集里的图像数据
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels，训练集里的图像对应的分类值，1是猫，0不是猫

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features，测试集里面的图像数据
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels，测试集里的对应的分类值

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes，以bytes类型保存的两个字符串数据：[b'non-cat' b'cat']

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(z):
    """
    :param z: 任何大小的标量或numpy.
    :return: sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zero(dim):
    """
    此函数为w创建一个维度为（dim,1)的0向量，并将b初始化为0。
    :param dim:我们想要的w矢量的大小
    :return:
    w ：维度为（dim,1)的初始化向量
    b : 初始化的标量
    """
    w = np.zeros(shape=(dim, 1))
    b = 0
    #使用断言来确保我要的数据是正确的
    assert (w.shape == (dim, 1))#w的维度是（dim,1)
    assert (isinstance(b, float) or isinstance(b, int))#b的类型为float或者int

    return w, b


def propagate(w, b, X, Y):
    """
    实现前向和后向传播的成本函数及其梯度
    :param w: 权重，大小不等的数组（num_px*num_px*3，1)
    :param b: 偏差，一个标量
    :param X: 矩阵类型为（num_px*num_px*3，训练集数量)
    :param Y: 真正的“标签”矢量（如果为猫为1，非猫为0），矩阵维度为（1，训练集数量）
    :return:
        cost 逻辑回归的负对数似然成本
        dw :相对于w的损失梯度
        db ：相对于b的损失梯度
    """
    m = X.shape[1]
    #正向传播
    A = sigmoid(np.dot(w.T, X) + b)#激活函数
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1 - A))) #成本函数
    #反向传播
    dz = A - Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)
    #使用断言保证我的数据正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    #创建一个字典，把dw和db保存起来
    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost


def optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False):
    """
    运行梯度下降算法来优化w和b
    :param w: 权重，大小不等的数组（num_px*num_px*3，1)
    :param b: 偏差，一个标量
    :param X: 矩阵类型为（num_px*num_px*3，训练集数量)
    :param Y: 真正的“标签”矢量（如果为猫为1，非猫为0），矩阵维度为（1，训练集数量）
    :param num_iterations:优化循环迭代次数
    :param learning_rate:梯度下降更新规则的学习率
    :param print_cost:每一百步打印一次损失值
    :return:
        params:包含权重w 和b的字典
        grads:包含权重
    notes:
    需要写下两个步骤并遍历它们：
        1、计算当前参数的成本和梯度，使用propagate().
        2、使用w和b的梯度下降法更新参数
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印成本数据
        if print_cost and i % 100 == 0:
            print("迭代次数： %i,误差值：%f" % (i, cost))
    params = {
        "w": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return params, grads, costs


def predict(w, b, X):
    """
    使用学习逻辑回归参数logistic(w,b)预测标签是0还是1
     :param w: 权重，大小不等的数组（num_px*num_px*3，1) 
     :param b: 偏差，一个标量                       
     :param X: 矩阵类型为（num_px*num_px*3，训练集数量)  
     :return:
        Y_prediction:包含X中所有图片的所有预测值【0或1】的一个numpy数组（向量
    """
    m = X.shape[1] #图片的数量
    Y_predition = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    #预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        #将概率a[0,i]转换为实际预测值p[0,i]
        Y_predition[0, i] = 1 if A[0, i] > 0.5 else 0
    #使用断言
    assert (Y_predition.shape == (1, m))

    return Y_predition


def model(X_train, Y_train, X_test, Y_test, num_iteration = 200, learning_rate = 0.5, print_cost = False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    :param X_train:numpy数组，维度为（num_px * num_px * 3, m_train)的训练集
    :param Y_train:numpy数组，维度为（1, m_train)的训练标签集
    :param X_test:numpy数组，维度为（num_px * num_px * 3, m_test)的测试集
    :param Y_test:numpy数组，维度为（1, m_test)的测试标签集
    :param num_iteration:用于优化参数的迭代次数
    :param learning_rate:用于optimize()更新规则中使用的学习速率
    :param print_cost:设置true时，以每100次迭代打印一次成本
    :return:
        d :包含有关模型信息的字典
    """
    w, b = initialize_with_zero(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iteration, learning_rate, print_cost)
    #从字典"参数“中检索参数w,b
    w, b = parameters["w"], parameters["b"]

    #预测测试、训练集的例子
    Y_prediction_test = predict(w, b , X_test)
    Y_prediction_train = predict(w, b , X_train)

    #打印训练后的准确性
    print("训练集的准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集的准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iteration": num_iteration
    }
    return d

