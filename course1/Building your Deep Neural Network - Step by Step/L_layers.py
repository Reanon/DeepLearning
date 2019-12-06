import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *  # 测试函数

np.random.seed(1)  # 指定随机种子


def initialize_parameters_deep(layer_dims):
    '''
    此函数是为了初始化多层网络参数
    :param layer_dims:包含我们网络中每个图层的节点的列表
    :return:
        parameters:包含参数“W1”,“b1”,“W2”……“WL”,"bL"的字典
            Wl:权重矩阵，维度为（layer_dims[l],layer_dims[l-1]）
            bl:偏向量，维度为（layer_dims[l],1）
    '''
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # 网络的层数
    for i in range(1, L):  # range下标从0开始,如果L=3，则list(range(1,3))=[1,2]
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) / np.sqrt(
            layer_dims[i - 1])  # 用除代替0.01
        # parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01 # 不知道为什么，初始化只能用上面的那个式子，不然训练不动
        parameters['b' + str(i)] = np.zeros(shape=(layer_dims[i], 1))  # 列表使用[]
        # 确保数据正确
        assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))
    return parameters  # 包含的参数数是隐藏层数的两倍


def linear_forward(A, W, b):
    '''
    实现前向传播的线性部分
    :param A:来自上一层（或输入数据）的激活，维度为（上一层节点数，样本数）
    :param W:权重矩阵，维度为（当前层的节点数，上一层的节点数）
    :param b:偏向量，维度为（当前层的节点数，1）
    :return:
        Z:激活函数的输入，也称为预激活参数
        cache:一个包含A,W,b的字典，储存它们以便后向传播的计算
    '''
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)  # cache是一个列表
    return Z, cache


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    '''
    实现linear->activation这一层的前向传播
    :param A_prev:来自上一层（或输入层）的激活，维度为（上一层节点数，样本数）
    :param W:权重矩阵，numpy数组，维度为（当前层节点数量，上一层节点数量）
    :param b:偏向量，numpy阵列，维度为（当前层节点数量，1）
    :param activation:选择在此层中的激活函数，字符串类型，【sigmoid,relu】
    :return:
        A:激活函数的输出，也称为激活后的值
        cache：一个包含'linear_cache'和'activation_cache'的字典，我们需要存储它以有效地计算后向传播
    '''
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)  # linear_cache = (A, W, b)
        A, activation_cache = sigmoid(Z)  # activation_cache = Z
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)  # linear_cache = (A, W, b)
        A, activation_cache = relu(Z)  # activation_cache = Z

    assert (A.shape == (W.shape[0], A.shape[1]))
    cache = (linear_cache, activation_cache)  # (A,W,b,Z)，其实是个一列表

    return A, cache


def L_model_forward(X, parameters):
    '''
    实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION
    也就是前L-1层用relu激活函数，输出层用sigmoid。
    :param X:输入数据，numpy数组，维度为（输入层节点数，样本数）
    :param parameters:包含W1、b1,W2,b2...的字典,是initialize_parameters_deep(layer_dims)的输出，
    :return:
        AL:最后的激活值，也就是Yhat
        caches:包含以下内容的缓存列表：
            linear_relu_forward()的每一个cache(缓存，Z），共有L-1个，索引从0-L-2
            linear_sigmoid_forward()的cache（Z),只有一个，索引为L-1
    '''
    caches = []  # 缓存是一个列表，也就是可变、可添加的
    A = X
    L = len(parameters) // 2  # 网络的层数，//是整除
    # 实现[LINEAR-> RELU] *（L-1），添加cache 到caches中
    for l in range(1, L):  # 1 -> L-1
        A_prev = A  # 与函数保持一致
        A, cache = linear_activation_forward(A_prev, W=parameters['W' + str(l)], b=parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)  # list 添加元素要用append，缓存的元素是Z

    # 实现LINEAR-> SIGMOID，添加cache到caches列表中
    AL, cache = linear_activation_forward(A, W=parameters['W' + str(L)], b=parameters['b' + str(L)],
                                          activation="sigmoid")
    caches.append(cache)  # list 添加元素要用append
    assert (AL.shape == (1, X.shape[1]))  # 维度为（1，样本数）
    return AL, caches


def compute_cost(AL, Y):
    '''
    计算成本函数
    :param AL: 与标签预测相对应的概率向量，维度为（1，样本数)
    :param Y:标签向量（例如：如果是猫则为1，不是猫则为0），维度为（1，样本数)
    :return:
        cost:交叉熵成本
    '''
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    cost = np.squeeze(cost)  # 让成本函数cost维度是所期望的，比如将[[17]]变成17
    assert (cost.shape == ())  # 一维

    return cost


def linear_backward(dZ, cache):
    '''
    为单层实现反向传播的线性部分(第l层)
    :param dZ: 相对于（当前l层的）线性输出的成本梯度
    :param cache:来自当前层前向传播的值的元组（A_prev,W,b)
    :return:
        dA_prev:相对于激活(前一层l-1)的成本梯度，与A_prev维度相同
        dW:相对于W（当前层l)的成本函数梯度，与w维度相同
        db:相对于b(当前层l）的成本函数梯度，与b维度相同
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]  # 样本数

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # 行向量求和，最后变成一个列向量
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def linear_activation_backward(dA, cache, activation):
    '''
    实现linear -> Activation 层的后向传播
    :param dA: 当前层激活后的梯度值
    :param cache: 我们存储用于有效计算反向传播的值的元组,值为（linear_cache(# linear_cache = (A, W, b)),activation_cache(# Z))
    :param activation:要在此层中使用的激活函数的名称，字符串类型，如["relu"|"sigmoid"]
    :return:
        dA_prev:相对于激活（前一层L-1）的成本梯度值，与A_prev的维度相同
        dW:相对于W(当前层l)的成本梯度值，与W维度相同
        db:相对于b(当前层l)的成本梯度值，与b维度相同
    '''
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)  # activation_cache = Z
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    '''
    构建多层模型的后向传播函数，对[LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID组执行反向传播
    :param AL:概率向量，正向传播的输出（L_model_forward())
    :param Y:标签向量，true "label" vector (containing 0 if non-cat, 1 if cat)
    :param caches:包含以下内容的cache列表
        linear_activation_forward("relu")的cache,不包含输出层
        linear_activation_forward("sigmoid")的cache #linear_cache, activation_cache)  # (A,W,b,Z)，其实是个一列表
    :return:
        grads:包含梯度值的字典
            grads["dA" + str(l)] = ...
            grads["dW" + str(l)] = ...
            grads["db" + str(l)] = ...
    '''
    grads = {}
    L = len(caches)  # 网络的层数，隐藏层+输出层
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # 因为有可能Y是个行向量，使之与AL保持一致
    # 初始化后向传播
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # 成本函数的导数
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]  # caches的最后一个cache
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid")
    for l in reversed(range(L - 1)):  # [L-2,...,0]
        current_cache = caches[l]  # l=L-2
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    "relu")  # dAL
        grads["dA" + str(l + 1)] = dA_prev_temp  # l=L-2 , l+1=L-1
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    '''
    使用梯度下降更新参数
    :param parameters:包含参数“W1”,“b1”,“W2”……“WL”,"bL"的字典
    :param grads:包含梯度值的字典，包含参数“dA1”,“dW1”,“db1”,“dW2”……“dWL”,"dbL",“dWL”
    :param learning_rate:学习参数
    :return:
        :parameters:包含更新参数的字典
            parameters["W" + str(l)] = ...
            parameters["b" + str(l)] = ...
    '''
    L = len(parameters) // 2  # 整除
    for l in range(L):  # 0-L-1
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    '''
    实现一个L层神经网络，[Linear -> Relu] *(L -1) -> Linera -> sigmoid.
    :param X:输入的数据，维度为（n_x,样本数）
    :param Y:标签向量，维度为（1，数量）
    :param layers_dims:层数的向量，维度为（n_x,n_h,……,n_y）
    :param learning_rate:学习率
    :param num_iterations:迭代的次数
    :param print_cost:是否打印
    :param isPlot:是否绘制出误差值的图谱
    :return:
        parameters:模型学习的参数，它们可以用来预测。
    '''
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)  # 前向传播

        cost = compute_cost(AL, Y)  # 计算成本

        grads = L_model_backward(AL, Y, caches)  # 后向传播，cache= linear_cache = (A, W, b) + activation_cache(Z)
        parameters = update_parameters(parameters, grads, learning_rate)  # 更新参数

        # 打印成本值
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))
    # 绘制成本值图形
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


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


def predict(X, Y, parameters):
    '''
    该函数用于预测L层神经网络的结果
    :param X: 测试集
    :param Y: 标签
    :param parameters: 训练模型的参数
    :return:
        p:给定数据集X的预测
    '''
    m = X.shape[1]  # 测试集的样本数
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))  # 同标签维度相同的预测

    # 根据参数进行前向传播
    AL, caches = L_model_forward(X, parameters)
    for i in range(0, AL.shape[1]):  # i: 0 -> 样本数-1
        if AL[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为" + str(float(np.sum(p == Y) / m)))


# def print_mislabeled_images(classes, X, y, p):
#     """
#     绘制预测和实际不同的图像。
#         X - 数据集
#         y - 实际的标签
#         p - 预测
#     """
#     a = p + y
#     mislabeled_indices = np.asarray(np.where(a == 1))
#     plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
#     num_images = len(mislabeled_indices[0])
#     for i in range(num_images):
#         index = mislabeled_indices[1][i]
#
#         plt.subplot(2, num_images, i + 1)
#         plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
#         plt.axis('off')
#         plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
#
#
# print_mislabeled_images(classes, test_x, test_y, pred_test)


# 加载数据
train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
# 改变训练样本和测试样本维度
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # -1表示维度可以通过数据进行判断，注意有转置
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# 标准化数据，使值介于0 — 1之间
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

# 数据加载完成，开始进行L层网络的训练
layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True, isPlot=True)

pred_train = predict(train_x, train_y, parameters)  # 训练集
pred_test = predict(test_x, test_y, parameters)  # 测试集
