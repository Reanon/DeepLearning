import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *  # 测试函数
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward  # 激活函数
import lr_utils

np.random.seed(1)  # 指定随机种子


def initialize_parameters(n_x, n_h, n_y):
    '''
    此函数是为了初始化两层网络参数而使用的函数
    :param n_x:输入层节点数量
    :param n_h:隐藏层节点数量
    :param n_y:输出层节点数量
    :return:
        parameters:包含以下参数的字典
            W1：权重矩阵，维度为（n_h,n_x)
            W2：权重矩阵，维度为（n_y,n_h)
            b1:偏向量，维度为（n_h,1)
            b2:偏向量，维度为（n_y,1)
    '''
    W1 = np.random.randn(n_h, n_x) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01
    b1 = np.zeros((n_h, 1))  # 注意np.zeros(shape)shape需要用括号包围起来
    b2 = np.zeros((n_y, 1))

    # 使用断言确保我的数据格式是正确的
    assert (W1.shape == (n_h, n_x))
    assert (W2.shape == (n_y, n_h))
    assert (b1.shape == (n_h, 1))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  'W2': W2,
                  'b1': b1,
                  'b2': b2}
    return parameters


# #测试一下initialize_parameters
# print("============测试initialize_parameters==============")
# parameters=initialize_parameters(2,2,1)
# print("W1 = "+str(parameters['W1']))
# print("W2 = "+str(parameters['W2']))
# print("b2 = "+str(parameters['b1']))
# print("b2 = "+str(parameters['b2']))

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
    L = len(layer_dims)
    for i in range(1, L):  # range下标从0开始,如果L=3，则list(range(1,3))=[1,2]
        # parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) / np.sqrt(
        #     layer_dims[i - 1])  # 用除代替0.01
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros(shape=(layer_dims[i], 1))  # 列表使用[]
        # 确保数据正确
        assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))
    return parameters  # 包含的参数数是隐藏层数的两倍


# # 测试一下initialize_parameters
# print("============测试initialize_parameters_deep==============")
# parameters = initialize_parameters_deep([5, 4, 3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

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


# #测试linear_forward
# print("=========测试linear_forward===============")
# A, W, b = linear_forward_test_case()
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))

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


# # 测试linear_activation_forward
# # print("=========测试linear_activation_forward===============")
# A_prev, W, b = linear_activation_forward_test_case()
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
# print("With sigmoid: A = " + str(A))
# print("linear_activation_cache"+str(linear_activation_cache))
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
# print("With ReLU: A = " + str(A))


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
    for l in range(1, L):
        A_prev = A  # 与函数保持一致
        A, cache = linear_activation_forward(A_prev, W=parameters['W' + str(l)], b=parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)  # list 添加元素要用append
    # 实现LINEAR-> SIGMOID，添加cache到caches列表中
    AL, cache = linear_activation_forward(A, W=parameters['W' + str(L)], b=parameters['b' + str(L)],
                                          activation="sigmoid")
    caches.append(cache)  # list 添加元素要用append
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


# # 测试一下L_model_forward(X, parameters)
# print("===================测试一下L_model_forward(X, parameters)===================================")
# X, parameters = L_model_forward_test_case()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))
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


# # 测试compute_cost
# print("============测试compute_cost================")
# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))

def linear_backward(dZ, cache):
    '''
    为单层实现反向传播的线性部分(第l层)
    :param dZ: 相对于（当前l层的）线性输出的成本梯度
    :param cache:来自当前层前向传播的值的元组（A_prev,W,b)
    :return:
        dA_prev:相对于激活(前一层l-1)的成本梯度，与A_prev维度相同
        dW:相对于W（当前层l)的成本梯度，与w维度相同
        db:相对于b(当前层l）的成本梯度，与b维度相同
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]  # 样本数

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # 行向量求和，最后变成一个列向量
    dA_prev = np.dot(dW.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


# # 测试linear_backward(dZ, cache)
# print("==============测试linear_backward(dZ, cache)====================")
# dZ, linear_cache = linear_backward_test_case()
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))
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


# #测试一下linear_activation_backward(dA, cache, activation)
# print("=================linear_activation_backward(dA, cache, activation)========================")
# AL, linear_activation_cache = linear_activation_backward_test_case()
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

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
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp  # l=L-2 , l+1=L-1
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


# # 测试L_model_backward(AL, Y, caches)
# print("===============测试L_model_backward(AL, Y, caches)====================")
# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print("dW1 = " + str(grads["dW1"]))
# print("db1 = " + str(grads["db1"]))
# print("dA1 = " + str(grads["dA1"]))

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


# # 测试update_parameters(parameters, grads, learning_rate)
# print("===========测试update_parameters===========")
#
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("W3 = " + str(parameters["W3"]))
# print("b3 = " + str(parameters["b3"]))


# 构建双层神经网络
def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    '''
     Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    :param X:输入数据input data, of shape (n_x, number of examples)
    :param Y:标签向量true "label" vector(containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims:层数的向量dimensions of the layers (n_x, n_h, n_y)
    :param learning_rate:学习率learning rate of the gradient descent update rule
    :param num_iterations:迭代的次数number of iterations of the optimization loop
    :param print_cost:是否打印If set to True, this will print the cost every 100 iterations
    :return:
        parameters:包含W1, W2, b1, and b2的字典向量a dictionary containing W1, W2, b1, and b2
    '''

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost 追踪成本函数
    m = X.shape[1]  # number of examples 样本数
    (n_x, n_h, n_y) = layers_dims
    # 初始化两层网络参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 开始迭代Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu") # 实现linear->activation这一层的前向传播
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        # 计算成本
        cost = compute_cost(A2, Y)

        # 初始化后向传播，得到dA2
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid") #  实现linear -> Activation 层的后向传播
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        #  更新参数 Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # 重新获得参数 Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # 迭代完成，根据条件绘制图像
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # 返回parameters
        return parameters


# 加载数据
train_set_x_orig, train_y, test_set_x_orig, test_y, classes = lr_utils.load_dataset()
# 改变训练样本和测试样本维度
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # -1表示维度可以通过数据进行判断，注意有转置
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# 标准化数据，使值介于0 — 1之间
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

# 数据加载完成，开始进行二层网络的训练
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
parameters = two_layer_model(train_x, train_set_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=False,
                             isPlot=True)


def predict(X, y, parameters):
    """
    该函数用于预测L层神经网络的结果，当然也包含两层

    参数：
     X - 测试集
     y - 标签
     parameters - 训练模型的参数

    返回：
     p - 给定数据集X的预测
    """

    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))

    return p


predictions_train = predict(train_x, train_y, parameters)  # 训练集
predictions_test = predict(test_x, test_y, parameters)  # 测试集
