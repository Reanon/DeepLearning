import numpy as np
import h5py
import matplotlib.pyplot as plt

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
    b1 = np.zeros(shape=(n_h, 1))  # 注意np.zeros(shape)shape需要用括号包围起来
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


def compute_cost(AL, Y):
    '''
    计算成本函数
    :param AL: 与标签预测相对应的概率向量，维度为（1，样本数)
    :param Y:标签向量（例如：如果是猫则为1，不是猫则为0），维度为（1，样本数)
    :return:
        cost:交叉熵成本
    '''
    m = Y.shape[1]  # 样本数
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
    for l in range(L):  # 0 -> L-1，这里l从0开始，所以下面就要加1.
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# 构建双层神经网络
def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=True, isPlot=True):
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
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")  # 实现linear->activation这一层的前向传播
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        # 计算成本
        cost = compute_cost(A2, Y)

        # 初始化后向传播，得到dA2
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")  # 实现linear -> Activation 层的后向传播
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


# 加载数据
train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
# 改变训练样本和测试样本维度
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # -1表示维度可以通过数据进行判断，注意有转置
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# 标准化数据，使值介于0 — 1之间
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

# 数据加载完成，开始进行二层网络的训练
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
parameters = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=3000, print_cost=True,
                             isPlot=True)


def predict(X, y, parameters):
    """
    该函数用于预测二层神经网络的结果
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

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    A1, cache1 = linear_activation_forward(X, W1, b1, "relu")  # 实现linear->activation这一层的前向传播
    A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
    # 根据参数前向传播
    probas = A2
    for i in range(0, probas.shape[1]):  # range =(0,m_train)
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))
    return p


# 进行预测
pred_train = predict(train_x, train_y, parameters)  # 训练集
pred_test = predict(test_x, test_y, parameters)  # 测试集
