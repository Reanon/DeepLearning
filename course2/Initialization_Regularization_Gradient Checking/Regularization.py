import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io as sio


def load_2D_dataset(is_plot=True):
    data = sio.loadmat('datasets/data2.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    if is_plot:
        plt.scatter(train_X[0, :], train_X[1, :], c=np.squeeze(train_Y), s=40,
                    cmap=plt.cm.Spectral)  # 将c=train_Y改为c=np.squeeze(train_Y)

    return train_X, train_Y, test_X, test_Y


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)

    return s


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])

    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1].
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()

    Returns:
    loss -- the loss function (vanilla logistic loss)
    """

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    return a3, cache


def forward_propagation_with_dropout(X, parameters, keep_prob):
    '''
    实现具有随机失活节点的前向传播
    LINEAR->RELU + DROPOUT->LINEAR->RELU + DROPOUT->LINEAR->SIGMOID.
    :param X:输入数据集，维度为（2，示例数）
    :param parameters:含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
            W1  - 权重矩阵，维度为（20,2）
            b1  - 偏向量，维度为（20,1）
            W2  - 权重矩阵，维度为（3,20）
            b2  - 偏向量，维度为（3,1）
            W3  - 权重矩阵，维度为（1,3）
            b3  - 偏向量，维度为（1,1）
    :param keep_prob:保存节点的概率，实数
    :return:
        A3  - 最后的激活值，维度为（1,1），正向传播的输出
        cache - 存储了一些用于计算反向传播的数值的元组
    '''
    np.random.seed(1)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # LINEAR->RELU
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    # DROPOUT 随机失活
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob

    # LINEAR->RELU
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    # DROPOUT 随机失活
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    # LINEAR->Sigmoid
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def compute_cost(a3, Y):
    """
    Implement the cost function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]

    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1. / m * np.nansum(logprobs)

    return cost


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    '''
    实现L2正则化的成本计算
    :param A3:前向传播输出结果，维度（输出节点，样本数）
    :param Y:标签，维度（输出节点，样本数）
    :param parameters:学习后的参数字典
    :param lambd:正则参化数
    :return:
        cost:正则化后的成本值
    '''
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    cross_entropy_cost = compute_cost(A3, Y)  # 交叉熵成本，也就是无正则化的成本值
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W3)) + np.sum(np.square(W2))) / (
            2 * m)  # L2正则化的成本
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1. / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))  # relu_back
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients


def backward_propagation_with_regularization(X, Y, cache, lambd):
    '''
    实现L2正则化后的模型的反向传播
    :param X:输入数据集
    :param Y:标签，维度（输出节点，样本数）
    :param cache:来自forward_propagation()的cache输出
    :param lambd:正则化的参数
    :return:
        gradients:一个包含每个参数、激活值和预激活值变量的梯度的字典
    '''
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - Y  # 不是很清楚这里为什么是这么求dZ3，而不是dZ3=dA3*relu_backward(Z3)
    dW3 = (1. / m) * np.dot(dZ3, A2.T) + (lambd * W3) / m
    db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1. / m) * np.dot(dZ2, A1.T) + (lambd * W2) / m
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1. / m) * np.dot(dZ1, X.T) + (lambd * W1) / m
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    实现我们随机失活模型的后向传播。
    参数：
        X ：输入数据集，维度为（2，示例数）
        Y  ：标签，维度为（输出节点数量，示例数量）
        cache ：来自forward_propagation_with_dropout（）的cache输出
        keep_prob ：随机删除的概率，实数

    返回：
        gradients ：一个关于每个参数、激活值和预激活变量的梯度值的字典
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))  # relu_back
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA1 = dA1 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of n_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters['W' + str(i)] = ...
                  parameters['b' + str(i)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)

    # Forward propagation
    a3, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

    return p


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
    '''
    实现三层神经网络: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    :param X:输入的数据，维度为（2，样本数）
    :param Y:标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
    :param learning_rate:学习速率
    :param num_iterations:迭代的次数
    :param print_cost:是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
    :param is_plot:是否绘制梯度下降的曲线图
    :param lambd:正则化的超参数，实数
    :param keep_prob: 随机保留节点的概率
    :return:
        parameters:学习后的参数
    '''
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]
    # 初始化参数
    parameters = initialize_parameters(layers_dims)
    # 开始学习
    for i in range(0, num_iterations):
        # 前向传播
        # 是否随机删除节点
        if keep_prob == 1:
            # 不随机删除节点
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            # 随机删除节点
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print("keep_prob参数错误")
            exit
        # 计算成本
        # 是否使用二范数
        if lambd == 0:
            # 不使用L2正则化
            cost = compute_cost(a3, Y)
        else:
            # 使用L2正则化
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
        # 反向传播
        # 可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用。
        assert (lambd == 0 or keep_prob == 1)

        # 两个参数的使用情况
        if (lambd == 0 and keep_prob == 1):
            # 不使用L2正则化和不使用随机删除节点
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            # 使用L2正则化，不使用随机删除节点
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            # 使用随机删除节点，不使用L2正则化
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 记录并打印成本
        if i % 1000 == 0:
            # 记录成本
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                # 打印成本
                print("第" + str(i) + "次迭代，成本值为" + str(cost))
    # 是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# 读取数据
train_X, train_Y, test_X, test_Y = load_2D_dataset(is_plot=False)
plt.show()

# # 普通训练
# parameters = model(train_X, train_Y, is_plot=True)
# print("On the training set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# # 正则化训练
# parameters = model(train_X, train_Y, lambd=0.7, is_plot=True)
# print("使用正则化，训练集:")
# predictions_train = predict(train_X, train_Y, parameters)
# print("使用正则化，测试集:")
# predictions_test = predict(test_X, test_Y, parameters)

# 随机失活模型，程序都可以24％的概率关闭第1层和第2层的每个神经元。
parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3, is_plot=True)
# 预测，查看模型对训练集和测试集的拟合程度。
print("使用随机删除节点，训练集:")
predictions_train = predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
predictions_test = predict(test_X, test_Y, parameters)

# 查看分类情况，以蓝红为界，中间的线是分界线
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
