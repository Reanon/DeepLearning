import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets


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


def compute_loss(a3, Y):
    """
    Implement the loss function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    loss - value of the loss function
    """

    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1. / m * np.nansum(logprobs)

    return loss


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
    dz2 = np.multiply(da2, np.int64(a2 > 0))
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


def load_dataset(is_plot=True):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


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
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)  # 需要将c=y改为c=np.squeeze(y)
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


def initialize_parameters_zeros(layers_dim):
    '''
    将模型参数全部设置为0
    :param layers_dim: 列表，模型的层数对应每一层节点的数
    :return:
         W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
         b1 -- bias vector of shape (layers_dims[1], 1)
         ...
         WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
         bL -- bias vector of shape (layers_dims[L], 1)
    '''
    parameters = {}
    L = len(layers_dim)  # 网络层数，包含了输入层
    for l in range(1, L):  # 1 -> L-1，如果L=4，则只有W1,b1……W3,b3.
        parameters['W' + str(l)] = np.zeros((layers_dim[l], layers_dim[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dim[l], 1))

        # 使用断言
        assert (parameters['W' + str(l)].shape == (layers_dim[l], layers_dim[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers_dim[l], 1))
    return parameters


def initialize_parameters_random(layers_dim):
    '''
    将模型参数随机初始化
    :param layers_dim: 列表，模型的层数对应每一层节点的数
    :return:
         W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
         b1 -- bias vector of shape (layers_dims[1], 1)
         ...
         WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
         bL -- bias vector of shape (layers_dims[L], 1)
    '''
    np.random.seed(3)  # 一定要指定随机种子
    parameters = {}
    L = len(layers_dim)  # 网络层数，包含了输入层
    for l in range(1, L):  # 1 -> L-1，如果L=4，则只有W1,b1……W3,b3.
        parameters['W' + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * 10  # 用十倍缩放
        parameters['b' + str(l)] = np.zeros((layers_dim[l], 1))  # zeros需要两层括号，rand()只需要一层

        # 使用断言
        assert (parameters['W' + str(l)].shape == (layers_dim[l], layers_dim[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers_dim[l], 1))
    return parameters


def initialize_parameters_he(layers_dims):
    '''
    将模型参数抑梯度异常初始化
    :param layers_dim: 列表，模型的层数对应每一层节点的数
    :return:
         W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
         b1 -- bias vector of shape (layers_dims[1], 1)
         ...
         WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
         bL -- bias vector of shape (layers_dims[L], 1)
    '''
    np.random.seed(3)  # 一定要指定随机种子
    parameters = {}
    L = len(layers_dims)  # 网络层数，包含了输入层
    for l in range(1, L):  # 1 -> L-1，如果L=4，则只有W1,b1……W3,b3.
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2 / layers_dims[l - 1])  # rand()只需要一层
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        # 使用断言
        assert (parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers_dims[l], 1))
    return parameters


def model(X, Y, learing_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", is_plot=True):
    '''
    实现三层的神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    :param X:输入数据，维度为（2，样本数）
    :param Y:标签向量，[0是红点，1是蓝点]，维度为（1，样本数）
    :param learing_rate:梯度下降的学习速率
    :param num_iterations:迭代次数
    :param print_cost:是否打印成本值，每迭代1000次打印一次
    :param initialization:字符串类型，初始化的类型("zeros","random" or "he")
    :param is_plot:是否绘制梯度下降的曲线图
    :return:
        parameters:学习后的参数
    '''
    grads = {}  # 梯度值
    costs = []  # 成本
    m = X.shape[1]  # 样本数
    layers_dims = [X.shape[0], 10, 5, 1]  # 三层神经网络

    # 初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else:
        print("错误参数，程序退出")  # 保障措施
        exit

    # 开始学习
    for i in range(0, num_iterations):
        # 前向传播
        a3, cache = forward_propagation(X, parameters)
        # 计算成本
        cost = compute_loss(a3, Y)
        # 反向传播
        grads = backward_propagation(X, Y, cache)
        # 更新参数
        parameters = update_parameters(parameters, grads, learing_rate)

        # 记录成本
        if i % 1000 == 0:
            costs.append(cost)
            # 打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    # 学习完毕，绘制成本函数
    if is_plot:
        plt.plot(costs)
        plt.xlabel("iterations")  # 绘制图像只能用英文
        plt.ylabel("costs")
        plt.title("learning rate=" + str(learing_rate))
        plt.show()

    # 返回学习好的参数
    return parameters

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 读取并绘制数据，圆圈中的蓝色/红色圆点
train_X, train_Y, test_X, test_Y = load_dataset(is_plot=False)

# # 用初始化为0来训练参数
# parameters = model(train_X, train_Y, initialization="zeros", is_plot=True)

# # 用随机初始化为来训练参数
# parameters = model(train_X, train_Y, initialization="random", is_plot=True)

# 用抑梯度异常初始化来训练参数
parameters = model(train_X, train_Y, initialization="he", is_plot=True)

# 预测结果
print("训练集")
predictions_train = predict(train_X, train_Y, parameters)  # 预测测试集
print("测试集")
predictions_test = predict(test_X, test_Y, parameters)  # 预测训练集

print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

# 绘制决策边界
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])

plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

plt.show() # 显示图像