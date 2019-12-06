import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_extra_datasets, load_planar_dataset

np.random.seed(1)  # 设置一个随机的种子，保证接下来的步骤中我们的结果一致

X, Y = load_planar_dataset()

# #绘制散点图
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# plt.show()

shape_x = X.shape
shape_y = Y.shape
m = Y.shape[1]
#
print("x 的维度为" + str(shape_x))
print("y 的维度为" + str(shape_y))
print("训练集里面的数据有" + str(m))


# 训练逻辑回归的分类器
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)

# 绘制决策边界
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)#匿名函数lambda x: clf.predict(x)
# plt.title("Logistic Regression")#图标题
# LR_predictions = clf.predict(X.T)#预测结果
# print('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) + '%' + "(percentage of correctly labelled datapoints)")

def layer_sizes(X, Y):
    """
    :param X: 输入数据集，维度为（输入的数量，训练集\测试集的数量）
    :param Y:标签，维度为（输出的数量，训练集|测试集数量）
    :return:
    n_x:输入层的数量
    n_h:隐藏层的数量
    n_y:输出层的数量
    """
    n_x = X.shape[0]
    n_h = 4  # 设定为4
    n_y = Y.shape[0]

    return n_x, n_h, n_y


# 测试一下layer_sizes
# X_assess, Y_assess = layer_sizes_test_case()
# n_x, n_h, n_y = layer_sizes(X_assess, Y_assess)
# print("输入层节点数为：n_x = " + str(n_x))
# print("输出层节点数为：n_y = " + str(n_y))
# print("隐藏层节点数为：n_h = " + str(n_h))

def initialize_parameters(n_x, n_h, n_y):
    """
    初始化模型的参数
    :param n_x: 输入层的数量
    :param n_h: 隐藏层的数量
    :param n_y: 输出层的数量
    :return:
     parameters:包含参数的字典
     W1：权重矩阵，维度为（n_h, n_x)
     b1：偏向量，维度为（n_h, 1)
     W2：权重矩阵，维度为（n_y, n_h)
     b2：偏向量，维度为（n_y，1）
    """
    np.random.seed(2)  # 指定种子，以便输出结果和例子保持一致
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 使用断言确保数据正确
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# #测试initialize_parameters
# n_x, n_h, n_y = initialize_parameters_test_case()
#
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    """
    前向传播函数
    :param X: 维度为（n_x,m）的输入数据
    :param parameters: 初始化函数（initialize_parameters)的输出
    :return:
        A2：使用sigmoid()函数计算的第二次激活函数后的数值
        cache:包含"Z1", "A1", "Z2" and "A2"的字典类型变量
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # 前向传播计算A2
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    # assert (A2.shape == (1, X.shape[1]))

    return A2, cache


# # 测试前向传播
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), cache['A2'])


def compute_cost(A2, Y, parameters):
    """
    计算成本函数J
    :param A2: 使用sigmoid()函数计算的第二次激活函数后的数值
    :param Y: “true”标签向量，维度为（1，训练样本数）
    :param parameters: 包含W1,b1,W2,b2的字典类型的变量
    :return:
        成本：交叉熵又方程13给出
    """
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']

    # 计算成本
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = - (1 / m) * np.sum(logprobs)
    cost = float(np.squeeze(cost))  # 确保成本维度是我们想要的，比如将[[14]]变成[17]

    assert (isinstance(cost, float))

    return cost


# #测试成本函数
# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def backward_propagation(parameters, cache, X, Y):
    """
    :param parameters: 包含W1,b1,W2,b2的字典类型的变量
    :param cache:包含"Z1", "A1", "Z2" and "A2"的字典类型变量
    :param X: 输入数据集，维度为（输入的数量，训练集\测试集的数量）
    :param Y:标签，维度为（输出的数量，训练集|测试集数量）
    :return:
        grads:包含W和b的导数的字典类型的变量
    """
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "dW2": dW2,
        "db1": db1,
        "db2": db2}
    return grads


# #测试反向传播
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    使用上面给出的梯度下降算法更新法则更新参数
    :param parameters:包含参数的字典类型变量
    :param grads:包含导数值的字典类型的变量
    :param learning_rate:学习速率
    :return:
        parameters:包含更新的字典类型的变量
    """
    # Retrieve（检索） each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter更新每个参数的规则
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# # 测试update_parameters
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
    """
    双层神经网络模型
    :param X:数据集，维度（2，示例数）dataset of shape (2, number of examples)
    :param Y:标签，维度（2，示例数）dataset of shape (2, number of examples)
    :param n_h:隐藏层数，size of the hidden layer
    :param num_iterations:梯度下降循环中的迭代次数，Number of iterations in gradient descent loop
    :param print_cost:如果为true,则每1000次迭代打印一次成本数值if True, print the cost every 1000 iterations
    :return:
        parameters： 模型学习的参数，可以用来进行预测parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)  # 随机种子
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs:n_x, n_h, n_y. Outputs = W1, b1, W2, b2, parameters
    # 初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        # Print the cost every 1000 iterations

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


# # 测试模型
# X_assess, Y_assess = nn_model_test_case()
#
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    使用学习的参数，为X中的每一个示例都预测一个类
    :param parameters:包含参数的字典类型变量python dictionary containing your parameters
    :param X:输入数据input data of size (n_x, m)
    :return:
        redictions：我们模型预测的向量（红色0，蓝色1）vector of predictions of our model (red: 0 / blue: 1)
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions


# #测试predict
# parameters, X_assess = predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))


# 正式运行
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
# 绘制边界 Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

plt.title("Decision Boundary for hidden layer size " + str(4))
# Print accuracy
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
# This may take about 2 minutes to run

#尝试不同的隐藏层
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()
