import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset, model

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#x显示图片
# index = 20
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +"' picture.")#np.squeeze()是压缩维度的

#train_set_x_orig是一个维度为（m_train,num_px,num_px,3)的数组
#train_set_y是一个维度为(1,m_train)的数组
m_train = train_set_y.shape[1] #训练集里图片的数量
m_test = test_set_y.shape[1]   #测试集里图片的数量
num_px = train_set_x_orig.shape[1]  #训练、测试集里图片的高度和宽带

print("训练集的数量： m_train= " + str(m_train))
print("测试集的数量： m_test= " + str(m_test))
print("每张图片的高： num_px= " + str(num_px))

print("训练集图片维度：" + str(train_set_x_orig.shape))
print("训练集标签维度：" + str(train_set_y.shape))
print("测试集图片维度：" + str(test_set_x_orig.shape))
print("测试集标签维度：" + str(test_set_y.shape))

#将训练和测试集的维度降低并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("训练集降维后的维度：" + str(train_set_x_flatten.shape))
print("训练集标签维度：" + str(train_set_y.shape))
print("测试集降维后的维度：" + str(test_set_x_flatten.shape))
print("测试集标签维度：" + str(test_set_y.shape))

#标准化数据使其位于[0,1]之间
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# #测试sigmoid
# print("sigmoid(0) = " + str(sigmoid'(0)))
#
# #测试一下propagate
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2],[3,4]]), np.array([[1, 0]])
# params, grads, cost = optimize(w, b, X, Y)
#
# print("w = " + str(params["w"]))
# print("b = " + str(params["b"]))
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))
# # print("cost = " + str(cost))
# print("predictions = " + str(predict(w, b, X)))

# #测试model
# d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iteration=2000, learning_rate=0.005, print_cost=True)
# #绘制图
# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('costs')
# plt.xlabel('iterations(per humdreds)')
# plt.title("Learning rate="+str(d['learning_rate']))
# plt.show()

#多次测试学习率
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is "+str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iteration=1500, learning_rate=i, print_cost=False)
    print('\n'+"------------------------------"+'\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

#将几张图做在同一张图里，并添加图例
legend = plt.legend(loc='best', shadow=True)#图例
# frame = legend.get_frame() #这两行代码不是很必要
# frame.set_facecolor('0.90')#这两行代码不是很必要
plt.show() #在pycharm中一定要写，不然没图
