import numpy as np

# a = np.zeros((2, 2, 2))
# a[:, :, 0] = ([[3, 6], [5, 8]])
# a[:, :, 1] = ([[2, 5], [7, 2]])
# b = np.zeros((2, 2, 2))
# b[:, :, 0] = ([[3, 2], [9, 6]])
# b[:, :, 1] = ([[7, 8], [1, 0]])
# c = a * b
# print(c)

# li = [2, 3, 4, 5]
# vec = np.array(li)
# vec1 = vec.copy()
# vec[2] = 77
# print((vec != vec1).any())

aaa = np.arange(2 * 3 * 4).reshape(2, 3, 4)
print(aaa.shape)
print(aaa)
aaa = aaa.transpose(0, 2, 1)
print(aaa.shape)
print(aaa)


# aa = np.arange(20 * 20 * 30 * 30).reshape(20, 20, 30, 30)
#
# # print(aa)
#
# bb = np.arange(20 * 30 * 1).reshape(20, 30, 1)
#
# # print(bb)
#
# cc = np.matmul(aa, bb)
#
# cc = np.squeeze(cc)
#
# print(cc.shape)
#
# dd = np.arange(20 * 1 * 20).reshape(20, 1, 20)
#
# print(dd.shape)
#
# ee = np.matmul(dd, cc)
#
# print(ee.shape)
#
# ee = ee.transpose((0, 2, 1))
#
# print(ee.shape)
#
# ee = np.squeeze(ee)
#
# print(ee.shape)


# np_arr = np.array([[1, 2, 0, 4], [0, 9, 2, 1]])
# print("转化前：")
# print(np_arr)
# print("转化后：")
# print(np.int64(np_arr > 0))
# print("转化后：")
# print(np.minimum(np_arr, 1))