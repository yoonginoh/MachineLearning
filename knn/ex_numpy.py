import numpy as np

# 배열과 차이
arr = [1,2,3,4]
print("arr:",arr)
# arr: [1,2,3,4]
print("arr type:",type(arr))
# arr type: <class 'list'>

a = np.array(arr)
# numpy에서 다루는 자료형 ndarray
# ndarray 에서 다차원을 다룰 수 있다.

print("ndarray shape:", a.shape)
# ndarray shape: (4,)
print("dtype: ", a.dtype)
# dtype:  int32
print("dim 차원:", a.ndim)
b= np.array([[1,2,3,4],[5,6,7,8]])
print("ndarray shape:", b.shape)
print("dtype: ", b.dtype)
print("dim 차원:", b.ndim)

c = np.array([[[1,2,3], [4,5,6]],[[7,8,9],[10,11,12]]])
print("ndarray shape:", c.shape)
print("dtype: ", c.dtype)
print("dim 차원:", c.ndim)

d = np.array([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]])
print("ndarray shape:", d.shape)
print("dtype: ", d.dtype)
print("dim 차원:", d.ndim)
print(d)

data1 = np.array([[1,2],[3,4]])
data2 = np.array([[0,1],[1,0]])
print(data1)
print(data2)

# dot product
# 백터의 곱을 지원해주는 함수
data3 = np.dot(data1, data2)
print(data3)

# 전치행렬
data4 = np.transpose(data3)
print(data4)
test = np.arrange(10)
print(test)
print(test.shape, test.dtype, test.ndim)

#형태변경 reshape
print('2 x 5', test.reshape(2,5))
print('5 x 2', test.reshape(5,2))
print('1d', test.reshape(-1,1))

# 거리계산이 크기계산을 할 수 있으면 미믹을 만들 수 있다.
