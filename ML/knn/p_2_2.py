# -*- coding: utf-8 -*-
"""2-1. 데이터 분할

## 생선 분류 문제

### 도미 데이터 준비하기
"""
#  bream 도미 25.4 cm, 242 g
import matplotlib.pyplot as plt

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

"""### 빙어 데이터 준비하기"""
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# datasets 만들기 (도미 + 빙어)
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# ========== 2-1 과 동일 ===========
import numpy as np
# column_stack 각각의 1-d 배열을 입력으로 받아 각 배열은 하나의 열로 가지는 2d 반환
print(np.column_stack(([1,2,3], [4,5,6])))

# 실행 결과
# [[1 4]
#  [2 5]
#  [3 6]]

# 이렇게 결합되는 것을 이용하여 length 와 weight를 결합시킬 수 있다.
fish_data = np.column_stack((length, weight))
print(fish_data)
print(np.ones(5))
# 실행 결과  [1. 1. 1. 1. 1.]
print(np.zeros(5))
# 실행 결과 [0. 0. 0. 0. 0.]
# 이걸 이용하여 임의의 정답 데이터를 만들어줄 예정
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
# p_1_3에서 fish_target =[1]*35 + [0]*14 이렇게 쓴거와 동일한 리스트가 만들어졌다.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(fish_data,fish_target,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print(knn.score(x_test, y_test))
print(knn.predict([[25,150]]))
import matplotlib.pyplot as plt
# plt.scatter(x_train[:,0], x_train[:,1])
# plt.scatter(25,150, marker='^')
# # 25,150의 경우 머신러닝에 의해서 빙어로 표시됨. why?
#
# distances, indexes = knn.kneighbors([[25,150]])
# plt.scatter(x_train[indexes, 0], x_train[indexes, 1], marker='D')
# # 25,150의 근처에 있는 애들을 diamond 마커로 표시
# # 확인해보니 3마리가 빙어 2마리가 도미라서 빙어로 추정
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 축을 기준으로 평균 구하기
mean = np.mean(x_train, axis=0) #axis축 0은세로(행방향)

# 축을 기준으로 표준편차 구하기
std = np.std(x_train, axis=0) #편차
print(mean,std)

# 값을 표준화 하기 (정규화)
# 주로 데이터의 스케일을 조절하고, 데이터 분포를 평균 0과 표준편차 1 또는 특정 범위로 조정하는 데 사용
# numpy에서 전체 행에 대하여 정규화를 한줄로 가능
x_train_scaled = (x_train - mean) /std


new = ([25,150] - mean) /std
# 그래프의 x 축 범위를 0에서 1000까지로 제한
plt.xlim(0,1000)
plt.scatter(x_train[:,0], x_train[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 정규화를 시킨뒤 다시 예측하게 하면?
knn.fit(x_train_scaled, y_train)
print(knn.predict([new]))
# 아까는 0 (빙어) 로 분류되던게 1 (도미)로 분류되어있음을 알 수 있다.