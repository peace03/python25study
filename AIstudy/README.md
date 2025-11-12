# 결정트리
결정 트리(Decision Tree) : 데이터의 특성(feature)에 대한 '스무고개'같은 if-then 질문(규칙)을 반복하여 데이터를 분류하거나 값을 예측하는 지도학습 알고리즘

<원리>  
  1. 현재 데이터(부모 노드)를 균일하도록 쪼갤 수 있는 '특성', '경계값'을 찾는다. (정보이득 or 지니계수 사용)  
  2. 찾은 조건을 기준으로 서브 데이터 세트로 나눈다.  
  3. 위 과정을 반복한다.  
  4. 노드가 순수해지면 정지 or 하이퍼파라미터로 강제정지  

정보이득 : 1 - 엔트로피 지수 (서로 다른값이 섞여있으면 엔트로피 높음, 같은 값이 섞여있으면 엔트로피 낮음)  
지니계수 : 0수렴 -> 평등, 1수렴 -> 불평등

<특징>  
* 장점  
    1. 알고리즘이 쉽고 직관적  
    2. 시각화 표현 가능  
    3. 특별한 경우 제외 스케일링, 정규화 같은 전처리 작업 필요없다.  
* 단점  
    1. 과적합으로 정확도가 떨어짐 (보완: 트리의 크기를 사전에 제한하는 튜닝)

<파라미터>  
  - min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터 수 (default = 2)  
  - min_samples_leaf  : 분할 후 왼쪽, 오른쪽 브랜치 노드에서 가져야할 최소한의 샘플 데이터 수

---
# k평균(mean)
K-평균(K-Means) : 데이터를 K개의 그룹으로 나누기 위해, 각 데이터가 가장 가까운 '중심점(평균)'에 속하도록 반복계산하여 군집을 형성하는 비지도 학습 알고리즘

<원리>
  1. 사용자가 정한 K개의 중심점(Centroid)를 데이터 공간에 무작위로 배치한다.
  2. 모든 데이터는 K개의 중심정 중 자신과 가장 가까운 중심점을 찾아, 그 중심점의 군집(cluster)으로 배정된다.
  3. 각 군집의 중심점을 그 군집에 새롭게 배정된 데이터들의 실제 평균(mean) 위치로 이동시킨다.
  4. 중심점이 이동했으므로 2번, 3번 과정을 중심점이 더 이상 움직이지 않을 때까지 반복한다.

<특징>
* 장점
    1. 알고리즘이 쉽고 직관적, 효율적
    2. 다양한 프로그래밍 언어로 구현 쉬움
    3. 데이터가 원형 또는 볼록한 형태로 뚜렷하게 구분될 때 좋은 군집 결과를 보여줌
* 단점
    1. 몇개의 군집(k)으로 나눌지 미리 정해줘야햠 (보완: 엘보우 방법 사용)
    2. 초기 중심점 민감성
    3. 클러스터 형태 한계: 길쭉하거나 불규칙한 모양 또는 도넛모양 군집은 잘 찾지 못함
    4. 이상치(Outlier)에 민감
    5. 데이터 스케일링 필요

<코드>
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(data)
km.labels_ #군집 결과
km.transform(data) #샘플에서 클러스터 중심까지의 거리로 변환
km.n_iter_ #몇번 클러스터 중심을 옮기면서 최적의 클러스터를 찾았는지?

#그림 출력
import matplotlib.pyplot as plt
fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
axs[0].imshow(data[:], cmap='gray_r')
```
---
# 주성분분석
주성분분석 Principal Component Analysis (PCA) : 주성분 분석은 데이터의 **정보 손실을 최소화**하면서 가장 중요한 **핵심 특성(주성분)만 남겨 차원을 줄이는** 비지도 학습 알고리즘입니다.

<원리>
1. 데이터 전처리: 데이터의 중심을 원점(0,0)으로 옮기고, 표준화(스케일링)를 통해 범위를 맞춥니다.
2. 주성분 1 찾기: 데이터가 가장 넓게 퍼져 있는(분산이 가장 큰) 방향의 첫 번째 축을 찾습니다.
3. 주성분 2 찾기: 첫 번째 축과 직각(90도)을 이루면서, 남은 분산을 가장 잘 설명하는 두 번째 축을 찾습니다.
4. 중요도 정렬: 찾아낸 축들을 정보량(분산의 크기)이 높은 순서대로 나열하여 중요도를 매깁니다.
5. 차원 축소(투영): 중요도가 낮은 축은 버리고, 남은 핵심 축들 위로 데이터를 옮겨 심어 차원을 줄입니다.

<특징>
- 장점
  1. 상관관계 제거
  2. 노이즈 제거
  3. 범용성과 속도
- 단점
  1. 비선형 구조 반영 불가
  2. 해석의 어려움
  3. 분류 정보 손실 가능성
 
  <코드>
  ```python
# ==========================================
# 0. 라이브러리 임포트 및 데이터 준비
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate

# 데이터 다운로드 (코랩 환경 가정)
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

# 데이터 로드 및 전처리
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)  # 3차원(300,100,100)을 2차원(300,10000)으로 펼침 (Flatten)

# 시각화를 위한 함수 정의 (이미지 그리기)
def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

# ==========================================
# 1. PCA 기본: 주성분 개수 지정 (n_components=50)
# ==========================================
print("--- 1. PCA (주성분 50개) ---")

# PCA 객체 생성: 주성분(특성)을 50개로 줄이겠다고 설정
pca = PCA(n_components=50)

# PCA 학습: 데이터의 분산이 가장 큰 방향(주성분)을 찾음 (비지도 학습이라 타깃 없음)
pca.fit(fruits_2d)

# 차원 축소: 10,000개의 특성을 50개로 압축 (transform)
fruits_pca = pca.transform(fruits_2d)
print(f"원본 데이터 크기: {fruits_2d.shape}")       # (300, 10000)
print(f"축소된 데이터 크기: {fruits_pca.shape}")    # (300, 50) -> 저장 공간 절약

# 데이터 복원: 50개의 특성을 다시 10,000개로 복원 (손실이 있지만 원본과 유사함)
fruits_inverse = pca.inverse_transform(fruits_pca)
print(f"복원된 데이터 크기: {fruits_inverse.shape}") # (300, 10000)

# 설명된 분산 비율: 50개의 주성분이 원본 데이터의 분산을 얼마나 표현하는지 확인
# 합계가 높을수록 원본 데이터의 정보 손실이 적다는 뜻
print(f"설명된 분산 비율 합계: {np.sum(pca.explained_variance_ratio_)}") # 약 92%

# ==========================================
# 2. 지도 학습 응용: 로지스틱 회귀 성능 비교
# ==========================================
print("\n--- 2. 로지스틱 회귀 성능 비교 (속도 차이 확인) ---")

# 타깃 데이터 생성 (사과:0, 파인애플:1, 바나나:2)
target = np.array([0]*100 + [1]*100 + [2]*100)

# 모델 생성
lr = LogisticRegression()

# A. 원본 데이터(10,000개 특성)로 교차 검증
scores_origin = cross_validate(lr, fruits_2d, target)
print(f"원본 데이터 훈련 시간: {np.mean(scores_origin['fit_time']):.4f}초")
print(f"원본 데이터 정확도: {np.mean(scores_origin['test_score']):.4f}")

# B. PCA 데이터(50개 특성)로 교차 검증
scores_pca = cross_validate(lr, fruits_pca, target)
print(f"PCA 데이터 훈련 시간: {np.mean(scores_pca['fit_time']):.4f}초") # 시간이 훨씬 단축됨
print(f"PCA 데이터 정확도: {np.mean(scores_pca['test_score']):.4f}")   # 성능은 비슷함

# ==========================================
# 3. PCA 응용: 분산 비율로 차원 축소 (n_components=0.5)
# ==========================================
print("\n--- 3. PCA (분산 50% 유지) ---")

# PCA 객체 생성: 원본 정보(분산)의 50%를 유지하는 최소한의 주성분 개수를 찾으라고 설정
pca_var = PCA(n_components=0.5)
pca_var.fit(fruits_2d)

# 몇 개의 주성분(특성)이 필요한지 확인
print(f"50% 분산을 위해 선택된 주성분 개수: {pca_var.n_components_}") # 2개

# 차원 축소: 10,000개 -> 2개로 압축
fruits_pca_var = pca_var.transform(fruits_2d)
print(f"비율로 축소된 데이터 크기: {fruits_pca_var.shape}") # (300, 2)

# ==========================================
# 4. 비지도 학습 응용: K-평균 군집화 & 시각화
# ==========================================
print("\n--- 4. K-Means 군집화 및 시각화 ---")

# K-Means 모델 생성 및 훈련 (2개 특성만 가진 fruits_pca_var 사용)
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca_var)

# 군집 결과 확인
print(f"군집별 샘플 개수: {np.unique(km.labels_, return_counts=True)}")

# 시각화: 특성이 2개뿐이므로 2차원 평면(x축, y축)에 산점도로 그릴 수 있음
for label in range(0, 3):
    data = fruits_pca_var[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1]) # 0번 주성분을 x축, 1번 주성분을 y축으로

plt.legend(['Apple', 'Pineapple', 'Banana'])
plt.title("Visualization of Clustering with 2 PCA Components")
plt.show()
  ```
