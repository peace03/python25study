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
---
# 인공신경망
인공신경망 Artificial Neural Network (ANN) : 사람의 뇌 신경 세포(뉴런)가 전기 신호를 주고받는 방식을 모방하여, 데이터의 복잡한 패턴을 스스로 학습하는 알고리즘

<원리>
1. 입력(Input): 들어온 데이터(x)에 중요도인 가중치(w)를 곱하고 편향(b)을 더해 신호의 총합(z)을 구합니다.
2. 활성화(Activation): 합쳐진 신호를 활성화 함수(시그모이드, 렐루 등)에 통과시켜 다음 단계로 신호를 보낼지 결정합니다.
3. 학습(Backpropagation): 예측값과 정답의 오차(손실)를 계산하고, 이를 줄이는 방향으로 가중치를 거꾸로 수정(역전파)하며 똑똑해집니다.

<특징>
- 장점
  1. 이미지, 음성, 텍스트 등 규칙을 정의하기 힘든 복잡한 비정형 데이터 처리에 압도적인 성능을 보임
- 단점
  1. 내부 계산이 너무 복잡하여, 모델이 왜 이런 결론을 내렸는지 설명하기 어려운 '블랙박스'문제가 있다.
  2. 제대로 된 성능을 내기 위해서는 아주 많은 양의 데이터와 긴 학습 시간(GPU)이 필요합니다.
 
<코드>
```python
# ==========================================
# 0. 환경 설정 및 라이브러리 임포트
# ==========================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 실행 결과의 재현성을 위해 랜덤 시드 고정 (필수)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

print("TensorFlow version:", tf.__version__)

# ==========================================
# 1. 데이터 준비 (로드 및 전처리)
# ==========================================
print("\n--- 1. 데이터 준비 ---")

# 패션 MNIST 데이터셋 로드 (훈련용 / 테스트용 자동 분할)
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 데이터 크기 확인
print(f"원본 훈련 데이터: {train_input.shape}, 타깃: {train_target.shape}")
print(f"원본 테스트 데이터: {test_input.shape}, 타깃: {test_target.shape}")

# [전처리 1] 정규화 (Normalization): 0~255 픽셀 값을 0~1 사이로 변환
train_scaled = train_input / 255.0
test_scaled = test_input / 255.0

# [전처리 2] 평탄화 (Flattening): 2차원 이미지(28x28)를 1차원 배열(784)로 펼침
# (DNN 모델 입력용. CNN 모델 사용 시에는 이 과정 생략 가능)
train_scaled = train_scaled.reshape(-1, 28*28)
test_scaled = test_scaled.reshape(-1, 28*28)

print(f"전처리 후 훈련 데이터: {train_scaled.shape}") # (60000, 784)

# [검증 세트 분리] 훈련 세트에서 20%를 떼어내어 검증(Validation) 세트로 만듦
# 딥러닝은 데이터가 많아 교차 검증 대신 검증 세트를 사용함
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
print(f"최종 훈련 세트: {train_scaled.shape}, 검증 세트: {val_scaled.shape}")

# ==========================================
# 2. 모델 만들기 (인공신경망 정의)
# ==========================================
print("\n--- 2. 모델 생성 ---")

# Dense 층 (밀집층/완전연결층) 정의
# 10: 뉴런 개수 (타깃 클래스가 10개이므로)
# activation='softmax': 다중 분류를 위한 활성화 함수 (이진 분류는 'sigmoid')
# input_shape=(784,): 입력 데이터의 크기 (첫 번째 층에만 지정)
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))

# Sequential 클래스로 모델 구성 (층을 순서대로 쌓음)
model = keras.Sequential([dense])

# 모델 구조 요약 출력 (파라미터 개수 등 확인)
model.summary()

# ==========================================
# 3. 모델 설정 (Compile)
# ==========================================
print("\n--- 3. 모델 컴파일 ---")

# compile(): 훈련 방법 설정
# loss: 손실 함수
#   - 'sparse_categorical_crossentropy': 타깃이 정수(0, 1, 2...)인 다중 분류
#   - 'categorical_crossentropy': 타깃이 원-핫 인코딩된 다중 분류
#   - 'binary_crossentropy': 이진 분류
# metrics: 훈련 도중 모니터링할 지표 (정확도 등)
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. 모델 훈련 (Fit)
# ==========================================
print("\n--- 4. 모델 훈련 ---")

# fit(): 실제 학습 진행
# epochs: 전체 데이터를 몇 번 반복해서 학습할지 설정
# validation_data: 검증 세트를 전달하여 에포크마다 검증 점수도 함께 확인 가능 (선택)
history = model.fit(train_scaled, train_target, epochs=5, validation_data=(val_scaled, val_target))

# ==========================================
# 5. 모델 평가 (Evaluate)
# ==========================================
print("\n--- 5. 모델 평가 ---")

# evaluate(): 검증 세트(또는 테스트 세트)로 최종 성능 확인
# 반환값: [손실(loss), 정확도(accuracy)]
loss, accuracy = model.evaluate(val_scaled, val_target)
print(f"검증 세트 손실: {loss:.4f}")
print(f"검증 세트 정확도: {accuracy:.4f}")

# ==========================================
# [부록] 예측 및 시각화 (활용 예시)
# ==========================================
print("\n--- [부록] 예측 결과 확인 ---")

# 훈련된 모델로 예측 수행 (검증 데이터 앞 10개)
# predict(): 각 클래스별 확률을 반환함
predictions = model.predict(val_scaled[:10])
print(f"예측 확률 모양: {predictions.shape}") # (10, 10)

# 가장 높은 확률을 가진 클래스 인덱스 찾기 (argmax)
predicted_classes = np.argmax(predictions, axis=-1)
print(f"예측된 클래스: {predicted_classes}")
print(f"실제 정답:     {val_target[:10]}")

# 손실과 정확도 그래프 그리기 (history 객체 활용)
plt.figure(figsize=(12, 4))

# 1. 손실(Loss) 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

# 2. 정확도(Accuracy) 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.show()
```
---
# 심층신경망
심층신경망 Deep Neural Network (DNN) : 인공신경망의 입력층과 출력층 사이에 있는 은닉층(Hidden Layer)을 2개 이상 깊게 쌓아 올려, 데이터의 더 복잡하고 추상적인 특징을 단계별로 학습하는 알고리즘

<원리>
1. 심층구조(Deep Architecture): 층이 하나뿐인 기존 신경망과 달리, 수십~수백개의 층을 깊게 배치하여 데이터를 훨씬 더 정교하고 세밀하게 분해합니다.
2. 표현학습(Representation Learning): 앝은 층에서는 선이나 점 같은 단순한 특징을 찾고, 깊은 층으로 갈수록 이를 조합해 눈,코,입 같은 복잡한 개념을 스스로 이해합니다.
3. 최적화(Optimization): 층이 너무 깊어지면 학습이 안되는 문제(기울기 소실)을 렐루(ReLU), 드롭아웃(Dropout) 등의 기법으로 해결하여 깊은 망까지 신호가 잘 전달되게 합니다.

<특징>
- 장점
  1. 데이터의 양이 많아질수록 성능이 정체되지 않고 계속 좋아짐.
  2. 사람 수준의 인지 능력 구현 가능
- 단점
  1. 학습해야할 파라미터(가중치)가 수백만개 이상이므로, 고성능 GPU와 방대한 데이터가 필요
  2. 훈련 데이터의 노이즈까지 너무 상세하게 외워버리는 과대적합(Overfitting)이 발생하기 쉬워 정교한 규제 기술이 필수
 
<코드>
```python
# ==========================================
# 0. 환경 설정 및 라이브러리 임포트
# ==========================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 실행 결과의 재현성을 위해 랜덤 시드 고정 (필수)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

print(f"TensorFlow version: {tf.__version__}")

# ==========================================
# 1. 데이터 준비 (로드 및 전처리)
# ==========================================
print("\n--- 1. 데이터 준비 ---")

# 패션 MNIST 데이터셋 로드
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 정규화 (0~255 -> 0~1)
# DNN은 픽셀 값이 0~1 사이일 때 학습이 가장 잘 됨
train_scaled = train_input / 255.0
test_scaled = test_input / 255.0

# 데이터 크기 확인
print(f"전처리 전 훈련 데이터: {train_scaled.shape}")

# 검증 세트 분리 (20%)
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
print(f"최종 훈련 세트: {train_scaled.shape}, 검증 세트: {val_scaled.shape}")

# ==========================================
# 2. 심층 신경망 (DNN) 모델 생성 함수
# ==========================================
def create_dnn_model():
    """
    은닉층이 포함된 심층 신경망 모델을 생성하여 반환하는 함수
    구조: Flatten -> Dense(ReLU) -> Dense(Softmax)
    """
    model = keras.Sequential(name='Fashion_MNIST_DNN')
    
    # [입력층] Flatten: 2차원 이미지(28x28)를 1차원(784)으로 펼침
    # 파라미터가 없는 층이지만 입력값의 차원을 변환해줌
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    # [은닉층] Dense: 100개의 뉴런, 활성화 함수는 ReLU
    # ReLU: 시그모이드의 단점(기울기 소실)을 해결하여 심층 신경망 학습에 유리함
    # 이미지 처리에서 특히 좋은 성능을 보임
    model.add(keras.layers.Dense(100, activation='relu', name='hidden'))
    
    # [출력층] Dense: 10개의 클래스 분류, 활성화 함수는 Softmax
    # 다중 분류 문제이므로 softmax 사용 (이진 분류라면 sigmoid)
    model.add(keras.layers.Dense(10, activation='softmax', name='output'))
    
    return model

# 모델 구조 확인
model_test = create_dnn_model()
model_test.summary()

# ==========================================
# 3. 다양한 옵티마이저(Optimizer) 비교 학습
# ==========================================
print("\n--- 3. 옵티마이저별 성능 비교 ---")

# 테스트할 옵티마이저 목록 (이름 문자열 또는 객체)
# 1. sgd: 기본 확률적 경사 하강법
# 2. adagrad: 적응적 학습률 (많이 변한 파라미터는 학습률을 낮춤)
# 3. rmsprop: 적응적 학습률 (최근 기울기를 더 반영, 케라스의 RNN 기본값)
# 4. adam: 모멘텀(관성) + RMSprop의 장점을 합친 알고리즘 (가장 널리 쓰임 ⭐️)

optimizers = ['sgd', 'adagrad', 'rmsprop', 'adam']
history_dict = {}

for opt_name in optimizers:
    print(f"\n[Optimizer: {opt_name.upper()}] 훈련 시작...")
    
    # 매번 새로운 모델 생성 (가중치 초기화)
    model = create_dnn_model()
    
    # 모델 컴파일
    # optimizer: 문자열로 지정하면 기본 설정값으로 객체가 자동 생성됨
    model.compile(loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'], 
                  optimizer=opt_name)
    
    # 모델 훈련
    history = model.fit(train_scaled, train_target, epochs=5, verbose=0) # verbose=0: 로그 숨김
    
    # 검증 세트 평가
    val_loss, val_acc = model.evaluate(val_scaled, val_target, verbose=0)
    print(f" -> 검증 정확도: {val_acc:.4f}")
    
    # 결과 저장
    history_dict[opt_name] = history

# ==========================================
# 4. 학습 결과 시각화 비교
# ==========================================
print("\n--- 4. 학습 곡선 시각화 ---")

plt.figure(figsize=(10, 6))

for opt_name, history in history_dict.items():
    plt.plot(history.history['loss'], label=f'{opt_name}')

plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Optimizer Comparison (Loss)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 5. (심화) 옵티마이저 세부 설정 예시
# ==========================================
# 문자열('adam') 대신 객체를 직접 생성하면 학습률(learning_rate) 등 세부 조정 가능

# 예: SGD에 모멘텀(관성) 적용 및 네스테로프 가속 경사 사용
sgd_custom = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

# 예: Adam의 학습률 조정
adam_custom = keras.optimizers.Adam(learning_rate=0.001) # 기본값

# 적용 방법
# model.compile(optimizer=adam_custom, ...)
```
---
# 인공신경망 기타 도구 (드롭아웃, 콜백 & 조기종료)
드롭아웃 Dropout : 훈련 과정에서 신경망의 일부 뉴런을 무작위로 비활성화해 특정 뉴런에 대한 의존도를 낮추고 과대적합을 방지하는 기법
콜백 Callback : 모델 훈련 과정의 중간 시점(예: 에포크 끝)마다 자동으로 호출되어 모델 저장, 학습률 조정 등의 작업을 수행하는 도구
조기종료 Early Stopping : 검증 세트의 성능(손실 등)이 더 이상 좋아지지 않을 때 훈련을 스스로 멈춰 과대적합을 막고 시간을 절약하는 기법

<원리>
- 드롭아웃
  1. 랜덤 선택 : 훈련(fit)이 시작되면, 매 단계마다 은닉층의 뉴런 중 일부를 무작위로 선택
  2. 비활성화 : 선택된 뉴런의 출력을 0으로 만들고, 연결된 가중치 계산을 일시적으로 종료
  3. 분산 학습 : 남은 뉴런들끼리만 힘을 합쳐 예측하고 오차를 역전파하여 학습 (특정 뉴런 독점 방지)
  4. 전체 복구 : 훈련이 끝나고 평가(evaluate)나 예측(predict)에 들어갈 때는 모든 뉴런을 다시 켜서 100% 성능을 낸다.
- 콜백
  1. 객체 생성 : 저장(ModelCheckpoint)이나 학습률 조정(ReduceLROnPlateau) 등 원하는 기능의 콜백 객체 생성
  2. 주입(Injection) : model.fit()메서드 실행할 때 callback=[...] 리스트에 담아 모델에게 건넨다.
  3. 감시(Hook) : 훈련 도중 에포크 시작/종료, 배치의 시작/종료 시점마다 콜백 함수가 자동으로 호출되어 훈련 상태(손실값, 정확도)를 조회한다.
  4. 조건부 실행 : '점수가 올랐으면 저장', '점수가 안오르면 학습률 감소'와 같이 조건이 맞을 때 설정된 작업 수행
- 조기종료
  1. 모니터링 : 매 에포크 종료 시점 검증세트의 손실값 확인
  2. 비교 및 카운트: 손실이 줄어들면 계속 진행, 손실이 같거나 늘어나면 카운트(patience) +1
  3. 중단 판정 : 설정한 카운트(ex: patience=2)를 초과하면 훈련 강제종료
  4. 복구 : 훈련을 종료 후, 지금까지 기록된 것 중 가장 점수가 좋았던 에포크의 가중치를 모델로 되돌려 놓는다.(restore_best_weights=True 설정시)
 
<특징>
- 드롭아웃
  - 장점
    1. 강력한 일반화 성능
    2. 강력한 과대적합 방지
    3. 구현의 단순함
  - 단점
    1. 훈련속도 저하 : 매번 뉴런을 끄고 학습하기 때문에, 손실이 줄어드는 속도가 느려져 더 많은 에포크가 필요 (보완: 배치 정규화 사용)
    2. 불확실성 : 훈련때마다 랜덤하게 작동하므로, 동일한 데이터라도 학습 결과가 미세하게 흔들릴 수 있음 (보완: 에포크 늘리기)
- 콜백
  - 장점
    1. 완전 자동화 : 지켜보지 않아도 저장, 중단, 파라미터 조정 등을 알라서 수행
    2. 유연한 확장성 : 텐서보드 연결, 슬랙/이메일 알림 등 훈련과정에서 원하는 기능을 자유롭게 붙일 수 있다.
  - 단점
    1. 설정의 복잡함 (보완: ModelCheckpoint, EarlyStopping 같은 핵심 콜백 위주로 가볍게 구성)
    2. 추가 연산 비용 (keras 내장 기능 활용)
- 조기종료
  - 장점
    1. 자원 절약
    2. 과대적합 자동 차단
    3. 최적 모델 보장 : restore_best_weights=True 옵션을 통해 가장 좋았던 상태로 되돌려줌
  - 단점
    1. 지역 최적해 : 나중에 좋아질 수도 있는데 너무 일찍 멈출수도 있다. (보완: ReduceLROnPlateau와 콤보 사용)
    2. 기준의 모호함 : patience를 정하는 명확한 기준이 없다.
   
<코드>
```python

```
