import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import subprocess
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ===== 데이터 전처리 =====

# 훈련 데이터 metadata 경로 및 폴더 경로 지정
train_metadata_path = '../2501ml_data/label/train_label.txt'
train_data_path = '../2501ml_data/train'

# 훈련 데이터의 음성 파일과 레이블을 저장할 리스트 초기화
train_x = []
train_y = []

# 메타데이터 파일 열기
with open(train_metadata_path, 'r') as f:
    for line in f:
        # 화자, 파일 이름, 기타 정보, 라벨로 분리
        spk, file_name, _, _, label = line.strip().split(' ')

        # 음성 파일 경로
        wav_path = os.path.join(train_data_path, file_name)

        # MFCC, Delta, Delta-Delta 특징 추출
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # 각 특징(mfcc, mfcc_delta, mfcc_delta2)의 평균과 표준편차를 결합
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

        # 모든 통계량 특징 결합
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            mfcc_delta_mean, mfcc_delta_std,
            mfcc_delta2_mean, mfcc_delta2_std
        ])

        train_x.append(features)
        train_y.append(label)

print("훈련 데이터 개수:", len(train_x))
print("훈련 데이터 X의 shape:", train_x[0].shape)

# 훈련 데이터 클래스 분포 확인 (시각화는 제거)
train_label_counts = Counter(train_y)
print("\n훈련 데이터 클래스 분포:", train_label_counts)

# ===== SVM 모델 학습 전 특징 스케일링 및 하이퍼파라미터 튜닝 =====

train_x = np.array(train_x)
train_y = np.array(train_y)

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x) # 훈련 데이터에 맞춰 스케일러 학습 및 변환

print("훈련 데이터 스케일링 완료. 스케일링 후 X의 shape:", train_x_scaled.shape)

# GridSearchCV를 위한 하이퍼파라미터 그리드 정의 (과적합 완화를 위한 C, gamma 조정)
param_grid = {
    'C': [0.01, 0.1, 1], # C 값을 더 낮게 설정
    'kernel': ['rbf', 'linear'], # rbf 우선, linear 포함
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01] # gamma 값을 더 낮게 설정
}

# SVC 모델 객체 생성
svm = SVC(random_state=42) # 재현성을 위해 random_state 설정

# GridSearchCV 객체 생성
# cv: 교차 검증 폴드 수 (일반적으로 5 또는 10)
# verbose: 진행 상황 출력 레벨 (높을수록 상세)
# n_jobs: 사용할 코어 수 (-1은 모든 코어 사용)
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2, n_jobs=-1)

print("GridSearchCV를 통한 최적의 하이퍼파라미터 탐색 시작...")
grid_search.fit(train_x_scaled, train_y) # 스케일링된 훈련 데이터로 튜닝
print("GridSearchCV 탐색 완료.")

# 최적의 하이퍼파라미터 및 최고 성능 출력
print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최고 교차 검증 점수:", grid_search.best_score_)

# 최적의 모델로 svm_model 업데이트
svm_model = grid_search.best_estimator_

# ===== 테스트 데이터 로딩 =====

# 테스트 메타데이터 경로 및 데이터 폴더 경로 지정
test_metadata_path = '../2501ml_data/label/test_label.txt'
test_data_path = '../2501ml_data/test'

test_x = []
test_y = []
test_file_names = []

with open(test_metadata_path, 'r') as f:
    for line in f:
        spk, file_name, _, _, label = line.strip().split(' ')
        wav_path = os.path.join(test_data_path, file_name)

        # MFCC, Delta, Delta-Delta 특징 추출
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # 각 특징(mfcc, mfcc_delta, mfcc_delta2)의 평균과 표준편차를 결합
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

        # 모든 통계량 특징 결합
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            mfcc_delta_mean, mfcc_delta_std,
            mfcc_delta2_mean, mfcc_delta2_std
        ])

        test_x.append(features)
        test_y.append(label)
        test_file_names.append(file_name)

test_x = np.array(test_x)
test_y = np.array(test_y)

# 테스트 데이터 스케일링 (훈련 데이터의 스케일러를 그대로 사용)
test_x_scaled = scaler.transform(test_x) # 훈련 데이터에 학습된 스케일러로 변환

print("테스트 데이터 스케일링 완료. 스케일링 후 X의 shape:", test_x_scaled.shape)


# 테스트 데이터 클래스 분포 확인 (시각화는 제거)
test_label_counts = Counter(test_y)
print("\n테스트 데이터 클래스 분포:", test_label_counts)

# ===== 예측 및 평가 =====

predictions = svm_model.predict(test_x_scaled) # 스케일링된 테스트 데이터로 예측
print("예측 결과:", predictions)

# 혼동 행렬 출력
cm = confusion_matrix(test_y, predictions)
print("Confusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Tuned SVM Model (Reduced Overfitting)') # 제목 업데이트
plt.show()

# ===== 결과 파일 저장 =====

result_path = './team_test_result.txt'
with open(result_path, 'w') as f:
    for i in range(len(predictions)):
        f.write(f"{test_file_names[i]} {predictions[i]}\n")

# ===== 평가 스크립트 실행 (Perl) =====

subprocess.run(['perl', '../eval.pl', result_path, '../2501ml_data/label/test_label.txt'])