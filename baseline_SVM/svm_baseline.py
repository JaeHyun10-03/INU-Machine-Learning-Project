# ===== 필수 라이브러리 임포트 =====
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

# ===== 주요 파라미터 설정 =====
TARGET_SR = 16000
N_MFCC = 35
USE_DELTA = True
USE_DELTA2 = True
USE_SPECTRAL_FEATURES = True

# ===== 데이터 경로 설정 =====
train_metadata_path = '../2501ml_data/label/train_label.txt'
train_data_path = '../2501ml_data/train'
test_metadata_path = '../2501ml_data/label/test_label.txt'
test_data_path = '../2501ml_data/test'

# ===== 특징 추출 함수 =====
def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=TARGET_SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = [mfcc_mean, mfcc_std]

    if USE_DELTA:
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        features += [mfcc_delta_mean, mfcc_delta_std]

    if USE_DELTA2:
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)
        features += [mfcc_delta2_mean, mfcc_delta2_std]

    # ===== 추가 특징 =====
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    spec_centroid_mean = np.mean(spectral_centroid)
    spec_centroid_std = np.std(spectral_centroid)
    spec_bandwidth_mean = np.mean(spectral_bandwidth)
    spec_bandwidth_std = np.std(spectral_bandwidth)
    flatness_mean = np.mean(spectral_flatness)
    flatness_std = np.std(spectral_flatness)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    frame_length = 2048
    hop_length = 512
    ste = np.array([
        np.sum(np.square(y[i:i + frame_length]))
        for i in range(0, len(y) - frame_length + 1, hop_length)
    ])
    ste_mean = np.mean(ste)
    ste_std = np.std(ste)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # ===== jitter 계산 (pyin 기반) =====
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        if f0 is not None and np.sum(voiced_flag) > 3:
            f0_voiced = f0[voiced_flag]
            periods = 1.0 / f0_voiced
            jitter = np.std(np.diff(periods)) / np.mean(periods)
        else:
            jitter = 0.0
    except:
        jitter = 0.0

    extra_features = np.array([
        zcr_mean, zcr_std,
        ste_mean, ste_std,
        rms_mean, rms_std,
        jitter  # 새로운 특징
    ])

    features += [extra_features]

    return np.concatenate(features)

# ===== 데이터 로딩 함수 =====
def load_data(metadata_path, data_path):
    x = []
    y = []
    file_names = []
    jitter_list = []
    label_list = []

    with open(metadata_path, 'r') as f:
        for line in f:
            spk, file_name, _, _, label = line.strip().split(' ')
            wav_path = os.path.join(data_path, file_name)
            features = extract_features(wav_path)
            x.append(features)
            y.append(label)
            file_names.append(file_name)
            jitter_list.append(features[-1])  # 마지막 값이 jitter
            label_list.append(label)

    return np.array(x), np.array(y), file_names, np.array(jitter_list), np.array(label_list)

# ===== 훈련 데이터 로딩 =====
train_x, train_y, _, train_jitter, train_labels = load_data(train_metadata_path, train_data_path)
print("훈련 데이터 개수:", len(train_x))
print("훈련 데이터 X의 shape:", train_x[0].shape)
print("\n훈련 데이터 클래스 분포:", Counter(train_y))

# ===== jitter 시각화 =====
plt.hist(train_jitter[train_labels == 'real'], bins=30, alpha=0.5, label='real')
plt.hist(train_jitter[train_labels == 'spoof'], bins=30, alpha=0.5, label='spoof')
plt.xlabel("Jitter Value")
plt.ylabel("Count")
plt.legend()
plt.title("Jitter Distribution by Class (Train Set)")
plt.show()

# ===== 테스트 데이터 로딩 =====
test_x, test_y, test_file_names, _, _ = load_data(test_metadata_path, test_data_path)
print("테스트 데이터 개수:", len(test_x))
print("테스트 데이터 X의 shape:", test_x[0].shape)
print("\n테스트 데이터 클래스 분포:", Counter(test_y))

# ===== 데이터 스케일링 =====
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)
print("스케일링 완료. 훈련 데이터 shape:", train_x_scaled.shape, "테스트 데이터 shape:", test_x_scaled.shape)

# ===== SVM 하이퍼파라미터 그리드 설정 =====
param_grid = {
    'C': [0.1, 0.4, 0.5, 0.6, 1],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01]
}

# ===== SVM 모델 학습 및 하이퍼파라미터 튜닝 =====
svm = SVC(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2, n_jobs=-1)
print("GridSearchCV를 통한 최적의 하이퍼파라미터 탐색 시작...")
grid_search.fit(train_x_scaled, train_y)
print("GridSearchCV 탐색 완료.")

print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최고 교차 검증 점수:", grid_search.best_score_)
svm_model = grid_search.best_estimator_

# ===== 테스트 데이터 예측 =====
predictions = svm_model.predict(test_x_scaled)
print("예측 결과:", predictions)

# ===== 혼동 행렬 시각화 =====
cm = confusion_matrix(test_y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Tuned SVM Model (Enhanced + Jitter)')
plt.show()

# ===== 결과 저장 =====
result_path = './team_test_result.txt'
with open(result_path, 'w') as f:
    for i in range(len(predictions)):
        f.write(f"{test_file_names[i]} {predictions[i]}\n")

# ===== 평가 스크립트 실행 =====
subprocess.run(['perl', '../eval.pl', result_path, '../2501ml_data/label/test_label.txt'])
