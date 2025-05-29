import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import subprocess

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

        # 특징 결합 및 평균
        mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        mfcc_mean = np.mean(mfcc_combined, axis=1)

        train_x.append(mfcc_mean)
        train_y.append(label)

print("훈련 데이터 개수:", len(train_x))
print("훈련 데이터 X의 shape:", train_x[0].shape)

# ===== SVM 모델 학습 =====

train_x = np.array(train_x)
train_y = np.array(train_y)

svm_model = SVC()
svm_model.fit(train_x, train_y)

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

        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        mfcc_mean = np.mean(mfcc_combined, axis=1)

        test_x.append(mfcc_mean)
        test_y.append(label)
        test_file_names.append(file_name)

test_x = np.array(test_x)
test_y = np.array(test_y)

# ===== 예측 및 평가 =====

predictions = svm_model.predict(test_x)
print("예측 결과:", predictions)

# 혼동 행렬 출력
cm = confusion_matrix(test_y, predictions)
print("Confusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# ===== 결과 파일 저장 =====

result_path = './team_test_result.txt'
with open(result_path, 'w') as f:
    for i in range(len(predictions)):
        f.write(f"{test_file_names[i]} {predictions[i]}\n")

# ===== 평가 스크립트 실행 (Perl) =====

subprocess.run(['perl', '../eval.pl', result_path, '../2501ml_data/label/test_label.txt'])
