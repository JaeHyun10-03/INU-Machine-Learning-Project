# ===== 필수 라이브러리 명시 =====
# 프로젝트에서 사용하는 라이브러리들을 명시
# librosa  # 오디오 분석 및 특징 추출을 위한 라이브러리
# matplotlib  # 데이터 시각화를 위한 라이브러리
# numpy  # 수치 연산 및 배열 처리 라이브러리
# scikit-learn  # 머신러닝 알고리즘 및 평가 도구 라이브러리
# ipykernel  # 주피터 노트북에서 실행하기 위한 커널
# === 위에 적혀진 라이브러리만 사용해야해 ===

# ===== 라이브러리 임포트 =====
import os  # 파일 경로 처리
import librosa  # 오디오 파일 로드 및 특징 추출
import matplotlib.pyplot as plt  # 그래프 및 시각화
import numpy as np  # 수치 연산 및 배열 처리
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # 모델 평가 도구
import subprocess  # 외부 프로그램 실행 (Perl 스크립트)
from collections import Counter  # 데이터 개수 세기
from sklearn.preprocessing import StandardScaler  # 데이터 표준화
from sklearn.model_selection import GridSearchCV  # 하이퍼파라미터 최적화

# ===== 데이터 전처리 섹션 =====

# 훈련 데이터의 메타데이터 파일 경로 (파일명과 라벨 정보가 담긴 텍스트 파일)
train_metadata_path = '../2501ml_data/label/train_label.txt'
# 실제 음성 파일들이 저장된 폴더 경로
train_data_path = '../2501ml_data/train'

# 특징 벡터와 라벨을 저장할 빈 리스트 초기화
train_x = []  # 음성 파일에서 추출한 특징들을 저장
train_y = []  # 해당 음성의 감정 라벨을 저장

# 메타데이터 파일을 한 줄씩 읽어서 처리
with open(train_metadata_path, 'r') as f:
    for line in f:
        # 각 줄을 공백으로 분리: 화자ID, 파일명, 기타정보1, 기타정보2, 감정라벨
        spk, file_name, _, _, label = line.strip().split(' ')

        # 실제 음성 파일의 전체 경로 생성
        wav_path = os.path.join(train_data_path, file_name)

        # ===== 음성 특징 추출 과정 =====
        # librosa로 음성 파일 로드 (샘플링 레이트 16kHz로 통일)
        y, sr = librosa.load(wav_path, sr=16000)
        
        # MFCC(Mel-frequency cepstral coefficients) 추출
        # 인간의 청각 특성을 반영한 주요 음성 특징 (20차원)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Delta 특징: MFCC의 1차 미분 (시간에 따른 변화율)
        # 음성의 동적 특성을 캡처
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Delta-Delta 특징: MFCC의 2차 미분 (변화율의 변화율)
        # 음성의 가속도 정보를 캡처
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # ===== 통계적 특징 계산 =====
        # 각 특징의 시간축에 대한 평균과 표준편차 계산
        # 가변 길이의 음성을 고정 길이 특징 벡터로 변환
        mfcc_mean = np.mean(mfcc, axis=1)          # MFCC 평균 (20차원)
        mfcc_std = np.std(mfcc, axis=1)            # MFCC 표준편차 (20차원)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)      # Delta 평균 (20차원)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)        # Delta 표준편차 (20차원)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)    # Delta-Delta 평균 (20차원)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)      # Delta-Delta 표준편차 (20차원)

        # 모든 통계적 특징을 하나의 벡터로 결합 (총 120차원)
        features = np.concatenate([
            mfcc_mean, mfcc_std,           # 40차원
            mfcc_delta_mean, mfcc_delta_std,   # 40차원
            mfcc_delta2_mean, mfcc_delta2_std  # 40차원
        ])

        # 특징 벡터와 라벨을 리스트에 추가
        train_x.append(features)
        train_y.append(label)

# 데이터 로딩 결과 확인
print("훈련 데이터 개수:", len(train_x))
print("훈련 데이터 X의 shape:", train_x[0].shape)

# 각 감정 클래스별 데이터 개수 확인 (데이터 불균형 체크)
train_label_counts = Counter(train_y)
print("\n훈련 데이터 클래스 분포:", train_label_counts)

# ===== SVM 모델 학습 전 데이터 전처리 =====

# 리스트를 numpy 배열로 변환 (scikit-learn 호환성)
train_x = np.array(train_x)
train_y = np.array(train_y)

# 특징 스케일링 (표준화)
# SVM은 특징들의 스케일에 민감하므로 표준화 필수
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)  # 평균=0, 표준편차=1로 변환

print("훈련 데이터 스케일링 완료. 스케일링 후 X의 shape:", train_x_scaled.shape)

# ===== 하이퍼파라미터 튜닝을 위한 그리드 서치 =====

# 과적합을 완화하기 위해 조정된 하이퍼파라미터 범위 설정
param_grid = {
    'C': [0.01, 0.1, 1],  # 정규화 강도 (낮을수록 더 강한 정규화, 과적합 방지)
    'kernel': ['rbf', 'linear'],  # 커널 함수 (rbf: 비선형, linear: 선형)
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01]  # RBF 커널의 영향 범위
}

# SVM 분류기 객체 생성 (재현 가능한 결과를 위해 random_state 설정)
svm = SVC(random_state=42)

# 그리드 서치 객체 생성
# cv=5: 5-fold 교차 검증으로 성능 평가
# verbose=2: 상세한 진행 상황 출력
# n_jobs=-1: 모든 CPU 코어 사용으로 병렬 처리
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2, n_jobs=-1)

print("GridSearchCV를 통한 최적의 하이퍼파라미터 탐색 시작...")
# 모든 파라미터 조합에 대해 교차 검증 수행
grid_search.fit(train_x_scaled, train_y)
print("GridSearchCV 탐색 완료.")

# 최적의 결과 출력
print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최고 교차 검증 점수:", grid_search.best_score_)

# 최적의 파라미터로 훈련된 모델 선택
svm_model = grid_search.best_estimator_

# ===== 테스트 데이터 로딩 및 전처리 =====

# 테스트 데이터 경로 설정
test_metadata_path = '../2501ml_data/label/test_label.txt'
test_data_path = '../2501ml_data/test'

# 테스트 데이터 저장 리스트 초기화
test_x = []
test_y = []
test_file_names = []  # 결과 파일 작성을 위한 파일명 저장

# 테스트 데이터도 훈련 데이터와 동일한 방식으로 처리
with open(test_metadata_path, 'r') as f:
    for line in f:
        spk, file_name, _, _, label = line.strip().split(' ')
        wav_path = os.path.join(test_data_path, file_name)

        # 훈련 데이터와 동일한 특징 추출 과정
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # 통계적 특징 계산
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

        # 특징 벡터 결합
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            mfcc_delta_mean, mfcc_delta_std,
            mfcc_delta2_mean, mfcc_delta2_std
        ])

        test_x.append(features)
        test_y.append(label)
        test_file_names.append(file_name)

# 테스트 데이터 배열 변환
test_x = np.array(test_x)
test_y = np.array(test_y)

# 테스트 데이터 스케일링 (중요: 훈련 데이터의 스케일러 사용)
# 새로운 스케일러를 만들지 않고 기존 스케일러로 변환해야 함
test_x_scaled = scaler.transform(test_x)

print("테스트 데이터 스케일링 완료. 스케일링 후 X의 shape:", test_x_scaled.shape)

# 테스트 데이터의 클래스 분포 확인
test_label_counts = Counter(test_y)
print("\n테스트 데이터 클래스 분포:", test_label_counts)

# ===== 모델 예측 및 평가 =====

# 훈련된 SVM 모델로 테스트 데이터 예측
predictions = svm_model.predict(test_x_scaled)
print("예측 결과:", predictions)

# 혼동 행렬(Confusion Matrix) 생성 및 시각화
# 실제 라벨과 예측 라벨 간의 관계를 행렬 형태로 표현
cm = confusion_matrix(test_y, predictions)
print("Confusion Matrix:")

# 혼동 행렬 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues)  # 파란색 계열 색상맵 사용
plt.title('Confusion Matrix for Tuned SVM Model (Reduced Overfitting)')
plt.show()

# ===== 결과 파일 저장 =====

# 예측 결과를 텍스트 파일로 저장
result_path = './team_test_result.txt'
with open(result_path, 'w') as f:
    for i in range(len(predictions)):
        # 파일명과 예측된 라벨을 한 줄씩 저장
        f.write(f"{test_file_names[i]} {predictions[i]}\n")

# ===== 평가 스크립트 실행 =====

# Perl로 작성된 외부 평가 스크립트 실행
# 예측 결과 파일과 정답 라벨 파일을 비교하여 성능 평가
subprocess.run(['perl', '../eval.pl', result_path, '../2501ml_data/label/test_label.txt'])