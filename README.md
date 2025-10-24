# codeit_ai_base_project

코드잇 AI 초급 프로젝트
파일 설명
📂 notebooks/ - Jupyter 노트북들
전처리.ipynb: 데이터 로딩 및 기본 EDA

train.parquet, test.parquet 불러오기
데이터 구조 파악 및 결측치 확인
주최측의 변수 정보 비공개로 인한 제한적 분석
cat_train.ipynb: CatBoost 모델 학습

5개의 다른 시드(0~4)로 각각 5-Fold CV 수행
Optuna 하이퍼파라미터 튜닝, Early Stopping
다운샘플링 및 K-Fold Target Encoding
총 25개 모델 파일(.cbm) 저장
hist_train.ipynb: HistGradientBoosting 모델 학습

동일한 5-seed × 5-fold 전략
각 시드별 최적 파라미터 적용, 클래스 가중치 조정
총 25개 모델 파일(.joblib) 저장
xgb_train.ipynb: XGBoost 모델 학습

동일한 5-seed × 5-fold 전략
scale_pos_weight를 통한 클래스 불균형 대응
총 25개 모델 파일(.json) 저장
inference.ipynb: 최종 앙상블 추론

CatBoost, HistGradientBoosting, XGBoost 세 모델의 예측값 로드
앙상블 전략 적용 (가중 평균 또는 단순 평균)
최종 제출 파일 생성
📂 scripts/ - 재사용 가능한 Python 모듈들
data_preprocessing.py: 데이터 전처리 유틸리티

데이터 로딩, 다운샘플링, K-Fold Target Encoding
Label Encoding, 시간 피처 생성 함수들
model_training.py: 모델 학습 유틸리티

CatBoost, HistGradientBoosting, XGBoost CV 학습 함수들
대회 평가 지표 계산, 모델 저장/로드 기능
ensemble_utils.py: 앙상블 유틸리티

예측 결과 로딩 및 앙상블, 가중 평균 계산
제출 파일 생성, 앙상블 조합 성능 평가
📂 config/ - 설정 파일들
model_config.py: 모델 하이퍼파라미터 및 실험 설정

CatBoost, HGB, XGBoost 파라미터
CV 설정, 전처리 설정, 앙상블 가중치
파일 경로 및 Optuna 튜닝 설정
requirements.txt: 패키지 의존성 목록

📂 models/ - 학습된 모델 파일들
cat_model/: CatBoost 모델들 (25개 .cbm 파일)
hist_model/: HistGradientBoosting 모델들 (25개 .joblib 파일)
xgb_model/: XGBoost 모델들 (25개 .json 파일)
📂 data/ - 데이터 파일들 (gitignore 적용)
train.parquet: 학습 데이터 (사용자 행동 데이터, clicked 타겟 변수)
test.parquet: 테스트 데이터 (ID 제외하고 사용)
sample_submission.csv: 제출 파일 템플릿
📂 outputs/ - 출력 파일들 (gitignore 적용)
예측 결과 파일들 및 제출 파일들
