# Overlap Report

## 중복 분석 기준

- `duplicate_files.csv`: SHA256 해시가 같은 완전 동일 파일
- `name_duplicates.csv`: 파일명이 같은 파일
- `topic_inventory.csv`: 파일명/경로 키워드 기반 주제 분류

## 완전 중복

현재 완전 중복으로 잡힌 항목은 8개 행이다.

- `ML_sample.ipynb`
- `국가데이터처_나라통계_우편번호_20211110.csv`
- `online_retail_uci_torch_association_rules_colab_모범답안.ipynb`
- `실습문제_7_모범답안_online_retail_uci_torch_association_rules_colab.ipynb`
- `cifar10_cnn_pytorch.ipynb`

## 같은 이름 중복

현재 같은 파일명 중복으로 잡힌 항목은 20개 행이다.

대표 예시는 다음과 같다.

- `iris_logistic_regression_pytorch.ipynb`
- `pytorch_knn_breast_cancer.ipynb`
- `pytorch_basic_using.ipynb`
- `cifar10_cnn_pytorch.ipynb`
- `data.zip`
- `iris_scaler.pkl`
- `iris_random_forest_model.pkl`

## 다음 분석 필요

해시와 파일명 기준은 1차 정리다. 다음 단계에서는 notebook 코드 셀만 추출해 다음 기준으로 유사도를 계산한다.

- 같은 데이터셋 + 같은 모델
- 같은 모델 + 다른 하이퍼파라미터
- 원본/실습/튜닝/완성본 관계
- 성능 개선 전후 비교

