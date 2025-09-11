# Deep learning 25_2 : Assignment Readme


This repository contains my solutions to the assignments of the Deep Learning class
offered by Professor Heewon Kim at Soongsil University (2nd semester, 2025).

The class is part of [RealityLab](https://reality.ssu.ac.kr/), which focuses on research 
in deep learning and related areas.

Assignments are implemented in Python with frameworks such as TensorFlow and PyTorch.


# 📘 프로젝트 1: k-Nearest Neighbor (kNN) 분류기 구현

이 프로젝트는 **CIFAR-10 이미지 데이터셋**을 대상으로 k-Nearest Neighbor 분류기를 구현하고 성능을 분석하는 과제입니다.  
제공된 코드 스켈레톤을 기반으로 핵심 함수를 직접 작성하며, 거리 계산 방식의 차이와 k 값 변화가 분류 정확도에 어떤 영향을 주는지 실험합니다.

---

## ⚙️ 실습 환경 설정

1. Conda 가상 환경 생성:

````bash
   conda create --name ssu_knn python=3.10
   conda activate ssu_knn
````

2. 필수 라이브러리 설치:

```bash
   pip install numpy==2.2.6
   pip install opencv-python==4.12.0.88
   pip install Pillow==11.3.0
```

---

## 🧪 실습할 내용

1. **데이터 로드 및 전처리**

   * CIFAR-10 dataset load function 참고 (https://github.com/SSU-Reality-Lab/deep_learning_25_2-week-1-knn-assignment-DL_2025_2/blob/master/data_utils.py)
   
2. **kNN 분류기 구현**

   * KNearestNeighbor Class 파일 참고 (https://github.com/SSU-Reality-Lab/deep_learning_25_2-week-1-knn-assignment-DL_2025_2/blob/master/k_nearest_neighbor.py)
   * `compute_distances_two_loops`: 이중 반복문으로 거리 계산
   * `compute_distances_one_loop`: 반복문 하나만 사용
   * `compute_distances_no_loops`: 완전 벡터화 연산으로 거리 계산
   * `predict_labels`: k개의 최근접 이웃을 이용한 Label 예측

---

## 🚀 실행 방법

 * `KNearestNeighbor` 클래스를 import하고 학습 데이터를 불러와, 거리 계산 함수를 호출해 데이터를 분류합니다.

---

## ❓ 질문 방법

* 코드 실행 에러나 환경 문제: 조교 메일 문의 ([por1329@naver.com](mailto:por1329@naver.com))
* 구현 아이디어/개념 관련: 강의 자료 및 QnA 활용
* **주의:** 지정된 Conda 환경을 사용하지 않아 발생한 문제는 답변하지 않음

---

## 🚨 주의사항

* Numpy 기본 연산과 반복문을 사용하여 직접 구현할 것
* `cv2`, `scipy`, `sklearn` 등 거리 계산 및 분류를 자동 처리하는 라이브러리 함수 사용 금지
* 무단 코드 복사/붙여넣기 적발 시 0점 처리
