추가 설명서 구조를 반영해 과제 Readme를 확장했습니다. `knn.ipynb` 파일을 확인했는데, 코드와 실험 결과가 담겨 있어 실행/보고 부분을 포함하는 게 적절합니다. 아래에 제안된 완성본을 드립니다.

---

# Deep learning 25\_2 : Assignment Readme

This repository contains my solutions to the assignments of the Deep Learning class offered by Professor Heewon Kim at Soongsil University (2nd semester, 2025).

The class is part of [RealityLab](https://reality.ssu.ac.kr/), which focuses on research in deep learning and related areas.

---

# 📘 프로젝트 1: k-Nearest Neighbor (kNN) 분류기 구현

이 프로젝트는 **CIFAR-10 이미지 데이터셋**을 대상으로 k-Nearest Neighbor 분류기를 구현하고 성능을 분석하는 과제입니다.
제공된 코드 스켈레톤을 기반으로 핵심 함수를 직접 작성하며, 거리 계산 방식의 차이와 k 값 변화가 분류 정확도에 어떤 영향을 주는지 실험합니다.

---

## ⚙️ 실습 환경 설정

1. Conda 가상 환경 생성:

```bash
conda create --name ssu_knn python=3.10
conda activate ssu_knn
```

2. 필수 라이브러리 설치:

```bash
pip install numpy==2.2.6
pip install opencv-python==4.12.0.88
pip install Pillow==11.3.0
```

---

## 🧪 실습할 내용

1. **kNN 분류기 구현**

   * KNearestNeighbor Class 참고 ([k\_nearest\_neighbor.py](https://github.com/SSU-Reality-Lab/deep_learning_25_2-week-1-knn-assignment-DL_2025_2/blob/master/k_nearest_neighbor.py))
   * `compute_distances_two_loops`: 이중 반복문으로 거리 계산
   * `compute_distances_one_loop`: 반복문 하나만 사용
   * `compute_distances_no_loops`: 완전 벡터화 연산으로 거리 계산
   * k 값을 변화시키며 성능 비교

2. **성능 평가**

   * validation set을 이용해 최적 k 선택
   * test set 분류 정확도 측정
   * distance 계산 방식별 속도 및 정확도 비교

---

## 🚀 실행 방법

1. Jupyter Notebook 실행:

```bash
jupyter notebook knn.ipynb
```

2. `KNearestNeighbor` 클래스를 import 후 학습 데이터를 불러옵니다.
3. 거리 계산 함수를 각각 호출해 분류를 수행하고 결과를 기록합니다.
4. Notebook 내 그래프와 출력 결과를 통해 분석합니다.

---

## 📊 결과 보고

* 본 reop를 본인 컴퓨터에 git pull하시고 k_nn.py과 knn.ipynb를 완성하시오.
* 그 다음 output 폴더 하나를 생성하시고 실습한 k_nn.py과 실행 로그가 담겨있는 knn.ipynb을 제출하시요.
* git push를 하면 자동으로 과제가 제출됩니다.

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
