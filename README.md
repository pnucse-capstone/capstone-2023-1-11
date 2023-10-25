<h2>심장질환 환자 ECG 데이터 분석을 위한 딥러닝 기법 설계 및 경량화 모델 구축</h2>

### 1. 프로젝트 소개
프로젝트 명, 목적, 개요 등 프로젝트에 대한 간단한 소개글을 작성하세요.

심장질환 환자의 심장 박동 패턴을 분석하고 비정상적인 패턴을 감지하여 분류 대상이 되는 심전도가 어떤 종류의 심장질환에 속하는 지 결정하는 딥러닝 모델을 구축하는 것이다. 부정맥과 관상동맥질환(심장동맥질환)의 진단에 많은 검사들이 이용되고 있으나, 그 중에서도 심전도(Electrocardiogram : ECG)를 이용한 검사는 많은 장점(비침습적, 빠르고 간편한 수행)을 가지며 임상에서 가장 많이 사용되는 검사이다. 이러한 모델은 심장질환 환자의 심장 박동 패턴을 실시간으로 모니터링하는 장치에 설치되어 환자의 건강 상태를 지속적으로 추적하고 긴급한 조치를 취할 수 있는 기회를 제공한다.
 더 넓은 응용 범위와 다양한 환경에서 이 모델을 활용하기 위해서는 모델을 경량화하여 임베디드 시스템에도 적용할 수 있어야 한다. 이를 위해  CNN을 이용하여 심전도 데이터를 분석하고 분류할 수 있는 딥러닝 모델을 개발하고, 이를 최적화하여 경량화하여 보는 것은 많은 도움이 될 것으로 생각하여 연구를 진행하게 되었다.

### 2. 팀 소개

프로젝트에 참여한 팀원들의 이름, 이메일, 역할를 포함해 팀원들을 소개하세요.
| 이름     | 이메일                  | 역할                                       |
|---------|-------------------------|------------------------------------------|
| 신다윗   | sin002277@gmail.com     | Inception을 이용한 CNN 모델 및 양자화 모델, Pruning 모델 설계 및 학습 |
| 안주현   | gsong@pusan.ac.kr       | CNN Base 모델 및 양자화 모델, 지식 증류 모델 설계 및 학습       |
| 이재현   | jaeback2435@naver.com   | CNN Base 모델 및 양자화 모델 설계 및 학습             |


### 3. 구성도

# PTB-XL 데이터 세트

PTB-XL 데이터 세트는 임상 12-lead ECG로 구성된 10초 길이의 데이터로, 총 18,869명의 환자로부터 수집되었습니다.

## 데이터 세트 정보

- **버전:** 1.0.3
- **출처:** [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)

## 데이터 구성

PTB-XL 데이터 세트의 주요 특징:

- **환자 수:** 18,869명
- **ECG 리드:** 12개 (12-lead ECG)
- **데이터 포인트 길이:** 10초

## 슈퍼클래스 정보

PTB-XL 데이터 세트의 슈퍼클래스 정보:

| 슈퍼클래스 | 레코드 수 | 설명                    |
| ----------- | ---------- | ------------------------ |
| NORM        | 9,514      | 정상 ECG 결과를 나타냅니다. 이 범주에 속하는 기록들은 심장 활동이 정상적으로 기록되었으며 심전도에 이상이 없음을 나타냅니다. |
| MI          | 5,469      | 심근경색(Myocardial Infarction)으로 인한 ECG 결과를 나타냅니다. 이 범주에 속하는 기록들은 심근경색의 증상을 보이며, 이는 심장 근육에 손상이 있음을 나타냅니다. |
| STTC        | 5,235      | ST 세그먼트와 T 웨이브에 변화가 있는 ECG 결과를 나타냅니다. 이러한 변화는 심장의 전기적인 활동에 영향을 미치며 다양한 의료 상황을 나타낼 수 있습니다. |
| CD          | 4,898      | 전도 장애(Conduction Disturbance)에 기인한 ECG 결과를 나타냅니다. 심장의 전기적인 전달에 이상이 있는 경우, 이러한 변화가 나타납니다. |
| HYP         | 2,649      | 심근 비대(Hypertrophy)로 인한 ECG 결과를 나타냅니다. 심근 비대는 심장 근육의 증대를 의미하며, 이러한 변화가 ECG에서 확인될 수 있습니다. |

## 환자 0의 12-Lead ECG 그래프

![환자 0의 12 Lead ECG 그래프](https://user-images.githubusercontent.com/49470426/277953392-337808e7-7c51-4fa6-b661-31ed4ec2146e.png)
*환자 0의 12-Lead ECG 그래프*

## 다중 레이블 클래스 수

![다중 레이블 클래스 수](https://user-images.githubusercontent.com/49470426/277953543-4a9b9190-04af-48d9-94a7-290d97e0651d.png)
*다중 레이블 클래스 수*

# Inception CNN Model with Batch Normalization

| Layer                  | Output Shape       | Param #   | Connected to                            |
|------------------------|--------------------|-----------|-----------------------------------------|
| input_1 (InputLayer)   | (None, 1000, 12)   | 0         |                                         |
| conv1d                 | (None, 1000, 32)   | 2720      | input_1[0][0]                           |
| batch_normalization    | (None, 1000, 32)   | 128       | conv1d[0][0]                            |
| conv1d_1               | (None, 1000, 128)  | 20608     | batch_normalization[0][0]              |
| max_pooling1d          | (None, 500, 128)   | 0         | conv1d_1[0][0]                         |
| batch_normalization_1  | (None, 500, 128)   | 512       | max_pooling1d[0][0]                    |
| conv1d_2               | (None, 500, 64)    | 57408     | batch_normalization_1[0][0]            |
| max_pooling1d_1        | (None, 250, 64)    | 0         | conv1d_2[0][0]                         |
| batch_normalization_2  | (None, 250, 64)    | 256       | max_pooling1d_1[0][0]                  |
| conv1d_3               | (None, 250, 64)    | 20544     | batch_normalization_2[0][0]            |
| max_pooling1d_2        | (None, 125, 64)    | 0         | conv1d_3[0][0]                         |
| batch_normalization_3  | (None, 125, 64)    | 256       | max_pooling1d_2[0][0]                  |
| conv1d_5               | (None, 125, 64)    | 4160      | batch_normalization_3[0][0]            |
| conv1d_7               | (None, 125, 64)    | 4160      | batch_normalization_3[0][0]            |
| max_pooling1d_3        | (None, 125, 64)    | 0         | batch_normalization_3[0][0]            |
| conv1d_4               | (None, 125, 64)    | 4160      | batch_normalization_3[0][0]            |
| conv1d_6               | (None, 125, 64)    | 12352     | conv1d_5[0][0]                         |
| conv1d_8               | (None, 125, 64)    | 20544     | conv1d_7[0][0]                         |
| conv1d_9               | (None, 125, 64)    | 4160      | max_pooling1d_3[0][0]                  |
| concatenate            | (None, 125, 256)   | 0         | conv1d_4[0][0], conv1d_6[0][0],        |
|                        |                    |           | conv1d_8[0][0], conv1d_9[0][0]        |
| flatten                | (None, 32000)       | 0         | concatenate[0][0]                     |
| dense                  | (None, 128)         | 4096128   | flatten[0][0]                          |
| batch_normalization_4  | (None, 128)         | 512       | dense[0][0]                            |
| dense_1                | (None, 128)         | 16512     | batch_normalization_4[0][0]            |
| batch_normalization_5  | (None, 128)         | 512       | dense_1[0][0]                          |
| dense_2                | (None, 5)           | 645       | batch_normalization_5[0][0]            |
|------------------------|--------------------|-----------|-----------------------------------------|
| Total params           |                    | 4266277   | Trainable: 4265189, Non-trainable: 1088 |


# Visualization of Inception CNN Model with Batch Normalization
![Inception CNN Model with Batch Normalization](https://user-images.githubusercontent.com/49470426/278013827-73750cb5-ea11-44bd-9cbc-cdf6a4f009df.png)
*Visualization of Inception CNN Model with Batch Normalization*

### 4. 소개 및 시연 영상

프로젝트 소개나 시연 영상을 넣으세요.

### 5. 사용법

프로젝트 결과을 사용 위해 필요한 소프트웨어 요구사항 및 설치법, 그리고 간단한 사용법을 작성하세요.

## 시스템 및 소프트웨어 정보

- 운영 체제: Windows 10 Pro
- 프로세서: Intel(R) Core(TM) i7-7700HQ CPU
- 운영 체제 아키텍처: 64비트, x64 기반 프로세서
- 그래픽 카드: NVIDA Geforce GTX 1060 6GB

### Anaconda 및 Python 환경

- Anaconda 버전: conda 23.5.2
- Python 버전: 3.9.18

### CUDA 및 딥러닝 프레임워크

- CUDA 버전: 11.8.0_522.06_windows
- Torch 버전: 2.0.1+cu118 (CUDA 11.8 지원)
- Tensorflow-gpu 버전: 2.10.1 (CUDA 11.2 지원)

