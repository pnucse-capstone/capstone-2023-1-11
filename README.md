<h2>심장질환 환자 ECG 데이터 분석을 위한 딥러닝 기법 설계 및 경량화 모델 구축</h2>

### 1. 프로젝트 소개

#### 목적
**목적**: 이 프로젝트의 주요 목적은 심장질환 환자의 심장 박동 패턴을 분석하고 비정상적인 패턴을 감지하여 분류 대상이 되는 심전도가 어떤 종류의 심장질환에 속하는 지 결정하는 딥러닝 모델을 구축하는 것입니다.

#### 개요
**개요**: 부정맥과 관상동맥질환(심장동맥질환)의 진단에 많은 검사들이 이용되고 있으나, 그 중에서도 심전도(Electrocardiogram: ECG)를 이용한 검사는 많은 장점(비침습적, 빠르고 간편한 수행)을 가지며 임상에서 가장 많이 사용되는 검사입니다. 이러한 모델은 심장질환 환자의 심장 박동 패턴을 실시간으로 모니터링하는 장치에 설치되어 환자의 건강 상태를 지속적으로 추적하고 긴급한 조치를 취할 수 있는 기회를 제공합니다.

더 넓은 응용 범위와 다양한 환경에서 이 모델을 활용하기 위해서는 모델을 경량화하여 임베디드 시스템에도 적용할 수 있어야 합니다. 이를 위해 CNN을 이용하여 심전도 데이터를 분석하고 분류할 수 있는 딥러닝 모델을 개발하고, 이를 최적화하여 경량화하여 보는 것은 많은 도움이 될 것으로 생각하여 연구를 진행하게 되었습니다.

### 2. 팀 소개

| 이름     | 이메일                  | 역할                                       |
|---------|-------------------------|------------------------------------------|
| 신다윗   | sin002277@gmail.com     | Inception을 이용한 CNN 모델 및 양자화 모델, Pruning 모델 설계 및 학습 |
| 안주현   | ajmok8703@naver.com       | CNN Base 모델 및 양자화 모델, 지식 증류 모델 설계 및 학습       |
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
*Caption*: 환자 0의 12-Lead ECG 그래프

## 다중 레이블 클래스 수

![다중 레이블 클래스 수](https://user-images.githubusercontent.com/49470426/277953543-4a9b9190-04af-48d9-94a7-290d97e0651d.png)
*Caption*: 다중 레이블 클래스 수

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
||||
| Total params           |                    | 4266277   | Trainable: 4265189, Non-trainable: 1088 |


# Visualization of Inception CNN Model with Batch Normalization
![Inception CNN Model with Batch Normalization](https://user-images.githubusercontent.com/49470426/278013827-73750cb5-ea11-44bd-9cbc-cdf6a4f009df.png)

*Caption*: Visualization of Inception CNN Model with Batch Normalization

# Inception CNN Model with Batch Normalization's Receiver Operating Characteristic (ROC) Curve and AUC for Test Data

![ROC Curve](https://user-images.githubusercontent.com/49470426/278016470-d0309e04-67c7-4dd9-bd8e-236699c40bc6.png)

*Caption*: Inception CNN Model with Batch Normalization's Receiver Operating Characteristic (ROC) Curve and AUC for Test Data

# ROC Curve for Pruned CNN Model on Test Data

![ROC Curve](https://user-images.githubusercontent.com/49470426/278021636-a8007250-0352-48fd-9f8b-b7c969331b5c.png)

*Caption*: ROC Curve for Pruned CNN Model on Test Data

# ROC Curve for Pruned and Quantized CNN Model on Test Data

![ROC Curve](https://user-images.githubusercontent.com/49470426/278022102-951101d9-7a09-4c78-8128-f4ae214ab633.png)

*Caption*: ROC Curve for Pruned and Quantized CNN Model on Test Data

# ROC Curve for Quantized CNN Model on Validation Data

![ROC Curve](https://user-images.githubusercontent.com/49470426/278037305-a53825b0-44b1-4ac5-9442-bf6479430a34.png)

*Caption*: ROC Curve for Quantized CNN Model on Validation Data

# Knowledge Distillation Teacher Model

| Layer (type)          | Output Shape       | Param #   |
|-----------------------|--------------------|-----------|
| Conv1d-1              | [-1, 100, 991]     | 12,100    |
| BatchNorm1d-2         | [-1, 100, 991]     | 200       |
| ReLU-3                | [-1, 100, 991]     | 0         |
| MaxPool1d-4           | [-1, 100, 495]     | 0         |
| Dropout-5             | [-1, 100, 495]     | 0         |
| Conv1d-6              | [-1, 250, 486]     | 250,250   |
| BatchNorm1d-7         | [-1, 250, 486]     | 500       |
| ReLU-8                | [-1, 250, 486]     | 0         |
| MaxPool1d-9           | [-1, 250, 243]     | 0         |
| Dropout-10            | [-1, 250, 243]     | 0         |
| Conv1d-11             | [-1, 500, 234]     | 1,250,500 |
| BatchNorm1d-12        | [-1, 500, 234]     | 1,000     |
| ReLU-13               | [-1, 500, 234]     | 0         |
| MaxPool1d-14          | [-1, 500, 117]     | 0         |
| Dropout-15            | [-1, 500, 117]     | 0         |
| Conv1d-16             | [-1, 1000, 108]    | 5,001,000 |
| BatchNorm1d-17        | [-1, 1000, 108]    | 2,000     |
| ReLU-18               | [-1, 1000, 108]    | 0         |
| MaxPool1d-19          | [-1, 1000, 54]     | 0         |
| AdaptiveAvgPool1d-20  | [-1, 1000, 1]      | 0         |
| Dropout-21            | [-1, 1000]         | 0         |
| Linear-22             | [-1, 5]            | 5,005     |
| Sigmoid-23            | [-1, 5]            | 0         |
||||
| Total params               |                    | 6,522,555   |
| Trainable params           |                    | 6,522,555   |
| Non-trainable params       |                    | 0           |
| Input size (MB)            |                    | 0.05        |
| Forward/backward pass size (MB) |             | 13.20       |
| Params size (MB)           |                    | 24.88       |
| Estimated Total Size       |                    | 38.13       |


# ROC Curve for the Knowledge Distillation Teacher Model on Test Data

![ROC Curve for the Knowledge Distillation Teacher Model on Test Data](https://user-images.githubusercontent.com/49470426/278029862-aead166f-ea26-42f2-8804-efb5546ba9c4.png)

*Caption*: ROC Curve for the Knowledge Distillation Teacher Model on Test Data

# Knowledge Distillation Student Model

| Layer              | Output Shape      | Param #    |
|------------------- |------------------- |----------- |
| Conv1d-1           | [-1, 50, 991]     | 6,050      |
| BatchNorm1d-2      | [-1, 50, 991]     | 100        |
| ReLU-3             | [-1, 50, 991]     | 0          |
| MaxPool1d-4        | [-1, 50, 495]     | 0          |
| Dropout-5          | [-1, 50, 495]     | 0          |
| Conv1d-6           | [-1, 150, 486]    | 75,150     |
| BatchNorm1d-7      | [-1, 150, 486]    | 300        |
| ReLU-8             | [-1, 150, 486]    | 0          |
| MaxPool1d-9        | [-1, 150, 243]    | 0          |
| Dropout-10         | [-1, 150, 243]    | 0          |
| Conv1d-11          | [-1, 300, 234]    | 450,300    |
| BatchNorm1d-12     | [-1, 300, 234]    | 600        |
| ReLU-13            | [-1, 300, 234]    | 0          |
| MaxPool1d-14       | [-1, 300, 117]    | 0          |
| AdaptiveAvgPool1d-15 | [-1, 300, 1]    | 0          |
| Dropout-16         | [-1, 300]         | 0          |
| Linear-17          | [-1, 5]           | 1,505      |
| Sigmoid-18         | [-1, 5]           | 0          |
||||
| Total params                  |                 | 6,522,555   |
| Trainable                     |                 | 6,522,555   |
| Non-trainable                 |                 | 0           |
| Input size (MB)               |                 | 0.05        |
| Forward/backward pass size (MB)|               | 13.20       |
| Params size (MB)              |                 | 24.88       |
| Estimated Total Size (MB)     |                 | 38.13       |


# ROC Curve for the Knowledge Distillation Student Model on Test Data

![ROC Curve for the Knowledge Distillation Student Model on Test Data](https://user-images.githubusercontent.com/49470426/278031917-f5a7bd0e-ee97-4996-94f6-4bb5492e7125.png)

*Caption*: ROC Curve for the Knowledge Distillation Student Model on Test Data

### 4. 소개 및 시연 영상

프로젝트 소개나 시연 영상을 넣으세요.

### 5. 사용법

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

