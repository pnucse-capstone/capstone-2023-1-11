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

프로젝트 결과물의 개괄적인 동작을 파악할 수 있는 이미지와 글을 작성하세요.

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

