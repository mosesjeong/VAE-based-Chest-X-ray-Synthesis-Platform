# VAE-based-Chest-X-ray-Synthesis-Platform


# VAE-Diffusion 이미지 생성 프로젝트 🎨

이 프로젝트는 VAE(Variational Autoencoder)와 Diffusion 모델을 결합하여 고품질의 이미지를 생성하는 딥러닝 모델을 학습하고, 학습된 모델을 사용해볼 수 있는 대화형 웹 애플리케이션을 제공합니다.

## 📁 프로젝트 구조

이 프로젝트는 두 개의 핵심 파일로 구성됩니다.

1.  **`13th_VAE+Diffusion_model.ipynb`**
    -   **역할**: 모델 학습용 Jupyter Notebook
    -   **기능**:
        -   VAE와 Diffusion 모델의 아키텍처를 정의합니다.
        -   지정된 이미지 데이터셋을 불러와 모델을 학습시킵니다.
        -   L1 손실과 지각 손실(Perceptual Loss)을 함께 사용하여 이미지 품질을 향상시킵니다.
        -   학습 과정에서 생성된 샘플 이미지와 최종 모델의 가중치(`.h5` 파일)를 저장합니다.

2.  **`250623_MS_VAE_streamlitweb_13thmodel_v2.py`**
    -   **역할**: 모델 데모용 Streamlit 웹 애플리케이션
    -   **기능**:
        -   위 Notebook에서 학습된 모델 가중치를 불러옵니다.
        -   사용자가 이미지 개수, 생성 스텝, 시드(seed) 등을 조절할 수 있는 대화형 UI를 제공합니다.
        -   '이미지 생성하기' 버튼을 누르면 설정값에 따라 실시간으로 새로운 이미지를 생성하여 보여줍니다.

## 🚀 실행 가이드

이 프로젝트는 **모델 학습**과 **웹 앱 실행**의 두 단계로 진행됩니다.

### **1단계: 모델 학습하기**

1.  **환경 설정**:
    -   이 리포지토리를 복제(clone)합니다.
    -   가상 환경을 생성하고 `requirements.txt` 파일을 이용해 필요한 라이브러리를 설치합니다.
        ```bash
        pip install -r requirements.txt
        ```

2.  **데이터셋 준비**:
    -   `13th_VAE+Diffusion_model.ipynb` 파일 내의 `DATASET_PATH` 변수에 자신의 이미지 데이터셋 경로를 지정합니다. (예: `"./img_align_celeba/*.jpg"`)

3.  **학습 실행**:
    -   Jupyter Notebook에서 `13th_VAE+Diffusion
