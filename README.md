# VAE-기반 흉부 X-ray 합성 플랫폼 🩺

이 프로젝트는 VAE(Variational Autoencoder) 모델을 사용하여 실제와 유사한 고품질의 흉부 X-ray 이미지를 합성하는 딥러닝 플랫폼입니다. 생성된 데이터는 의료 영상 연구, 데이터 증강(Data Augmentation) 또는 교육 목적으로 활용될 수 있으며, 학습된 모델을 통해 새로운 X-ray 이미지를 생성해볼 수 있는 웹 애플리케이션을 포함합니다.

---

## 📁 프로젝트 구조

이 프로젝트는 주로 두 가지 핵심 파일로 구성됩니다.

1.  **`train_chest_xray_vae.ipynb`**
    * **역할**: VAE 모델 학습용 Jupyter Notebook
    * **기능**:
        * 흉부 X-ray 이미지에 최적화된 VAE 모델의 아키텍처를 정의합니다.
        * NIH ChestX-ray, CheXpert 등의 공개 데이터셋을 불러와 모델을 학습시킵니다.
        * 재구성 손실(Reconstruction Loss)과 KL 발산(KL Divergence)을 최소화하여 학습을 진행합니다.
        * 학습 완료 후, 모델의 가중치(`encoder.h5`, `decoder.h5`)를 저장합니다.

2.  **`app_streamlit.py`**
    * **역할**: 모델 데모 및 이미지 생성용 Streamlit 웹 애플리케이션
    * **기능**:
        * 학습된 VAE 인코더와 디코더 모델 가중치를 불러옵니다.
        * 사용자가 잠재 공간(Latent Space) 벡터를 조절하거나 랜덤 시드를 통해 새로운 이미지를 생성할 수 있는 UI를 제공합니다.
        * 'X-ray 이미지 생성하기' 버튼을 클릭하면, 설정에 따라 새로운 흉부 X-ray 이미지를 실시간으로 생성하여 화면에 표시합니다.

---

## 🚀 실행 가이드

이 프로젝트는 **모델 학습**과 **웹 앱 실행**의 두 단계로 진행됩니다.

### **1단계: 모델 학습하기**

1.  **환경 설정**:
    * 이 리포지토리를 로컬 환경에 복제(clone)합니다.
      ```bash
      git clone https://github.com/mosesjeong/VAE-based-Chest-X-ray-Synthesis-Platform
      cd VAE-based-Chest-X-ray-Synthesis-Platform
      ```
    * 가상 환경을 생성한 후, `requirements.txt` 파일을 이용해 필요한 라이브러리를 설치합니다.
        ```bash
        pip install -r requirements.txt
        ```

2.  **데이터셋 준비**:
    * 공개 흉부 X-ray 데이터셋(예: [NIH ChestX-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC))을 다운로드합니다.
    * `train_chest_xray_vae.ipynb` 파일 내의 `DATASET_PATH` 변수에 다운로드한 이미지 데이터셋의 경로를 지정합니다.

3.  **학습 실행**:
    * Jupyter Notebook에서 `train_chest_xray_vae.ipynb` 파일을 열고, 전체 셀을 순서대로 실행하여 모델 학습을 시작합니다.
    * 학습이 완료되면 지정된 경로에 모델 가중치 파일(`.h5`)이 저장됩니다.

### **2단계: 웹 앱 실행하기**

1.  **학습된 모델 준비**:
    * 1단계에서 생성된 모델 가중치 파일들(`encoder.h5`, `decoder.h5`)을 `app_streamlit.py`가 있는 디렉토리로 옮기거나, 코드 내의 경로를 올바르게 수정합니다.

2.  **Streamlit 앱 실행**:
    * 터미널에서 아래 명령어를 입력하여 웹 애플리케이션을 실행합니다.
        ```bash
        streamlit run app_streamlit.py
        ```

3.  **이미지 생성**:
    * 웹 브라우저에 앱 페이지가 자동으로 열립니다.
    * UI 컨트롤을 사용하여 생성 조건을 설정한 후, **'X-ray 이미지 생성하기'** 버튼을 눌러 새로운 합성 이미지를 확인합니다.
