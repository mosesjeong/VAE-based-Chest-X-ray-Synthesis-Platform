import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# ----------------------------------------------------------------------
# 1. 페이지 설정 및 기본 구성
# ----------------------------------------------------------------------

# Streamlit 페이지의 제목, 아이콘, 레이아웃 등 기본 설정을 지정합니다.
st.set_page_config(
    page_title="VAE & Diffusion Image Generator",
    page_icon="🎨",
    layout="wide",
)

# ----------------------------------------------------------------------
# 2. 모델 아키텍처 정의
# 학습 시 사용된 모델과 동일한 구조로 정의해야 가중치를 불러올 수 있습니다.
# ----------------------------------------------------------------------

# --- 하이퍼파라미터 (학습 당시와 동일해야 함) ---
IMG_SIZE = 64
LATENT_DIM = 512
ENCODER_CHANNELS = [64, 128, 256, 512]
DECODER_CHANNELS = [256, 128, 64, 32]
DENOISER_CHANNELS = [512, 256, 128, 64]


def build_encoder(latent_dim, channels):
    """이미지를 잠재 벡터로 압축하는 인코더 모델을 빌드합니다."""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for ch in channels:
        x = layers.Conv2D(ch, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    mean = layers.Dense(latent_dim, name="mean")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    return keras.Model(inputs, [mean, log_var], name="encoder")


def build_decoder(latent_dim, channels):
    """잠재 벡터를 이미지로 복원하는 디코더 모델을 빌드합니다."""
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * channels[0])(inputs)
    x = layers.Reshape((4, 4, channels[0]))(x)
    for ch in channels:
        x = layers.Conv2DTranspose(ch, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    outputs = layers.Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
    return keras.Model(inputs, outputs, name="decoder")


def build_denoiser(latent_dim, channels):
    """노이즈가 낀 잠재 벡터에서 노이즈를 예측하는 디노이저 모델을 빌드합니다."""
    latent_input = layers.Input(shape=(latent_dim,))
    time_input = layers.Input(shape=(1,))
    time_embedding = layers.Dense(latent_dim, activation='swish')(time_input)
    x = layers.Add()([latent_input, time_embedding])
    x = layers.Dense(latent_dim, activation='swish')(x)
    for ch in channels:
        x_res = x
        x = layers.Dense(ch, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        if x_res.shape[-1] != ch:
            x_res = layers.Dense(ch)(x_res)
        x = layers.Add()([x, x_res])
    output = layers.Dense(latent_dim)(x)
    return keras.Model([latent_input, time_input], output, name="denoiser")


class VAE(keras.Model):
    """인코더와 디코더를 결합한 VAE 모델 클래스입니다."""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


class DiffusionModel(keras.Model):
    """디노이저를 포함하는 Diffusion 모델 클래스입니다."""

    def __init__(self, denoiser, **kwargs):
        super().__init__(**kwargs)
        self.denoiser = denoiser


# ----------------------------------------------------------------------
# 3. 모델 로딩 및 핵심 기능 함수
# ----------------------------------------------------------------------

@st.cache_resource
def load_models():
    """미리 학습된 모델의 가중치를 로드합니다.
    @st.cache_resource를 사용하여 앱 실행 중 한 번만 로드되도록 합니다."""
    # 모델 구조 빌드
    encoder = build_encoder(LATENT_DIM, ENCODER_CHANNELS)
    decoder = build_decoder(LATENT_DIM, DECODER_CHANNELS)
    denoiser = build_denoiser(LATENT_DIM, DENOISER_CHANNELS)

    # VAE 및 Diffusion 모델 인스턴스 생성
    vae_model = VAE(encoder, decoder)
    diffusion_model = DiffusionModel(denoiser)

    # 저장된 가중치 로드
    # 사용자는 여기에 실제 가중치 파일 경로를 지정해야 합니다.
    vae_weights_path = 'vae_epoch_30.h5'
    diffusion_weights_path = 'diffusion_epoch_30.h5'

    if os.path.exists(vae_weights_path) and os.path.exists(diffusion_weights_path):
        # build a dummy input to initialize model weights
        dummy_img = tf.random.normal([1, IMG_SIZE, IMG_SIZE, 3])
        vae_model(dummy_img)
        dummy_latent = tf.random.normal([1, LATENT_DIM])
        dummy_time = tf.constant([0.5], dtype=tf.float32)
        diffusion_model([dummy_latent, dummy_time])

        # 가중치 로드
        vae_model.load_weights(vae_weights_path)
        diffusion_model.load_weights(diffusion_weights_path)
        print("모델 가중치 로딩 완료.")
    else:
        # 가중치 파일이 없을 경우 경고 메시지 표시
        st.error(f"모델 가중치 파일을 찾을 수 없습니다: '{vae_weights_path}', '{diffusion_weights_path}'")
        return None, None

    return vae_model, diffusion_model


def get_random_latent_vector(num_images):
    """지정된 개수만큼 무작위 잠재 벡터(시드)를 생성합니다."""
    return tf.random.normal(shape=(num_images, LATENT_DIM))


def generate_images(num_images, steps, latent_vector_seed):
    """잠재 벡터 시드로부터 이미지를 생성하는 전체 프로세스를 수행합니다."""
    vae, diffusion = load_models()
    if vae is None:
        return []

    # 입력된 시드를 현재 잠재 벡터로 설정
    current_latents = latent_vector_seed

    # Diffusion의 역방향 프로세스 (노이즈 제거 과정)
    # 총 `steps`에 걸쳐 점진적으로 노이즈를 제거합니다.
    for t_step in np.linspace(1.0, 0.0, steps):
        t = tf.constant([t_step] * num_images, dtype=tf.float32)
        predicted_noise = diffusion([current_latents, t])
        # 노이즈 제거 강도를 스텝 수에 따라 조절
        alpha = 1.0 / steps
        current_latents -= alpha * predicted_noise

    # 최종적으로 노이즈가 제거된 잠재 벡터를 디코더에 통과시켜 이미지 생성
    generated_images = vae.decoder(current_latents)
    return generated_images


# ----------------------------------------------------------------------
# 4. Streamlit UI 구성
# ----------------------------------------------------------------------

st.title("🎨 VAE-Diffusion 이미지 생성기")
st.markdown("""
이 웹 앱은 **VAE(Variational Autoencoder)**와 **Diffusion Model**을 결합하여 새로운 이미지를 생성합니다.
사이드바에서 생성할 이미지의 수, 노이즈 제거 단계(Steps), 그리고 생성 시드(Seed)를 조절하여 다양한 이미지를 만들어보세요.
""")

# --- 사이드바 컨트롤 ---
with st.sidebar:
    st.header("⚙️ 컨트롤 패널")

    # 생성할 이미지 개수 조절 슬라이더
    num_images_to_gen = st.slider("생성할 이미지 개수", 1, 16, 4)

    # 노이즈 제거 단계 조절 슬라이더
    steps = st.slider("생성 스텝 (Steps)", 10, 100, 50)

    st.markdown("---")
    st.subheader("🌱 생성 시드 (Seed)")

    # session_state를 사용하여 시드 값을 여러 인터랙션에 걸쳐 유지
    if 'latent_seed' not in st.session_state:
        st.session_state.latent_seed = get_random_latent_vector(num_images_to_gen)

    # 새로운 무작위 시드 생성 버튼
    if st.button("새로운 랜덤 시드 생성"):
        st.session_state.latent_seed = get_random_latent_vector(num_images_to_gen)
        st.success("새로운 랜덤 시드가 생성되었습니다!")

    # 사용자가 시드 값을 텍스트로 직접 입력
    seed_input = st.text_area("또는 시드 값을 직접 입력하세요 (쉼표로 구분)",
                              value=", ".join(map(str, st.session_state.latent_seed[0, :5].numpy())) + ", ...")

# --- 메인 화면 ---
st.markdown("---")

# 이미지 생성을 트리거하는 메인 버튼
if st.button("🖼️ 이미지 생성하기 (GENERATE)"):
    # 입력된 시드 값 파싱 (오류 발생 시 무작위 시드 사용)
    try:
        # 사용자가 입력한 텍스트 시드를 다시 텐서로 변환
        seed_values = [float(x.strip()) for x in seed_input.split(',') if x.strip()]
        if len(seed_values) == LATENT_DIM:
            current_seed = tf.constant([seed_values] * num_images_to_gen, dtype=tf.float32)
        else:
            # 사용자가 일부만 입력했거나 잘못 입력한 경우, session_state의 시드를 사용
            current_seed = st.session_state.latent_seed
            st.warning(f"입력된 시드 값이 유효하지 않아 기존 시드를 사용합니다. {LATENT_DIM}개의 숫자가 필요합니다.")
    except:
        current_seed = st.session_state.latent_seed
        st.error("시드 값 파싱 중 오류가 발생했습니다. 기존 시드를 사용합니다.")

    # 로딩 스피너와 함께 이미지 생성 함수 호출
    with st.spinner('이미지를 생성하는 중입니다... 잠시만 기다려주세요.'):
        generated_images = generate_images(num_images_to_gen, steps, current_seed)

    # 생성된 이미지를 화면에 표시
    if len(generated_images) > 0:
        st.success("이미지 생성이 완료되었습니다!")
        # 생성된 이미지를 여러 열에 걸쳐 보기 좋게 표시
        cols = st.columns(4)
        for i, img_tensor in enumerate(generated_images):
            # 텐서 값을 [0, 1] 범위의 numpy 배열로 변환
            img_array = (img_tensor.numpy() + 1) / 2.0
            cols[i % 4].image(img_array, caption=f"Image {i + 1}", use_column_width=True)
    else:
        st.error("이미지 생성에 실패했습니다. 모델 가중치 파일이 올바른지 확인하세요.")