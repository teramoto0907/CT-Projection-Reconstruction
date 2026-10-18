"""CT teaching demo: acquisition and animated filtered/unfiltered backprojection."""
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon, radon, resize

SIZE = 256
DELAY = 0.015
FILTERS = {
    "なし (Unfiltered)": None,
    "Ram-Lak (Ramp)": "ramp",
    "Shepp-Logan": "shepp-logan",
    "Cosine": "cosine",
    "Hamming": "hamming",
    "Hann": "hann",
}

st.set_page_config(page_title="CT投影・画像再構成デモ", layout="wide")
st.title("CT投影・画像再構成デモ")


@st.cache_data(show_spinner=False)
def phantom_image():
    return resize(shepp_logan_phantom(), (SIZE, SIZE),
                  anti_aliasing=True, preserve_range=True)


@st.cache_data(show_spinner=False)
def full_sinogram():
    # The full scan has one angular sample per degree (0 through 359).
    angles = np.arange(360, dtype=float)
    return radon(phantom_image(), theta=angles, circle=True,
                 preserve_range=True)


def selected_indices(n):
    # Evenly select integer-degree views from the existing 0..179-degree data.
    return np.floor(np.arange(n) * 180 / n).astype(int)


@st.cache_data(show_spinner=False)
def final_reconstruction(n, filter_name):
    indices = selected_indices(n)
    sino = full_sinogram()[:, indices]
    return iradon(sino, theta=indices.astype(float), filter_name=filter_name,
                  circle=True, output_size=SIZE, preserve_range=True)


def plot_phantom(angle=None):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(phantom_image(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if angle is not None:
        a = np.deg2rad(angle + 90)
        c = (SIZE - 1) / 2
        r0, r1 = SIZE * 0.43, SIZE * 0.52
        ax.annotate("", xy=(c+r0*np.cos(a), c-r0*np.sin(a)),
                    xytext=(c+r1*np.cos(a), c-r1*np.sin(a)),
                    arrowprops={"arrowstyle": "->", "color": "red", "lw": 2.5})
    fig.tight_layout(pad=0)
    return fig


def plot_sinogram(selected=None):
    full = full_sinogram()
    if selected is None:
        frame = full
    else:
        frame = np.full_like(full, np.nan, dtype=float)
        frame[:, selected] = full[:, selected]
    cmap = plt.colormaps["gray"].copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(frame, cmap=cmap, aspect="auto", origin="lower",
              extent=(0, 360, -full.shape[0]/2, full.shape[0]/2),
              vmin=0, vmax=float(full.max()))
    ax.set_xlabel("Projection angle [deg]")
    ax.set_ylabel("Detector position [pixel]")
    fig.tight_layout()
    return fig


def plot_reconstruction(image, vmax):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def filter_frequency_response(filter_name):
    """Frequency response matching scikit-image's iradon filter construction."""
    # iradon pads a SIZE-element detector profile to at least twice its size.
    size = max(64, 2 ** int(np.ceil(np.log2(2 * SIZE))))
    odd = np.concatenate((
        np.arange(1, size // 2 + 1, 2, dtype=int),
        np.arange(size // 2 - 1, 0, -2, dtype=int),
    ))
    impulse = np.zeros(size)
    impulse[0] = 0.25
    impulse[1::2] = -1 / (np.pi * odd) ** 2
    response = 2 * np.real(np.fft.fft(impulse))

    if filter_name == "shepp-logan":
        omega = np.pi * np.fft.fftfreq(size)[1:]
        response[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        response *= np.fft.fftshift(np.sin(freq))
    elif filter_name == "hamming":
        response *= np.fft.fftshift(np.hamming(size))
    elif filter_name == "hann":
        response *= np.fft.fftshift(np.hanning(size))
    elif filter_name is None:
        response[:] = 1.0

    frequencies = np.fft.rfftfreq(size)
    return frequencies, response[:size // 2 + 1]


def plot_filter_response(filter_name):
    frequencies, response = filter_frequency_response(filter_name)
    ref_frequencies, ref_response = filter_frequency_response("ramp")
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(ref_frequencies, ref_response, linestyle="--", color="gray", linewidth=1.2, alpha=0.7, label="Ram-Lak (reference)")
    ax.plot(frequencies, response, color="tab:blue", lw=2, label="Selected filter")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Spatial frequency [cycles/pixel]")
    ax.set_ylabel("Frequency response")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def show(area, fig):
    area.pyplot(fig, use_container_width=True)
    plt.close(fig)


with st.sidebar:
    st.subheader("投影処理")
    n_acquisition = st.slider("アニメーションの投影数", 1, 360, 180)
    st.subheader("画像再構成")
    n_reconstruction = st.slider("再構成に使用する投影数", 1, 180, 60)
    filter_label = st.selectbox("再構成フィルタ", list(FILTERS.keys()), index=1)
    filter_name = FILTERS[filter_label]

if "acquisition_angle" not in st.session_state:
    st.session_state.acquisition_angle = None
if "acquisition_indices" not in st.session_state:
    st.session_state.acquisition_indices = []
if "recon_image" not in st.session_state:
    st.session_state.recon_image = None
if "recon_settings" not in st.session_state:
    st.session_state.recon_settings = None

left, middle, right = st.columns(3)
with left:
    st.subheader("Shepp–Loganファントム")
    phantom_area = st.empty()
with middle:
    st.subheader("サイノグラム")
    sinogram_area = st.empty()
with right:
    st.subheader("再構成画像")
    reconstruction_area = st.empty()

show(phantom_area, plot_phantom(st.session_state.acquisition_angle))
show(sinogram_area, plot_sinogram(st.session_state.acquisition_indices))

with right:
    st.subheader("再構成フィルタの周波数応答")
    filter_fig = plot_filter_response(filter_name)
    st.pyplot(filter_fig, use_container_width=True)
    plt.close(filter_fig)

settings = (n_reconstruction, filter_name)
if st.session_state.recon_settings == settings and st.session_state.recon_image is not None:
    final = final_reconstruction(*settings)
    vmax = max(float(final.max()), 1e-6)
    show(reconstruction_area, plot_reconstruction(st.session_state.recon_image, vmax))
else:
    reconstruction_area.info("「画像再構成開始」を押してください。")

b1, b2 = st.columns(2)
with b1:
    run_acquisition = st.button("投影処理開始", use_container_width=True)
with b2:
    run_reconstruction = st.button("画像再構成開始", use_container_width=True)

if run_acquisition:
    chosen = np.floor(np.arange(n_acquisition) * 360 / n_acquisition).astype(int)
    for i, angle in enumerate(chosen):
        show(phantom_area, plot_phantom(float(angle)))
        show(sinogram_area, plot_sinogram(chosen[:i + 1]))
        time.sleep(DELAY)
    st.session_state.acquisition_angle = float(chosen[-1])
    st.session_state.acquisition_indices = list(range(360))
    show(sinogram_area, plot_sinogram())

if run_reconstruction:
    indices = selected_indices(n_reconstruction)
    sino = full_sinogram()
    final = final_reconstruction(n_reconstruction, filter_name)
    vmax = max(float(final.max()), 1e-6)
    accumulated = np.zeros((SIZE, SIZE), dtype=float)
    for i, degree in enumerate(indices, start=1):
        # A single-view iradon is the contribution of that angle, scaled
        # by 1/n. Its sum reproduces the full iradon result, including
        # the same Fourier filter and normalization.
        show(phantom_area, plot_phantom(float(degree)))
        one = iradon(sino[:, degree:degree + 1], theta=np.array([float(degree)]),
                     filter_name=filter_name, circle=True,
                     output_size=SIZE, preserve_range=True)
        accumulated += one / n_reconstruction
        show(reconstruction_area, plot_reconstruction(accumulated, vmax))
        time.sleep(DELAY)
    # Ensure final display is identical to the batch reconstruction.
    show(phantom_area, plot_phantom(180.0))
    show(reconstruction_area, plot_reconstruction(final, vmax))
    st.session_state.acquisition_angle = 180.0
    st.session_state.recon_image = final
    st.session_state.recon_settings = settings
