import torch
import soundfile as sf
from audiomentations import LowPassFilter, HighPassFilter, PitchShift, Gain, BandStopFilter, AddBackgroundNoise, LoudnessNormalization, ApplyImpulseResponse, SevenBandParametricEQ 
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import warnings
import torchaudio

def display_original(text, PATH):
    audio_np, sr = sf.read(PATH, dtype="float32")
    audio = torch.tensor(audio_np)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.T
    print(text)
    display(Audio(audio.numpy(), rate=sr))


def display_audio(text, audio, sr):
    print(text)
    # audio has shape (channels, samples)
    display(Audio(audio.numpy(), rate=sr))


def low_pass_filter(first_audio, PATH, prob, prev_taudio, prev_sr, min_freq, max_freq):
    if (first_audio):
        # if first audio, then read audio from file path
        # prev_taudio and prev_sr are not used
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        mono = audio.mean(dim=0)
    else:
        # in this case, we will have a prev_taudio and prev_sr
        # taudio currently has shape (channels, samples)
        mono = prev_taudio.mean(dim=0) # convert to mono
        sr = prev_sr
    augment = LowPassFilter(
            p=prob,
            min_cutoff_freq=min_freq,
            max_cutoff_freq=max_freq
        )
    augmented = augment(mono.numpy(), sample_rate=sr)
    taudio = torch.tensor(augmented).unsqueeze(0)
    return taudio, sr
    
def high_pass_filter(first_audio, PATH, prob, prev_taudio, prev_sr, min_freq, max_freq):
    if (first_audio):
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T 
        mono = audio.mean(dim=0)
    else:
        mono = prev_taudio.mean(dim=0)
        sr = prev_sr
    augment = HighPassFilter(
        p=prob,
        min_cutoff_freq=min_freq,
        max_cutoff_freq=max_freq
    )
    augmented = augment(mono.numpy(), sample_rate=sr)
    taudio = torch.tensor(augmented).unsqueeze(0)
    return taudio, sr


def add_bandstop_filter(first_audio, PATH, prob, prev_taudio, prev_sr, min_freq, max_freq, min_bf, max_bf):
    if (first_audio):
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        mono = audio.mean(dim=0)
    else:
        mono = prev_taudio.mean(dim=0)
        sr = prev_sr
    augment = BandStopFilter(
        min_center_freq = min_freq,
        max_center_freq = max_freq,
        min_bandwidth_fraction = min_bf,
        max_bandwidth_fraction = max_bf,
        p=prob,
    )
    augmented = augment(mono.numpy(), sample_rate=sr)
    taudio = torch.tensor(augmented).unsqueeze(0)
    return taudio, sr

def pitch_shift(first_audio, PATH, prob, prev_taudio, prev_sr, min_semitones, max_semitones):
    if (first_audio):
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        mono = audio.mean(dim=0)
    else:
        mono = prev_taudio.mean(dim=0)
        sr = prev_sr
    augment = PitchShift(
        min_semitones=min_semitones,
        max_semitones=max_semitones,
        p=prob
    )
    augmented = augment(mono.numpy(), sample_rate=sr)
    taudio = torch.tensor(augmented).unsqueeze(0)
    return taudio, sr

def add_gain(first_audio, PATH, prob, prev_taudio, prev_sr, min_gain, max_gain,):
    if (first_audio):
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        mono = audio.mean(dim=0)
    else:
        mono = prev_taudio.mean(dim=0)
        sr = prev_sr
    augment = Gain(
        min_gain_db=min_gain,
        max_gain_db=max_gain,
        p=prob,
    )
    augmented = augment(mono.numpy(), sample_rate=sr)
    taudio = torch.tensor(augmented).unsqueeze(0)
    return taudio, sr


def add_background(first_audio, PATH, prob, sounds_path, prev_taudio, prev_sr, min_snr, max_snr):
    if (first_audio):
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        mono = audio.mean(dim=0)
    else:
        mono = prev_taudio.mean(dim=0)
        sr = prev_sr
    augment = AddBackgroundNoise(
        sounds_path=sounds_path,
        min_snr_db=min_snr,
        max_snr_db=max_snr,
        p=prob,
    )
    augmented_mono = augment(
        samples=mono.numpy(),
        sample_rate=sr
    )
    taudio = torch.tensor(augmented_mono).unsqueeze(0)
    return taudio, sr

def apply_ir(first_audio, PATH, prob, ir_path, prev_taudio, prev_sr):
    if (first_audio):
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        mono = audio.mean(dim=0)
    else:
        mono = prev_taudio.mean(dim=0)
        sr = prev_sr
    augment = ApplyImpulseResponse(
        ir_path=ir_path,
        p=prob
    )
    augmented_mono = augment(
        samples=mono.numpy(),
        sample_rate=sr
    )
    taudio = torch.tensor(augmented_mono).unsqueeze(0)
    return taudio, sr

def apply_seven_band_peq(first_audio, PATH, prob, prev_taudio, prev_sr, min_gain, max_gain):
    if (first_audio):
        audio_np, sr = sf.read(PATH, dtype="float32")
        audio = torch.tensor(audio_np)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        mono = audio.mean(dim=0)
    else:
        mono = prev_taudio.mean(dim=0)
        sr = prev_sr
    augment = SevenBandParametricEQ(
        min_gain_db=-12.0,
        max_gain_db=12.0,
        p=prob,
    )
    augmented_mono = augment(
        samples=mono.numpy(),
        sample_rate=sr
    )
    taudio = torch.tensor(augmented_mono).unsqueeze(0)
    return taudio, sr

def plot_waveform(waveform, sr, title, ax=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr
    ax.plot(time_axis, waveform[0], linewidth=1, color="#2488B2")
    ax.grid(True)
    ax.set_xlim(0, time_axis[-1])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return plt

def plot_spectrogram(waveform, sr, title, min_db, max_db, ax=None):
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    spectrogram = spectrogram_transform(waveform)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bands (Hz)")
    pos = ax.imshow(
        spec_db[0].numpy(), 
        origin="lower", 
        aspect="auto", 
        cmap='magma',
        vmin = min_db, 
        vmax = max_db)
    plt.colorbar(pos, ax=ax, format="%+2.0f dB")
    return plt

def plot_all(audio, taudio, sr, og_title_prefix, augmented_title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_waveform(audio, sr, f"{og_title_prefix} Waveform", ax=axs[0,0])
    plot_waveform(taudio, sr, f"{augmented_title_prefix} Waveform", ax=axs[0,1])
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    spec = spectrogram_transform(audio)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    tspec = spectrogram_transform(taudio)
    tspec_db = torchaudio.transforms.AmplitudeToDB()(tspec)
    min_db = min(spec_db.min().item(), tspec_db.min().item())
    max_db = max(spec_db.max().item(), tspec_db.max().item())
    
    plot_spectrogram(audio, sr, f"{og_title_prefix} Spectrogram", ax=axs[1,0],
                     min_db=min_db, max_db=max_db)
    plot_spectrogram(taudio, sr, f"{augmented_title_prefix} Spectrogram", ax=axs[1,1],
                     min_db=min_db, max_db=max_db)
    plt.tight_layout()
    plt.show()
