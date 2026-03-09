import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchaudio
import torchaudio.transforms as T

from pathlib import Path

class AudioCPUprocessor:
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(self, file_path, debug_mode=False):
        return self._process_audio_librosa(file_path, debug_mode)

    def _process_audio_librosa(self, file_path, debug_mode=False):
        # 1. 加载音频 (采样率设为 22050Hz 是业界平衡性能与质量的常用值)
        # Load audio (22050Hz is an industry standard for performance/quality balance)
        y, sr = librosa.load(file_path, sr=self.sr)

        # 2. 计算短时傅里叶变换 (STFT)
        # Compute STFT
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)

        # 3. 映射到梅尔刻度
        # Map to Mel scale
        mel_spec = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=self.n_mels)

        # 4. 对数压缩 (Log-scaling)
        # Log-scaling for better feature dynamic range
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # --- 可视化部分 / Visualization Section ---
        if debug_mode:
            plt.figure(figsize=(12, 8))

            # 子图1：原始波形 (Time Domain)
            # Subplot 1: Waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr, color='blue')
            plt.title('Original Waveform (Time Domain)')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')

            # 子图2：梅尔频谱 (Frequency Domain)
            # Subplot 2: Mel-Spectrogram
            plt.subplot(2, 1, 2)
            img = librosa.display.specshow(log_mel_spec, sr=sr, hop_length=self.hop_length,
                                           x_axis='time', y_axis='mel', cmap='magma')
            plt.colorbar(img, format='%+2.0f dB')
            plt.title('Log-Mel Spectrogram')

            plt.tight_layout()
            plt.show()

        return log_mel_spec  # Shape: (n_mels, Frames)

    def save(self, mel_spec, output_path):
        """Save a log-mel spectrogram array to a .npy file.

        Args:
            mel_spec (np.ndarray): Log-mel spectrogram, e.g. returned by process().
            output_path (str | Path): Path to the output .npy file.
        """
        np.save(output_path, mel_spec)

    def process_batch(self, txt_file):
        """Batch process audio files listed in a txt file.

        Args:
            txt_file (str | Path): Path to a txt file with one audio path per line.

        Returns:
            List[np.ndarray]: List of log-mel spectrogram arrays.
        """
        txt_file = Path(txt_file)
        base_dir = txt_file.parent
        raw = txt_file.read_text().strip().splitlines()
        paths = [(base_dir / p.strip()).resolve() for p in raw if p.strip()]
        specs = [self.process(p) for p in paths]
        return specs

    def save_batch(self, mel_specs, txt_file, output_dir, list_file):
        """Save a list of log-mel spectrograms to npy files and write a npy path list.

        Args:
            mel_specs (List[np.ndarray]): Log-mel spectrograms, e.g. from process_batch().
            txt_file (str | Path): The same txt file used in process_batch(), one audio path per line.
            output_dir (str | Path): Directory to save npy files.
            list_file (str | Path): Path to write the npy file list (one path per line).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_file = Path(txt_file)
        base_dir = txt_file.parent
        raw = txt_file.read_text().strip().splitlines()
        paths = [(base_dir / p.strip()).resolve() for p in raw if p.strip()]
        npy_paths = []
        for spec, p in zip(mel_specs, paths):
            out_path = output_dir / (Path(p).stem + ".npy")
            self.save(spec, out_path)
            npy_paths.append(str(out_path))
        Path(list_file).write_text("\n".join(npy_paths))


class AudioGPUprocessor(torch.nn.Module):
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128,
                 device=None):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = torch.device(device if device is not None
                                   else "cuda" if torch.cuda.is_available() else "cpu")

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.to(self.device)

    def load_from_path(self, file_path):
        """Load a single audio file, resample to target sr, and return as waveform tensor.

        Args:
            file_path (str | Path): Path to the audio file.

        Returns:
            Tensor: Shape (1, Channel, Time)
        """
        waveform, orig_sr = torchaudio.load(file_path)
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=self.sr)
        return waveform.unsqueeze(0).to(self.device)  # (1, Channel, Time)

    def load_from_path_batch(self, txt_file):
        """Load multiple audio files listed in a txt file and return a batched waveform tensor.
        Shorter waveforms are zero-padded to the length of the longest one.

        Args:
            txt_file (str | Path): Path to a txt file with one audio path per line.

        Returns:
            Tensor: Shape (Batch, Channel, Time)
        """
        paths = Path(txt_file).read_text().strip().splitlines()
        waveforms = [self.load_from_path(p.strip()) for p in paths if p.strip()]
        # each element: (1, Channel, Time_i), Time_i may differ
        max_time = max(w.shape[-1] for w in waveforms)
        padded = [
            torch.nn.functional.pad(w, (0, max_time - w.shape[-1]))
            for w in waveforms
        ]
        return torch.cat(padded, dim=0)  # (Batch, Channel, Time)

    def load_mel_spec(self, npy_path):
        """Load a pre-saved log-mel spectrogram from a .npy file.

        Args:
            npy_path (str | Path): Path to a .npy file saved by AudioCPUprocessor.save().

        Returns:
            Tensor: Shape (1, n_mels, Frames)
        """
        arr = np.load(npy_path)                              # (n_mels, Frames)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, n_mels, Frames)

    def load_mel_spec_batch(self, txt_file):
        """Load multiple pre-saved log-mel spectrograms listed in a txt file.
        Shorter spectrograms are zero-padded along the time axis.

        Args:
            txt_file (str | Path): Path to a txt file with one .npy path per line.

        Returns:
            Tensor: Shape (Batch, n_mels, Frames)
        """
        paths = Path(txt_file).read_text().strip().splitlines()
        specs = [self.load_mel_spec(p.strip()) for p in paths if p.strip()]
        # each element: (1, n_mels, Frames_i), Frames_i may differ
        max_frames = max(s.shape[-1] for s in specs)
        padded = [
            torch.nn.functional.pad(s, (0, max_frames - s.shape[-1]))
            for s in specs
        ]
        return torch.cat(padded, dim=0)  # (Batch, n_mels, Frames)

    def forward(self, waveform):
        """Convert a batched waveform tensor to log-mel spectrogram.

        Args:
            waveform (Tensor): Shape (Batch, Channel, Time)

        Returns:
            Tensor: Shape (Batch, n_mels, Frames)
        """
        # mix down to mono by averaging channels: (Batch, Channel, Time) -> (Batch, 1, Time)
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        mel_spec = self.mel_spectrogram(waveform)   # (Batch, 1, n_mels, Frames)
        log_mel = self.amplitude_to_db(mel_spec)
        return log_mel.squeeze(1)                   # (Batch, n_mels, Frames)