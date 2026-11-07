"""Drop-in replacement for audio/stft.py (the original breaks on librosa>=0.10)."""
import torch
from librosa.filters import mel as librosa_mel_fn


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, n_mel_channels,
                 sampling_rate, mel_fmin, mel_fmax):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels,
                                   fmin=mel_fmin, fmax=mel_fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("window", torch.hann_window(win_length))

    def mel_spectrogram(self, y):
        """y: (B, T) waveform in [-1, 1] -> log-mel (B, n_mels, frames), energy (B, frames)."""
        spec = torch.stft(y, self.filter_length, self.hop_length, self.win_length,
                          window=self.window, center=True, pad_mode="reflect", return_complex=True)
        magnitudes = spec.abs()
        mel = torch.matmul(self.mel_basis, magnitudes)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        energy = torch.norm(magnitudes, dim=1)
        return mel, energy
