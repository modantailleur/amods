import librosa
import torch
from denoiser import pretrained


class FbDM:
    """Wraps Facebook's denoiser (dns64 pretrained model)."""

    def __init__(self, sr):
        """Load the dns64 pretrained model; ``sr`` is the sample rate of the audio ``predict`` will receive."""
        self.model = pretrained.dns64().cpu()
        self.sr = sr

    def predict(self, x):
        """Denoise ``x`` (mono float32), resampling to/from the model's own sample rate as needed."""
        noisy = x.copy()
        if self.sr != self.model.sample_rate:
            noisy = librosa.resample(x, orig_sr=self.sr, target_sr=self.model.sample_rate)

        with torch.no_grad():
            noisy = torch.from_numpy(noisy).float().unsqueeze(0)

            enhanced = self.model(noisy[None])
            enhanced = enhanced.squeeze().numpy()
            if self.sr != self.model.sample_rate:
                enhanced = librosa.resample(enhanced, orig_sr=self.model.sample_rate, target_sr=self.sr)

        return enhanced
