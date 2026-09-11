import librosa
import numpy as np
import torch
from silero_vad import load_silero_vad
from ten_vad import TenVad


def select_vad_model(config, sr=None):
    """Build the VAD backend named by ``config["vad_type"]``."""
    if config["vad_type"] == "rms":
        return RmsVAD(db_threshold=config["db_threshold"])
    elif config["vad_type"] == "webrtc":
        return WebrtcVAD(
            logit_threshold=config["logit_threshold"],
            aggressiveness=config["aggressiveness"],
            sr=sr
        )
    elif config["vad_type"] == "silero":
        return SileroVAD(
            logit_threshold=config["logit_threshold"],
            sr=sr
        )
    elif config["vad_type"] == "ten":
        return TenVAD(
            logit_threshold=config["logit_threshold"],
            sr=sr
        )
    elif config["vad_type"] == "none":
        return IdentityVAD()
    else:
        raise ValueError(f"Unknown vad model config name: {config['vad_type']}")


class RmsVAD:
    """Loudness-only VAD: speech is declared whenever the chunk's RMS level exceeds a dB threshold."""

    def __init__(self, db_threshold=-50.0):
        self.db_threshold = db_threshold
        self.rms_threshold = 10 ** (db_threshold / 20)  # Convert dB to linear scale

    def predict(self, x):
        """Return True if ``x``'s RMS level is above ``db_threshold``."""
        rms_value = np.sqrt(np.mean(x**2))
        return rms_value > self.rms_threshold


class WebrtcVAD:
    """Wraps Google's WebRTC VAD, run per 10ms frame over the chunk and averaged into a speech ratio."""

    def __init__(self, logit_threshold=None, aggressiveness=1, sr=16000):
        # Some issues with installing pyWebRTC VAD on servers without C++ build tools,
        # hence the import is inside the class. If import fails, consider using "silero" or "ten" VAD types instead.
        import webrtcvad

        self.model = webrtcvad.Vad(aggressiveness)
        self.sr = sr
        self.logit_threshold = logit_threshold
        self.pad_end = True  # Whether to pad the end of audio to fit frame size
        self.frame_duration = 0.010  # WebRTC VAD supports 10, 20, or 30 ms frames

    def predict(self, x):
        """
        x: np.ndarray of float audio in [-1, 1], shape (n,) or (n, ch)
        returns: bool (speech?) based on mean(frame_is_speech) > threshold
        """
        x = np.asarray(x)

        # If stereo/multichannel: downmix to mono
        if x.ndim == 2:
            x = x.mean(axis=1)

        # Handle empty input
        if x.size == 0:
            return False

        # Ensure float32 and clamp to [-1, 1] before int16 conversion
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32, copy=False)
        x = np.clip(x, -1.0, 1.0)

        # Convert to 16-bit PCM little-endian
        audio_int16 = (x * 32767.0).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Compute exact bytes per frame
        samples_per_frame = int(self.sr * self.frame_duration)  # 10 ms frames
        bytes_per_frame = samples_per_frame * 2  # int16 -> 2 bytes

        # If too short for one frame: either pad or return False
        if len(audio_bytes) < bytes_per_frame:
            if not self.pad_end:
                return False
            audio_bytes += b"\x00" * (bytes_per_frame - len(audio_bytes))

        # Pad end so that every frame is valid (optional but avoids tail issues)
        remainder = len(audio_bytes) % bytes_per_frame
        if remainder != 0:
            if self.pad_end:
                audio_bytes += b"\x00" * (bytes_per_frame - remainder)
            else:
                audio_bytes = audio_bytes[:len(audio_bytes) - remainder]

        # Run VAD per frame
        speech_flags = []
        for start in range(0, len(audio_bytes), bytes_per_frame):
            frame = audio_bytes[start:start + bytes_per_frame]
            speech_flags.append(self.model.is_speech(frame, self.sr))

        speech_ratio = float(np.mean(speech_flags)) if speech_flags else 0.0

        out = speech_ratio > self.logit_threshold if self.logit_threshold is not None else speech_ratio
        return out


class SileroVAD:
    """Wraps the Silero VAD neural model, run per 512-sample frame (at 16kHz) and averaged into a speech probability."""

    def __init__(self, logit_threshold=None, sr=16000):
        self.model = load_silero_vad()
        self.vad_sr = 16000  # Silero VAD fixed sampling rate
        self.sr = sr
        self.logit_threshold = logit_threshold
        self.pad_end = True  # Whether to pad the end of audio to fit frame size
        self.frame_size = 512

    def predict(self, x):
        """
        x: np.ndarray of float audio in [-1, 1], shape (n,) or (n, ch)
        returns: bool (speech?) based on mean(frame_is_speech) > threshold
        """
        x = np.asarray(x)

        # If stereo/multichannel: downmix to mono
        if x.ndim == 2:
            x = x.mean(axis=1)

        # Handle empty input
        if x.size == 0:
            return False

        # Ensure float32 and clamp to [-1, 1]
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32, copy=False)
        x = np.clip(x, -1.0, 1.0)

        if self.sr != self.vad_sr:
            x = librosa.resample(x, orig_sr=self.sr, target_sr=self.vad_sr)

        if not isinstance(x, torch.Tensor):
            x_torch = torch.from_numpy(x)
        else:
            x_torch = x

        # Convert to 16-bit PCM and split into chunks
        # Cut from beginning if not exact multiple of frame_size
        remainder = len(x_torch) % self.frame_size
        if remainder != 0:
            x_torch = x_torch[remainder:]
        chunks = [x_torch[i:i + self.frame_size] for i in range(0, len(x_torch), self.frame_size)]

        speech_probs = []
        with torch.no_grad():
            for chunk in chunks:
                speech_prob = self.model(chunk, self.vad_sr).item()
                speech_probs.append(speech_prob)

        speech_ratio = np.mean(speech_probs) if speech_probs else 0.0

        out = speech_ratio > self.logit_threshold if self.logit_threshold is not None else speech_ratio
        return out


class TenVAD:
    """
    TenVAD accessible at: https://github.com/TEN-framework/ten-vad/tree/main
    @misc{TEN VAD,
    author = {TEN Team},
    title = {TEN VAD: A Low-Latency, Lightweight and High-Performance Streaming Voice Activity Detector (VAD)},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {https://github.com/TEN-framework/ten-vad.git},
    email = {developer@ten.ai}
    }
    """
    def __init__(self, logit_threshold=None, sr=16000):
        """Load TenVAD; ``sr`` is the sample rate of the audio ``predict`` will receive."""
        self.frame_size = 256
        self.model = TenVad(hop_size=self.frame_size, threshold=logit_threshold if logit_threshold is not None else 0.5)
        self.vad_sr = 16000  # Silero VAD fixed sampling rate
        self.sr = sr
        self.logit_threshold = logit_threshold
        self.pad_end = True  # Whether to pad the end of audio to fit frame size

    def predict(self, x):
        """
        x: np.ndarray of float audio in [-1, 1], shape (n,) or (n, ch)
        returns: bool (speech?) based on mean(frame_is_speech) > threshold
        """
        x = np.asarray(x)

        # If stereo/multichannel: downmix to mono
        if x.ndim == 2:
            x = x.mean(axis=1)

        # Handle empty input
        if x.size == 0:
            return False

        # Ensure float32 and clamp to [-1, 1]
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32, copy=False)
        x = np.clip(x, -1.0, 1.0)

        if self.sr != self.vad_sr:
            x = librosa.resample(x, orig_sr=self.sr, target_sr=self.vad_sr)

        x = (x * 32767).astype('int16')

        # Convert to 16-bit PCM and split into chunks
        # Cut from beginning if not exact multiple of frame_size
        remainder = len(x) % self.frame_size
        if remainder != 0:
            x = x[remainder:]
        chunks = [x[i:i + self.frame_size] for i in range(0, len(x), self.frame_size)]

        speech_probs = []
        for chunk in chunks:
            # process() returns (speech_probability, out_flag); out_flag is a hard
            # speech/non-speech decision, but we only need the soft probability here.
            speech_probability, _ = self.model.process(chunk)
            speech_probs.append(speech_probability)

        speech_ratio = np.mean(speech_probs) if speech_probs else 0.0

        out = speech_ratio > self.logit_threshold if self.logit_threshold is not None else speech_ratio
        return out


class IdentityVAD:
    """No-op VAD used to test the pipeline without a real VAD: always speech."""

    def predict(self, x):
        """Always return True."""
        return True
