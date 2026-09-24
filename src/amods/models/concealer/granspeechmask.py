import math
import threading
from collections import deque

import numpy as np

from ...audio import apply_fade
from ..adversary import ReverseAM
from ..denoiser import FbDM
from .base import GranSpeechMaskCM, register_concealer


@register_concealer("granspeechmask")
class GranSpeechMask(GranSpeechMaskCM):
    """
    Conceals speech by continuously recording short clips of the speaker's
    own recent voice into memory (denoised, faded, and optionally played
    reversed), then, once enough clips have accumulated, playing back
    whichever one's spectral fingerprint best matches what's being said right
    now.
    """

    def __init__(self, sr, config_path, config):
        """See the ``concealer`` config component for every parameter this reads (``config`` keys below)."""
        super().__init__(sr, config_path, config)

        self.config_path = config_path
        # Parameters
        # maximum number of concealers to store in memory
        self.memory_maxlen = config["memory_maxlen"]
        # number of buffers before possible to reuse same concealer
        self.max_countdown_reuse = config["max_countdown_reuse"]
        # distance (in number of concealers) from the end of memory to consider for concealing
        # i.e., do not use the last N concealers added to memory. This avoids reusing too recent concealers.
        self.max_concealer_distance_to_buffer = config["max_concealer_distance_to_buffer"]
        # duration in seconds of each concealer
        self.concealer_duration = config["concealer_duration"]
        # Cooldown between concealing events, randomly sampled between these two
        # bounds - each expressed as a multiple of one concealer clip's own
        # duration (not literal seconds), e.g. 0.5 means half a clip's length.
        self.concealing_min_timeout_ratio = config["concealing_min_timeout_ratio"]
        self.concealing_max_timeout_ratio = config["concealing_max_timeout_ratio"]
        # duration in seconds of the fade applied to each concealer
        self.fade_duration = config["fade_duration"]
        self.is_stream = config["is_stream"]
        self.pending_conc_max_size = config["pending_conc_max_size"]
        self.freeze_learning = config["freeze_learning"]
        self.max_len_strat = config.get("max_len_strat", "queue")  # "queue" or "diversity_marginal_gain"
        self.distance_type = config.get("distance_type", "cosine")  # "p-correlation", "cosine"
        self.decision_win = config.get("decision_win", 0.3)  # seconds; sliding window of context used to match the live buffer against memory

        self.sr = sr
        self.config = config
        self.denoise = config.get("denoise", True)

        if self.denoise:
            self.denoiser = FbDM(sr=sr)

        self.random_reverse = config.get("random_reverse", True)
        if self.random_reverse:
            self.am = ReverseAM()

        # Derived sizes
        self.concealer_size = int(sr * self.concealer_duration)
        self.fade_size = int(sr * self.fade_duration)

        # Used internally
        self.audio_accum = deque()
        self.feature_accum = deque()
        self.pending_voice = deque()

        # For marginal gain
        self.cross_corr = None

        self.pending_voice_max_duration = 2  # Keep up to 2s of pending voice for concealer generation
        self.n_concealer_before_denoise = math.ceil(self.pending_voice_max_duration / self.concealer_duration)
        self.cur_size_before_update_memory = 0
        self.stop_processing = False
        self.cur_concealing = False
        self.concealing_countdown = 0  # samples
        # In streaming mode, pending_voice is a rolling window that's never
        # cleared after a feed (unlike file mode - see _feed_memory), so it
        # sits at its ~2s cap continuously once full. Without this counter,
        # update_memory's own "is there 2s pending" check would stay true
        # forever after the first feed, re-triggering _feed_memory again
        # the instant the previous background feed finishes - on an
        # almost-unchanged window, barely shifted by however little audio
        # arrived in between. That flooded memory with near-duplicate
        # clips, and (worse) let concealing clips selected close together
        # end up with very different absolute levels, since each is
        # independently RMS-matched to pending_voice at its own selection
        # moment - audible as clicking/pumping, especially at high gain.
        # Requiring a full new pending_voice_max_duration worth of audio
        # since the last feed forces genuinely fresh material each time.
        self.new_voice_since_feed = 0

    def update(self, x):
        """While speech is active: extend the pending-voice window with ``x`` and count down any active concealing cooldown."""
        self.pending_voice.extend(x)
        while len(self.pending_voice) > int(self.pending_voice_max_duration * self.sr):
            self.pending_voice.popleft()
        self.new_voice_since_feed += len(x)

        if self.cur_concealing:
            self.concealing_countdown -= len(x)
            if self.concealing_countdown <= 0:
                self.cur_concealing = False
                self.concealing_countdown = 0

    def refresh(self, x):
        """During silence: keep extending the pending-voice window with ``x``, but drop any in-progress concealing state."""
        self.pending_voice.extend(x)
        while len(self.pending_voice) > int(self.pending_voice_max_duration * self.sr):
            self.pending_voice.popleft()
        self.audio_accum.clear()
        self.feature_accum.clear()
        self.cur_concealing = False
        self.concealing_countdown = 0

    def update_memory(self, x):
        """Age existing memory entries' cooldowns every ``concealer_duration`` worth of audio, and, once enough pending voice has accumulated, feed it into memory (in the background if streaming live, inline otherwise)."""
        self.cur_size_before_update_memory += len(x)

        if self.cur_size_before_update_memory >= self.concealer_size:
            self.memory = [(concealer, concealer_feat, max(0, cooldown - 1), voice_name)
                           for concealer, concealer_feat, cooldown, voice_name in self.memory]
            self.cur_size_before_update_memory = 0

        if not self.freeze_learning:
            pending_voice_full = len(self.pending_voice) >= int(self.pending_voice_max_duration * self.sr)
            # File mode already gets a fresh window for free (pending_voice
            # is cleared after every feed there - see _feed_memory), so
            # "full" alone is a correct, sufficient trigger. Streaming mode
            # needs the extra new_voice_since_feed check (see __init__).
            has_fresh_material = (
                not self.is_stream or self.new_voice_since_feed >= int(self.pending_voice_max_duration * self.sr)
            )
            if (not self.stop_processing) and pending_voice_full and has_fresh_material:
                self.stop_processing = True
                if self.is_stream:
                    bg_thread = threading.Thread(target=self._background_task, daemon=True)
                    bg_thread.start()
                else:
                    self._background_task()

    def _background_task(self):
        """Feed the currently pending voice into memory; run on a daemon thread when streaming live."""
        self._feed_memory(self.pending_voice)

    def _feed_memory(self, y, voice_name=None):
        """
        Split ``y`` into ``concealer_duration``-sized clips, denoise them (if
        enabled), fade their edges, optionally reverse them, extract each
        clip's features from its non-denoised original, keep only the clips
        the concealer's own VAD still calls speech, and merge the result into
        memory per ``max_len_strat`` ("queue": drop oldest past
        ``memory_maxlen``; "diversity_marginal_gain": keep the most diverse
        set, see :meth:`GranSpeechMaskCM._compute_swap_marginal_gain`).
        """
        new_memory = []

        pending_audio = np.array(y)

        if self.denoise:
            pending_audio_denoised = self.denoiser.predict(pending_audio)
        else:
            pending_audio_denoised = pending_audio

        if len(pending_audio_denoised) > len(pending_audio):
            pending_audio_denoised = pending_audio_denoised[:len(pending_audio)]
        elif len(pending_audio_denoised) < len(pending_audio):
            pending_audio_denoised = np.pad(pending_audio_denoised, (0, len(pending_audio) - len(pending_audio_denoised)), mode='constant')

        new_concealers = np.array_split(pending_audio_denoised, self.n_concealer_before_denoise)
        # Same split, on the original (non-denoised) audio: pending_audio and
        # pending_audio_denoised are the same length (see padding/truncation
        # above), so array_split produces identical segment boundaries for
        # both - this just gives each denoised chunk its original counterpart.
        new_concealers_original = np.array_split(pending_audio, self.n_concealer_before_denoise)

        fade_size = int(self.sr * self.fade_duration)
        fade_size = fade_size if fade_size > self.sr // 100 else self.sr // 100  # min 10ms of fade

        for concealer, concealer_original in zip(new_concealers, new_concealers_original):
            concealer = apply_fade(concealer, fade_size)
            if self.random_reverse:
                concealer = self.am.predict(concealer)
            # Features come from the original (non-denoised) segment, not the
            # denoised/faded/reversed one played back below: the live query
            # side (pending_voice in get_concealer) is raw mic audio that's
            # never denoised, so matching against denoised candidate features
            # would compare across two different audio domains.
            concealer_feat = self._feature_extractor(concealer_original, norm=True)
            if self.vad.predict(concealer):
                new_memory.append((concealer.astype(np.float32, copy=False), concealer_feat.astype(np.float32, copy=False), 0, voice_name if voice_name is not None else ""))

        # Fill memory up to max length
        while len(self.memory) < self.memory_maxlen and len(new_memory) > 0:
            self.memory.append(new_memory.pop(0))

        if self.max_len_strat == "queue":
            self.memory = self.memory + new_memory
            # Enforce max length manually (drop oldest)
            while len(self.memory) > self.memory_maxlen:
                self.memory.pop(0)

        elif self.max_len_strat == "diversity_marginal_gain":
            # Decide to keep new concealers based on greedy marginal gain in diversity (log-Mel frequencies correlation)
            if len(new_memory) > 0:
                if self.cross_corr is None:
                    if self.distance_type == "p-correlation":
                        self.cross_corr = self._compute_correlation_matrix(self.memory)  # shape: (k, k)
                    if self.distance_type == "cosine":
                        self.cross_corr = self._compute_cosine_similarity_matrix(self.memory)  # shape: (k, k)
                for e in new_memory:
                    self.memory, self.cross_corr = self._compute_swap_marginal_gain(self.memory, self.cross_corr, e, distance_type=self.distance_type)

        if not self.is_stream:
            self.pending_voice.clear()
        self.new_voice_since_feed = 0
        self.stop_processing = False

    def learn(self, y, voice_name=None):
        """
        Pre-seed memory offline from a batch of audio ``y`` (e.g. before a
        live session starts), splitting it into ``pending_voice_max_duration``
        chunks and feeding each one through :meth:`_feed_memory` in turn,
        tagged with ``voice_name`` if given.
        """
        pending_audio = np.array(y)
        chunk_size = int(self.pending_voice_max_duration * self.sr)

        for i in range(0, len(pending_audio), chunk_size):
            chunk = pending_audio[i:i + chunk_size]
            if len(chunk) == chunk_size:
                self._feed_memory(chunk, voice_name=voice_name)

    def copy(self):
        """Return an independent GranSpeechMask with the same config and memory clips, but reset cooldowns."""
        new_cm = GranSpeechMask(self.sr, self.config_path, self.config)
        new_cm.memory = [(concealer, concealer_feat, 0, voice_name)
                         for concealer, concealer_feat, cooldown, voice_name in self.memory]
        return new_cm
