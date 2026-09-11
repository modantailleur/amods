"""
The contract a concealer algorithm must satisfy to plug into :class:`amods.stream.Stream`,
plus a registry so new algorithms can be added without editing a dispatcher.

To add a new algorithm:

1. Write a class implementing :class:`ConcealerModel` (constructor signature
   ``__init__(self, sr, config_path, config)``).
2. Decorate it with ``@register_concealer("your_type_name")``.
3. Import that module somewhere reachable at startup (see
   ``amods.models.concealer.__init__``) so the decorator runs.
4. Set ``concealer_type: your_type_name`` in a concealer config YAML.

If your algorithm is itself memory-based (accumulate candidate clips, VAD
each one, pick the best match for the live buffer), subclass
:class:`GranSpeechMaskCM` instead of :class:`ConcealerModel` directly and you
only need to implement ``update``, ``refresh``, and ``update_memory`` -
``get_concealer`` and the memory bookkeeping are already there. See
:mod:`amods.models.concealer.granspeechmask` for an example.
"""
from abc import ABC, abstractmethod

import numpy as np

from ...config import resolve_config
from ..vad import select_vad_model


class ConcealerModel(ABC):
    """
    Base contract every concealer algorithm must implement. Consumed by
    :class:`amods.stream.Stream` once per input chunk.
    """

    @abstractmethod
    def get_concealer(self, x: np.ndarray) -> tuple:
        """
        Called every chunk while the source VAD detects voice activity.

        Returns ``(audio, label)``: ``audio`` is added into a pending output
        buffer (it may be longer than one chunk - the caller accumulates it
        across future chunks), ``label`` is a free-form string describing
        what's playing (empty string if concealing with silence).
        """

    @abstractmethod
    def refresh(self, x: np.ndarray) -> None:
        """Called every chunk while the source VAD detects silence."""

    @abstractmethod
    def copy(self) -> "ConcealerModel":
        """Return an independent copy carrying the algorithm's runtime state."""

    def status(self) -> dict:
        """
        Optional progress info for UIs (see ``amods.gui``'s status panel) that
        don't want to know this algorithm's internals. Override to report
        something more useful; the default says "always ready, nothing to
        report".

        Keys: ``ready`` (bool); ``progress``/``target`` (int or None, e.g. a
        memory's current/max size - for a general "how full is it" display);
        ``threshold`` (int or None, the ``progress`` value at which ``ready``
        flips to True - may differ from ``target``, e.g. a memory that keeps
        filling up to a much larger cap after concealing has already started);
        ``label`` (str, e.g. "Memory").
        """
        return {"ready": True, "progress": None, "target": None, "threshold": None, "label": ""}


def register_concealer(type_name):
    """Class decorator registering a :class:`ConcealerModel` under ``type_name``."""
    def _decorator(cls):
        _REGISTRY[type_name] = cls
        return cls
    return _decorator


_REGISTRY = {}


def select_concealer_model(sr, config_path, config):
    """Instantiate the concealer registered under ``config["concealer_type"]``."""
    type_name = config["concealer_type"]
    try:
        cls = _REGISTRY[type_name]
    except KeyError:
        raise ValueError(f"Unknown concealer model config name: {type_name}")
    return cls(sr, config_path, config)


class VadCM:
    """Mixin handling VAD-related setup for concealer candidates."""

    def __init__(self, sr, config_path, config):
        """Resolve and build the VAD named by ``config["concealer_vad_config"]``."""
        self.vad_config_name = config["concealer_vad_config"]
        self.vad_config = resolve_config("vad", self.vad_config_name, search_dir=config_path)
        self.vad = select_vad_model(self.vad_config, sr)


class GranSpeechMaskCM(VadCM, ConcealerModel):
    """
    Partial :class:`ConcealerModel` for algorithms that accumulate a memory
    of candidate clips and, once it's large enough, play back whichever
    candidate best matches the live buffer. Subclasses implement
    ``update_memory`` (grow/age the memory) and ``update``/``refresh``
    (per-chunk bookkeeping); this class provides the memory storage,
    matching, save/load, and the ``get_concealer``/``status`` contract.
    """

    def __init__(self, sr, config_path, config):
        """Set up the (initially empty) memory; subclasses add their own parameters on top."""
        super().__init__(sr, config_path, config)
        self.memory = []
        # Below this many candidates, the memory is too repetitive/small to
        # conceal from without it being obviously the same few clips on loop.
        self.min_memory_to_conceal = config.get("min_memory_to_conceal", 20)

    @abstractmethod
    def update_memory(self, x):
        """Grow/age the memory with this chunk's audio; called every chunk regardless of voice activity."""

    @abstractmethod
    def update(self, x):
        """Per-chunk bookkeeping (e.g. tracking recent audio, cooldowns) run while there IS voice activity."""

    def status(self) -> dict:
        """Report memory fill level and whether it's large enough to start concealing (see :meth:`ConcealerModel.status`)."""
        return {
            "ready": len(self.memory) >= self.min_memory_to_conceal,
            "progress": len(self.memory),
            "target": self.memory_maxlen,
            "threshold": self.min_memory_to_conceal,
            "label": "Memory",
        }

    def _feature_extractor(self, x, norm=False):
        """
        Reduce ``x`` to a log-mel spectral fingerprint (mean over time, one
        value per mel band), used to compare a memory clip against the live
        buffer. Set ``norm=True`` to peak-normalize ``x`` first, so loudness
        differences don't affect the match.
        """
        import librosa

        if norm:
            x_copy = x.copy() / (np.max(np.abs(x)) + 1e-10)
        else:
            x_copy = x

        mel = librosa.feature.melspectrogram(
            y=x_copy,
            sr=self.sr,
            n_mels=32,
            fmin=100,
            fmax=6000
        )

        logmel = librosa.power_to_db(mel, ref=1.0)
        return logmel.mean(axis=1)

    def get_concealer(self, x):
        """
        Grow the memory with ``x``, then, if not already mid-concealment and
        the memory is large enough, pick the clip whose features best match
        the recent live buffer (falling back to a random eligible clip if
        none are eligible for a distance comparison), put it on cooldown, and
        return it energy-normalized to match the live buffer. Otherwise
        returns silence. See :meth:`ConcealerModel.get_concealer` for the
        general contract.
        """
        self.update_memory(x)
        self.update(x)  # extends pending_voice with x, so it's included below

        should_conceal = (not self.cur_concealing) and (len(self.memory) >= self.min_memory_to_conceal)

        if should_conceal:
            self.cur_concealing = True

            # The decision of which concealer matches the current buffer isn't
            # limited to the raw buffer x: it's made over the trailing
            # decision_win seconds of audio (buffer x plus whatever context
            # precedes it), a window that slides forward by one buffer every
            # callback. pending_voice already holds x (see self.update(x)
            # above) plus prior history, so slicing its tail gives exactly
            # that sliding window.
            decision_win_samples = int(self.sr * self.decision_win)
            decision_audio = np.array(self.pending_voice)[-decision_win_samples:]
            x_feat = self._feature_extractor(decision_audio, norm=True)

            # Avoid reusing same concealer too often using cooldown
            length_condition = max(0, len(self.memory) - self.max_concealer_distance_to_buffer)

            if self.distance_type == "p-correlation":
                mel_distances = [
                    1 - np.corrcoef(x_feat, clip_feat)[0, 1] if cooldown == 0 and i < length_condition else np.inf
                    for i, (_, clip_feat, cooldown, _) in enumerate(self.memory)
                ]
            elif self.distance_type == "cosine":
                x_feat_norm = np.linalg.norm(x_feat)
                mel_distances = [
                    1 - (np.dot(x_feat, clip_feat) / (x_feat_norm * (np.linalg.norm(clip_feat) + 1e-10))) if cooldown == 0 and i < length_condition else np.inf
                    for i, (_, clip_feat, cooldown, _) in enumerate(self.memory)
                ]
            else:
                raise ValueError(f"Unknown distance_type: {self.distance_type}")

            min_distance = np.inf
            best_idx = 0
            for i, distance in enumerate(mel_distances):
                if distance < min_distance:
                    min_distance = distance
                    best_idx = i
            if min_distance == np.inf:
                random_idx_range = len(self.memory) - self.max_concealer_distance_to_buffer \
                    if len(self.memory) - self.max_concealer_distance_to_buffer >= 0 else len(self.memory)
                best_idx = int(np.random.randint(0, random_idx_range)) if random_idx_range > 0 else 0

            # Set chosen concealer cooldown
            concealer, concealer_feat, _, voice_name = self.memory[best_idx]
            self.memory[best_idx] = (concealer, concealer_feat, self.max_countdown_reuse, voice_name)

            # Normalize energy to match input buffer
            concealer = concealer * (np.sqrt(np.mean(np.array(self.pending_voice)**2)) / np.sqrt(np.mean(concealer**2)))

            concealing_min_timeout_size = int(len(concealer) * self.concealing_min_timeout_ratio)
            concealing_max_timeout_size = int(len(concealer) * self.concealing_max_timeout_ratio)
            self.concealing_countdown = np.random.randint(concealing_min_timeout_size, concealing_max_timeout_size) \
                if concealing_max_timeout_size > concealing_min_timeout_size else \
                concealing_min_timeout_size
        else:
            # Return silence
            concealer = np.zeros(self.pending_conc_max_size, dtype=np.float32)
            voice_name = ""
        return concealer, voice_name

    def save_memory(self, save_path):
        """Save the memory's clips, features, and voice names to an ``.npz`` file at ``save_path``."""
        concealers = [c for c, _, _, _ in self.memory]
        features = [f for _, f, _, _ in self.memory]
        names = [name for _, _, _, name in self.memory]

        # Built element-by-element rather than np.array(concealers,
        # dtype=object): when every concealer clip happens to have the same
        # length, np.array() silently collapses them into a single 2D array
        # instead of an array of 1D float32 arrays, losing the dtype.
        concealers_obj = np.empty(len(concealers), dtype=object)
        for i, c in enumerate(concealers):
            concealers_obj[i] = c

        np.savez(
            save_path,
            concealers=concealers_obj,
            features=np.array(features),
            names=np.array(names)
        )

    def load_memory(self, load_path):
        """Replace the memory with clips/features/voice names loaded from an ``.npz`` file saved by :meth:`save_memory`."""
        data = np.load(load_path, allow_pickle=True)

        concealers = data["concealers"]
        features = data["features"]
        names = data["names"]

        self.memory = [
            (c, f, 0, str(name))
            for c, f, name in zip(concealers, features, names)
        ]

    def get_avg_memory_features(self):
        """Return the mean feature vector across all memory clips, or None if the memory is empty."""
        if len(self.memory) == 0:
            return None
        features = np.stack([item[1] for item in self.memory])  # shape: (k, d)
        return np.mean(features, axis=0)  # shape: (d,)

    def _compute_swap_marginal_gain(self, memory, cross_corr, new_elem, distance_type="cosine"):
        """
        Greedily decide whether ``new_elem`` should replace an existing memory
        entry to increase overall diversity (used by the ``"diversity_marginal_gain"``
        ``max_len_strat``): find the entry whose removal, combined with adding
        ``new_elem``, most increases total pairwise distance, and swap it in
        only if that gain is positive. Returns the (possibly unchanged) memory
        list and its updated pairwise-distance matrix ``cross_corr``.
        """
        n = len(memory)

        if distance_type == "p-correlation":
            c_new = 1 - np.array([
                np.corrcoef(new_elem[1], memory[j][1])[0, 1]
                for j in range(n)
            ])
        elif distance_type == "cosine":
            c_new = 1 - np.array([
                np.dot(new_elem[1], memory[j][1]) / (np.linalg.norm(new_elem[1]) * (np.linalg.norm(memory[j][1]) + 1e-10))
                for j in range(n)
            ])
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}")

        # --- precompute row-wise absolute sums (fast reuse) ---
        row_sums = np.sum(cross_corr, axis=1)

        gains = np.zeros(n)
        for i in range(n):
            remove = row_sums[i]  # all correlations involving i
            # Add new element contribution (excluding interaction with i, since i is removed)
            add = np.sum(c_new) - c_new[i]
            gains[i] = -remove + add

        new_memory = memory.copy()
        best_remove = np.argmax(gains)

        if gains[best_remove] > 0:
            # Remove selected element and append new one at the end
            new_memory.pop(best_remove)
            new_memory.append(new_elem)

            mask = np.ones(n, dtype=bool)
            mask[best_remove] = False
            reduced_corr = cross_corr[mask][:, mask]  # (n-1, n-1)
            c_new_reduced = np.delete(c_new, best_remove)  # (n-1,)

            new_cross_corr = np.zeros((n, n), dtype=np.float32)
            new_cross_corr[:-1, :-1] = reduced_corr
            new_cross_corr[-1, :-1] = c_new_reduced
            new_cross_corr[:-1, -1] = c_new_reduced
            new_cross_corr[-1, -1] = 0.0
        else:
            new_cross_corr = cross_corr

        return new_memory, new_cross_corr

    def _compute_correlation_matrix(self, memory):
        """Return the pairwise Pearson-distance matrix (``1 - correlation``) between all memory clips' features."""
        features = np.stack([item[1] for item in memory])  # shape: (k, d)
        return 1 - np.corrcoef(features)  # shape: (k, k)

    def _compute_cosine_similarity_matrix(self, memory):
        """Return the pairwise cosine-distance matrix (``1 - cosine similarity``) between all memory clips' features."""
        features = np.stack([item[1] for item in memory])  # shape: (k, d)
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
        normalized_features = features / norms
        return 1 - normalized_features @ normalized_features.T  # shape: (k, k)
