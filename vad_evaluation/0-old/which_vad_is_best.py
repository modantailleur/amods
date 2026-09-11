import sys
from pathlib import Path

# Add parent directories to path to import vad_models
sys.path.insert(0, str(Path(__file__).parent.parent))

from joblib import Parallel, delayed
from vad_models import WebrtcVAD, SileroVAD, TenVAD
import numpy as np
import os
import librosa

# =========================
# CONFIG
# =========================

WINDOW_MS = 50
sr = 16000

# DATASET_DIR = '/media/user/EXTbackup/speech_dataset/OfficeSpeech-EBR-ASR/'
# OUT_DIR = "/media/user/MT-SSD-3/0-PROJETS_INFO/Post-doc/vadConcealerExp/"

DATASET_DIR = '../OfficeSpeech-EBR-ASR/'
OUT_DIR = "../vadConcealerExp/"
os.makedirs(OUT_DIR, exist_ok=True)

sub_datasets = ['ebr-high-asr-high', 'ebr-mid-asr-mid', 'ebr-low-asr-low']

speakers_list = [
    'p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266',
    'p267','p268','p269','p270','p271','p272','p273','p274','p275','p276',
    'p277','p278','p279','p280','p281','p282','p283','p284','p285','p286',
    'p287','p288','p292','p293','p294','p295','p297','p298','p299','p300',
    'p301','p302','p303','p304','p305','p306','p307','p308','p310','p311',
    'p312','p313','p314','p315','p316','p317','p318','p323','p326','p329',
    'p330','p333','p334','p335','p336','p339','p340','p341','p343','p345',
    'p347','p351','p360','p361','p362','p363','p364','p374','p376','s5'
]

vad_models_names = ["webrtc1", "webrtc2", "webrtc3", "silero", "ten"]
thresholds = np.arange(0.1, 1.0, 0.1)

# =========================
# BUILD VAD CONFIGS
# =========================

vad_configs = [
    (vad_name, ths)
    for vad_name in vad_models_names
    for ths in thresholds
]

# =========================
# VAD FACTORY
# =========================

def build_vad(vad_name, ths):
    if "webrtc" in vad_name:
        return WebrtcVAD(
            sr=sr,
            logit_threshold=ths,
            aggressiveness=int(vad_name[-1])
        )
    elif vad_name == "silero":
        return SileroVAD(sr=sr, logit_threshold=ths)
    elif vad_name == "ten":
        return TenVAD(sr=sr, logit_threshold=ths)
    else:
        raise ValueError(f"Unknown VAD: {vad_name}")

# =========================
# FILE PROCESSING
# =========================

def process_file(file_path, file_name, sub_dataset, split):
    try:
        # ---- Load audio ONCE ----
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        audio = audio.astype(np.float32)

        # ---- Frame once ----
        win_len = int(round(sr * WINDOW_MS / 1000.0))
        n_frames = int(np.ceil(len(audio) / win_len))
        pad_len = n_frames * win_len - len(audio)

        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len), mode="constant")

        frames = audio.reshape(n_frames, win_len)

        # ---- Build all VADs once per file ----
        vads = [
            (vad_name, ths, build_vad(vad_name, ths))
            for vad_name, ths in vad_configs
        ]

        # ---- Run all configs ----
        for vad_name, ths, vad in vads:
            vad_decisions = np.array([vad.predict(frame) for frame in frames])

            out_file = os.path.join(
                OUT_DIR,
                f"{file_name}_{vad_name}_th{ths:.1f}_{sub_dataset}.npy"
            )
            np.save(out_file, vad_decisions)

        print(f"[OK] {file_name}")

    except Exception as e:
        print(f"[ERROR] {file_name}: {e}")

# =========================
# BUILD FILE LIST
# =========================

def build_file_list():
    all_files = []

    splits = ["evaluation", "calibration"]

    for sub_dataset in sub_datasets:
        for split in splits:
            for speaker in speakers_list:

                full_path = f"{DATASET_DIR}/{sub_dataset}/{split}/pan_0/{speaker}"

                if not os.path.exists(full_path):
                    continue

                for file in os.listdir(full_path):
                    if file.endswith(".flac") and "mic1" in file:
                        file_path = os.path.join(full_path, file)

                        # include split in metadata to avoid ambiguity
                        all_files.append((file_path, file, sub_dataset, split))

    return all_files

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    all_files = build_file_list()
    print(f"Total files: {len(all_files)}")

    Parallel(n_jobs=50, backend="loky")(
        delayed(process_file)(file_path, file_name, sub_dataset, split)
        for file_path, file_name, sub_dataset, split in all_files
    )