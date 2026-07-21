from pathlib import Path

import pytest
import torch

from motionlab.data.beat2 import Beat2Dataset, collate, emotion_from_name, speaker_from_name

ROOT = Path("data/beat2/beat_english_v2.0.0")

pytestmark = pytest.mark.skipif(
    not (ROOT / "smplxflame_30").is_dir(), reason="BEAT2 not downloaded yet")


def test_emotion_mapping():
    assert emotion_from_name("2_scott_0_103_103") == 6      # fear range
    assert emotion_from_name("2_scott_0_1_1") == 0          # neutral
    assert emotion_from_name("2_scott_0_65_65") == 1        # happiness
    assert emotion_from_name("garbage") == 0
    assert speaker_from_name("21_ayana_0_1_1") == 21


def test_windows_shapes():
    ds = Beat2Dataset(str(ROOT), window=64, stride=64, split="", require_textgrid=False, max_files=3)
    assert len(ds) > 0
    item = ds[0]
    assert item.poses.shape == (64, 165)
    assert item.trans.shape == (64, 3)
    assert torch.isfinite(item.poses).all()
    assert item.poses.abs().max() < 3.2  # axis-angle sane range


def test_collate_and_words():
    ds = Beat2Dataset(str(ROOT), window=64, stride=64, split="", require_textgrid=True, max_files=5)
    if len(ds) == 0:
        pytest.skip("no textgrid-paired files downloaded yet")
    batch = collate([ds[0], ds[min(1, len(ds) - 1)]])
    assert batch["poses"].shape == (2, 64, 165)
    assert batch["emotion"].shape == (2,)
    # words are (start_frame, end_frame, str) within window bounds
    for words in batch["words"]:
        for s, e, w in words:
            assert 0 <= s <= 64 and 0 <= e <= 64 and isinstance(w, str)
