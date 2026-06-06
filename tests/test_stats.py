"""SessionStats のテスト（純粋ロジック）。"""

from __future__ import annotations

from gaze import ContactState
from stats import SessionStats


def test_face_lost_pauses_visible_timer() -> None:
    stats = SessionStats()
    stats.update(1.0, ContactState.CONTACT)
    stats.update(1.0, None)  # 顔ロスト中は維持率の分母に入れない
    assert stats.total_time == 2.0
    assert stats.visible_time == 1.0
    assert stats.contact_time == 1.0
    assert stats.current_streak == 0.0  # 顔ロストでキープは途切れる


def test_contact_ratio() -> None:
    stats = SessionStats()
    stats.update(1.0, ContactState.CONTACT)
    stats.update(1.0, ContactState.CONTACT)
    stats.update(2.0, ContactState.NO_CONTACT, drift="down")
    assert stats.contact_ratio == 0.5  # 2s / 4s


def test_longest_streak_tracks_max() -> None:
    stats = SessionStats()
    stats.update(1.0, ContactState.CONTACT)
    stats.update(1.0, ContactState.CONTACT)  # 連続 2s
    stats.update(1.0, ContactState.NO_CONTACT, drift="down")  # 途切れ
    stats.update(1.0, ContactState.CONTACT)  # 連続 1s
    assert stats.longest_streak == 2.0
    assert stats.current_streak == 1.0


def test_dominant_drift() -> None:
    stats = SessionStats()
    stats.update(3.0, ContactState.NO_CONTACT, drift="down")
    stats.update(1.0, ContactState.NO_CONTACT, drift="left")
    assert stats.dominant_drift() == "down"


def test_dominant_drift_none_when_no_drift() -> None:
    stats = SessionStats()
    stats.update(1.0, ContactState.CONTACT)
    assert stats.dominant_drift() is None


def test_reset_clears_everything() -> None:
    stats = SessionStats()
    stats.update(1.0, ContactState.CONTACT)
    stats.update(1.0, ContactState.NO_CONTACT, drift="down")
    stats.reset()
    assert stats.total_time == 0.0
    assert stats.visible_time == 0.0
    assert stats.contact_time == 0.0
    assert stats.longest_streak == 0.0
    assert all(v == 0.0 for v in stats.drift_time.values())


def test_summary_lines_includes_ratio() -> None:
    stats = SessionStats()
    stats.update(1.0, ContactState.CONTACT)
    lines = stats.summary_lines()
    assert any("維持率" in line for line in lines)
