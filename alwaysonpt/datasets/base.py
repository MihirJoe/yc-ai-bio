"""
BioSignalRecord — unified abstraction for any physiological signal recording.
Works across EMG, gait VGRF, gait timing, ECG, and RR interval data.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BioSignalRecord:
    record_id: str
    domain: str                           # 'gait_vgrf', 'gait_timing', 'ecg_12lead', 'ecg_2ch', 'rr_interval', 'emg'
    signals: dict                         # channel_name -> np.ndarray
    fs: int                               # sampling rate (Hz), 0 for event-based (RR intervals)
    duration_s: float
    metadata: dict = field(default_factory=dict)
    ground_truth: dict = field(default_factory=dict)

    @classmethod
    def from_emg_segment(cls, segment) -> 'BioSignalRecord':
        """Convert an EMGSegment dataclass into a BioSignalRecord."""
        signals = {'emg': segment.emg}
        if hasattr(segment, 'gonio') and segment.gonio is not None:
            signals['gonio'] = segment.gonio

        return cls(
            record_id=f"S{segment.subject_id}_{segment.motion_class}",
            domain='emg',
            signals=signals,
            fs=segment.fs,
            duration_s=segment.duration_s,
            metadata={'subject_id': segment.subject_id},
            ground_truth={'motion_class': segment.motion_class},
        )

    def channel_names(self) -> list:
        return list(self.signals.keys())

    def primary_signal(self) -> np.ndarray:
        """Return the first signal channel."""
        return next(iter(self.signals.values()))

    def summary(self) -> str:
        channels = ", ".join(
            f"{k}({len(v)} samples)" for k, v in self.signals.items()
        )
        return (
            f"BioSignalRecord[{self.record_id}] "
            f"domain={self.domain}, fs={self.fs}Hz, "
            f"duration={self.duration_s:.1f}s, "
            f"channels=[{channels}]"
        )
