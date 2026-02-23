"""Multi-modal event classifier combining video and audio signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.analysis.audio_analyzer import AudioEvent
from src.analysis.ball_tracker import ShotEvent
from src.analysis.scene_detector import Scene
from src.config import EventConfig

logger = logging.getLogger(__name__)


@dataclass
class GameEvent:
    """A classified basketball game event."""

    event_type: str  # "made_shot", "three_pointer", "dunk", "fast_break", etc.
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    confidence: float  # fused confidence score 0-1
    video_confidence: float
    audio_confidence: float
    details: dict = field(default_factory=dict)


class EventClassifier:
    """Combines video analysis (shots, tracking) with audio analysis (crowd, whistles)
    to classify basketball game events and assign confidence scores.

    The fusion model weights video and audio signals independently,
    then combines them. Corroborating signals from both modalities
    boost overall confidence.
    """

    def __init__(self, config: EventConfig | None = None, fps: float = 29.97):
        self.config = config or EventConfig()
        self.fps = fps

    def classify(
        self,
        shot_events: list[ShotEvent],
        audio_events: list[AudioEvent],
        scenes: list[Scene],
    ) -> list[GameEvent]:
        """Run event classification on all analysis outputs.

        Args:
            shot_events: Detected shot attempts/makes from BallTracker.
            audio_events: Crowd excitement and whistle events from AudioAnalyzer.
            scenes: Scene boundaries from SceneDetector.

        Returns:
            List of classified GameEvent objects, filtered by min confidence.
        """
        events: list[GameEvent] = []

        # Classify shot events (video-primary)
        for shot in shot_events:
            event = self._classify_shot(shot, audio_events)
            if event and event.confidence >= self.config.min_confidence:
                events.append(event)

        # Find crowd excitement peaks not already covered by shot events
        standalone_crowd = self._find_standalone_crowd_events(audio_events, events)
        events.extend(standalone_crowd)

        # Sort by time and merge overlapping events
        events.sort(key=lambda e: e.start_sec)
        events = self._merge_nearby(events)

        logger.info("Classified %d game events", len(events))
        return events

    def _classify_shot(
        self, shot: ShotEvent, audio_events: list[AudioEvent]
    ) -> GameEvent | None:
        """Classify a shot event, boosting with audio context."""
        start_sec = shot.start_frame / self.fps
        end_sec = shot.end_frame / self.fps

        # Base video confidence from shot detection
        video_conf = 0.7 if shot.made else 0.5

        # Higher confidence for larger/cleaner arcs
        if shot.arc_height_px > 150:
            video_conf = min(1.0, video_conf + 0.1)

        # Boost for arcs whose descent lands near the hoop
        if shot.hoop_x_distance is not None and shot.hoop_x_distance < 50:
            video_conf = min(1.0, video_conf + 0.1)

        # Boost for clean descent (ball fell well past peak)
        if shot.descent_ratio is not None and shot.descent_ratio > 0.7:
            video_conf = min(1.0, video_conf + 0.05)

        # Check for corroborating audio
        audio_conf = self._find_audio_correlation(start_sec, end_sec, audio_events)

        # Determine event type
        if shot.made:
            event_type = "made_shot"
        else:
            event_type = "shot_attempt"

        # Fuse confidences
        fused = (
            self.config.video_weight * video_conf
            + self.config.audio_weight * audio_conf
        )

        details = {
            "arc_height_px": shot.arc_height_px,
            "ball_positions_count": len(shot.ball_positions),
        }
        if shot.hoop_x is not None:
            details["hoop_position"] = [shot.hoop_x, shot.hoop_y]

        return GameEvent(
            event_type=event_type,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
            confidence=fused,
            video_confidence=video_conf,
            audio_confidence=audio_conf,
            details=details,
        )

    def _find_audio_correlation(
        self,
        start_sec: float,
        end_sec: float,
        audio_events: list[AudioEvent],
    ) -> float:
        """Find the best audio correlation for a time window.

        Looks for crowd excitement or whistle events that overlap with
        or closely follow the given time window (within a few seconds,
        since crowd reaction lags the play slightly).
        """
        search_start = start_sec - 1.0
        search_end = end_sec + 4.0  # crowd reaction can lag by a few seconds

        best_score = 0.0
        for ae in audio_events:
            if ae.end_sec < search_start or ae.start_sec > search_end:
                continue

            if ae.event_type == "crowd_excitement":
                best_score = max(best_score, ae.score)
            elif ae.event_type == "whistle":
                # Whistle after a made shot is a strong signal
                if ae.start_sec >= start_sec:
                    best_score = max(best_score, ae.score * 0.8)

        return best_score

    def _find_standalone_crowd_events(
        self,
        audio_events: list[AudioEvent],
        existing_events: list[GameEvent],
    ) -> list[GameEvent]:
        """Find high crowd excitement not already explained by shot events.

        These become generic "crowd_excitement" events — useful for plays
        that video analysis might miss (steals, blocks, fast breaks
        without shots, etc.).
        """
        standalone = []

        for ae in audio_events:
            if ae.event_type != "crowd_excitement":
                continue
            if ae.score < self.config.min_confidence:
                continue

            # Check if this time window is already covered
            covered = False
            for ev in existing_events:
                if self._windows_overlap(ae.start_sec, ae.end_sec, ev.start_sec - 2, ev.end_sec + 2):
                    covered = True
                    break

            if not covered:
                start_frame = int(ae.start_sec * self.fps)
                end_frame = int(ae.end_sec * self.fps)
                standalone.append(
                    GameEvent(
                        event_type="crowd_excitement",
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_sec=ae.start_sec,
                        end_sec=ae.end_sec,
                        confidence=ae.score,
                        video_confidence=0.0,
                        audio_confidence=ae.score,
                        details={"source": "audio_only"},
                    )
                )

        return standalone

    def _merge_nearby(self, events: list[GameEvent]) -> list[GameEvent]:
        """Merge events that are very close together in time."""
        if not events:
            return events

        merged = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            gap = ev.start_sec - prev.end_sec
            if gap <= self.config.merge_gap_sec and ev.event_type == prev.event_type:
                # Merge: extend previous event, keep higher confidence
                merged[-1] = GameEvent(
                    event_type=prev.event_type,
                    start_frame=prev.start_frame,
                    end_frame=ev.end_frame,
                    start_sec=prev.start_sec,
                    end_sec=ev.end_sec,
                    confidence=max(prev.confidence, ev.confidence),
                    video_confidence=max(prev.video_confidence, ev.video_confidence),
                    audio_confidence=max(prev.audio_confidence, ev.audio_confidence),
                    details={**prev.details, **ev.details},
                )
            else:
                merged.append(ev)

        return merged

    @staticmethod
    def _windows_overlap(s1: float, e1: float, s2: float, e2: float) -> bool:
        return s1 < e2 and s2 < e1
