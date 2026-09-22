"""
Phase 2A: Feature Engineering Foundation

Converts stored desktop activities and sessions into numerical feature vectors
suitable for unsupervised learning (clustering) and later supervised learning (classification).

Design principles:
- Features capture behavioral context, not raw text
- Window titles are NOT used as features (privacy, high-cardinality)
- All features are deterministic and explainable
- Features are extracted from application sequences and temporal patterns
- No scikit-learn dependencies in this module
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence
import math


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """A complete set of numerical features for a session."""

    session_id: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    metadata: dict

    def __post_init__(self) -> None:
        """Validate that names and values are aligned."""
        if len(self.feature_names) != len(self.feature_values):
            raise ValueError(
                f"Feature count mismatch: {len(self.feature_names)} names vs "
                f"{len(self.feature_values)} values"
            )

    def to_dict(self) -> dict[str, float]:
        """Convert feature vector to name → value dictionary."""
        return dict(zip(self.feature_names, self.feature_values))

    def as_array(self) -> list[float]:
        """Return feature values as a list for numpy/sklearn compatibility."""
        return list(self.feature_values)


class FeatureExtractor:
    """
    Extract behavioral features from stored desktop activities.

    Features are designed to capture:
    - Session duration and intensity
    - Application usage patterns
    - Application transitions and focus
    - Temporal context (time of day, day of week)
    - Activity concentration/entropy

    All features are normalized to comparable ranges to prevent scale bias.
    """

    # Feature metadata: (name, description, expected_range)
    FEATURES = (
        ("session_duration_minutes", "Total session duration in minutes", (0, 1440)),
        ("activity_segment_count", "Number of distinct activity segments", (0, 100)),
        ("unique_application_count", "Number of unique applications used", (0, 50)),
        ("dominant_app_percentage", "% time spent in most-used application (0-1)", (0, 1)),
        ("app_diversity_entropy", "Shannon entropy of app usage distribution (0-1)", (0, 1)),
        ("application_switch_count", "Number of application transitions in session", (0, 100)),
        ("application_switch_rate", "App switches per minute of activity", (0, 10)),
        ("session_start_hour", "Hour of day session started (0-23)", (0, 23)),
        ("session_end_hour", "Hour of day session ended (0-23)", (0, 23)),
        ("session_start_day_of_week", "Day of week session started (0=Monday, 6=Sunday)", (0, 6)),
        ("is_business_hours", "Whether session primarily in business hours 9-17 (0 or 1)", (0, 1)),
        ("session_concentration_ratio", "Ratio of dominant app duration to session duration", (0, 1)),
        ("average_segment_duration_seconds", "Average duration per activity segment", (0, 3600)),
        ("temporal_entropy_concentration", "How concentrated activity is in time (0-1)", (0, 1)),
    )

    FEATURE_NAMES = tuple(name for name, _, _ in FEATURES)

    def __init__(self) -> None:
        """Initialize feature extractor with no state."""
        pass

    def extract_features(
        self,
        session_id: str,
        session_started_at: datetime,
        session_ended_at: datetime,
        activities: Sequence,
    ) -> FeatureVector:
        """
        Extract features from a complete session with its activities.

        Args:
            session_id: Unique session identifier
            session_started_at: Session start time (UTC)
            session_ended_at: Session end time (UTC)
            activities: Sequence of StoredActivity objects for this session
                       (Must have: started_at, ended_at, duration_seconds, application)

        Returns:
            FeatureVector with extracted features and metadata
        """
        if session_ended_at.tzinfo is None or session_started_at.tzinfo is None:
            raise ValueError("Session times must be timezone-aware")

        # Handle empty sessions gracefully
        if not activities:
            return self._create_zero_feature_vector(session_id, session_started_at)

        features = {}

        # 1. Session Duration
        session_duration_seconds = (session_ended_at - session_started_at).total_seconds()
        session_duration_minutes = max(session_duration_seconds / 60.0, 0.0)
        features["session_duration_minutes"] = session_duration_minutes

        # 2. Activity Segment Count
        features["activity_segment_count"] = float(len(activities))

        # 3. Application information
        app_durations = self._compute_app_durations(activities)
        features["unique_application_count"] = float(len(app_durations))

        # 4. Dominant Application
        if app_durations:
            max_duration = max(app_durations.values())
            total_duration = sum(act.duration_seconds for act in activities)
            dominant_percentage = (
                max_duration / total_duration if total_duration > 0 else 0.0
            )
        else:
            dominant_percentage = 0.0
        features["dominant_app_percentage"] = min(dominant_percentage, 1.0)

        # 5. Application Diversity (Shannon Entropy)
        entropy = self._compute_app_entropy(activities)
        features["app_diversity_entropy"] = entropy

        # 6. Application Transitions
        transitions = self._count_app_transitions(activities)
        features["application_switch_count"] = float(transitions)

        # 7. Application Switch Rate (per minute)
        if session_duration_minutes > 0:
            switch_rate = transitions / session_duration_minutes
        else:
            switch_rate = 0.0
        features["application_switch_rate"] = switch_rate

        # 8-10. Temporal Features
        features["session_start_hour"] = float(session_started_at.hour)
        features["session_end_hour"] = float(session_ended_at.hour)
        features["session_start_day_of_week"] = float(session_started_at.weekday())

        # 11. Business Hours Feature
        is_business_hours = self._is_business_hours_session(session_started_at, session_ended_at)
        features["is_business_hours"] = float(is_business_hours)

        # 12. Session Concentration Ratio
        features["session_concentration_ratio"] = min(dominant_percentage, 1.0)

        # 13. Average Segment Duration
        if len(activities) > 0:
            avg_segment_duration = sum(
                act.duration_seconds for act in activities
            ) / len(activities)
        else:
            avg_segment_duration = 0.0
        features["average_segment_duration_seconds"] = avg_segment_duration

        # 14. Temporal Entropy (activity concentration in time)
        temporal_entropy = self._compute_temporal_entropy(activities)
        features["temporal_entropy_concentration"] = temporal_entropy

        # Build feature vector in defined order
        feature_values = tuple(features[name] for name in self.FEATURE_NAMES)

        return FeatureVector(
            session_id=session_id,
            feature_names=self.FEATURE_NAMES,
            feature_values=feature_values,
            metadata={
                "segment_count": len(activities),
                "unique_apps": list(app_durations.keys()) if app_durations else [],
                "total_duration_seconds": session_duration_seconds,
            },
        )

    def _create_zero_feature_vector(self, session_id: str, session_start: datetime) -> FeatureVector:
        """Create a feature vector for an empty session."""
        zero_features = tuple(0.0 for _ in self.FEATURE_NAMES)
        # Set temporal features from session start
        feature_dict = {name: 0.0 for name in self.FEATURE_NAMES}
        feature_dict["session_start_hour"] = float(session_start.hour)
        feature_dict["session_start_day_of_week"] = float(session_start.weekday())
        zero_features = tuple(feature_dict[name] for name in self.FEATURE_NAMES)

        return FeatureVector(
            session_id=session_id,
            feature_names=self.FEATURE_NAMES,
            feature_values=zero_features,
            metadata={
                "segment_count": 0,
                "unique_apps": [],
                "total_duration_seconds": 0.0,
            },
        )

    def _compute_app_durations(self, activities: Sequence) -> dict[str, float]:
        """
        Compute total duration for each application in the session.

        Returns:
            Dictionary mapping application name to total duration in seconds.
            Returns empty dict if no activities have application names.
        """
        app_durations = {}
        for activity in activities:
            # Use application name if available, fall back to process_name
            app_name = activity.application or activity.process_name or "Unknown"
            app_durations[app_name] = app_durations.get(app_name, 0.0) + activity.duration_seconds

        return app_durations

    def _compute_app_entropy(self, activities: Sequence) -> float:
        """
        Compute Shannon entropy of application usage distribution.

        Higher entropy = more diverse app usage (many apps, similar durations)
        Lower entropy = focused usage (one or few dominant apps)

        Returns:
            Entropy normalized to 0-1 range (0 = all time in one app, 1 = perfectly uniform)
        """
        app_durations = self._compute_app_durations(activities)
        if not app_durations:
            return 0.0

        total_duration = sum(app_durations.values())
        if total_duration == 0:
            return 0.0

        # Compute Shannon entropy: H = -sum(p_i * log2(p_i))
        entropy = 0.0
        for duration in app_durations.values():
            if duration > 0:
                probability = duration / total_duration
                entropy -= probability * math.log2(probability)

        # Normalize to 0-1 by dividing by max entropy (log2(num_apps))
        num_apps = len(app_durations)
        if num_apps > 1:
            max_entropy = math.log2(num_apps)
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0

        return min(normalized_entropy, 1.0)

    def _count_app_transitions(self, activities: Sequence) -> int:
        """
        Count the number of times the application changes during the session.

        Returns:
            Count of transitions (0 if only one activity or all same app)
        """
        if len(activities) <= 1:
            return 0

        transitions = 0
        prev_app = None
        for activity in activities:
            current_app = activity.application or activity.process_name
            if prev_app is not None and current_app != prev_app:
                transitions += 1
            prev_app = current_app

        return transitions

    def _is_business_hours_session(self, started: datetime, ended: datetime) -> bool:
        """
        Determine if session is primarily in business hours (9 AM - 5 PM).

        Returns:
            True if session started within business hours
        """
        return 9 <= started.hour < 17

    def _compute_temporal_entropy(self, activities: Sequence) -> float:
        """
        Compute how concentrated activity is across the session timeline.

        High value = activity concentrated in early/late part of session (not spread out)
        Low value = activity spread evenly throughout session

        This captures focus intensity: many short bursts (low entropy) vs
        sustained focus on few apps (high entropy).

        Returns:
            Entropy value normalized to 0-1
        """
        if not activities or len(activities) < 2:
            return 0.0

        # Divide session into 10 time buckets, count activities in each
        total_duration = sum(act.duration_seconds for act in activities)
        if total_duration == 0:
            return 0.0

        num_buckets = min(10, len(activities))
        bucket_durations = [0.0] * num_buckets

        for activity in activities:
            # Estimate which bucket this activity falls into
            # Use start time position in session
            session_start = min(activities, key=lambda a: a.started_at).started_at
            activity_offset = (activity.started_at - session_start).total_seconds()

            bucket_index = min(
                int(activity_offset / total_duration * num_buckets),
                num_buckets - 1,
            )
            bucket_durations[bucket_index] += activity.duration_seconds

        # Compute entropy of bucket distribution
        entropy = 0.0
        for duration in bucket_durations:
            if duration > 0:
                probability = duration / total_duration
                entropy -= probability * math.log2(probability)

        # Normalize
        if num_buckets > 1:
            max_entropy = math.log2(num_buckets)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 0.0

        return min(normalized_entropy, 1.0)


class BatchFeatureExtractor:
    """
    Convenience wrapper for extracting features from multiple sessions at once.

    Useful for preparing training data before model training.
    """

    def __init__(self) -> None:
        self.extractor = FeatureExtractor()

    def extract_batch(
        self,
        sessions_with_activities: Sequence[tuple],
    ) -> list[FeatureVector]:
        """
        Extract features for multiple sessions.

        Args:
            sessions_with_activities: List of tuples:
                (StoredSession, List[StoredActivity])

        Returns:
            List of FeatureVector objects
        """
        feature_vectors = []
        for session, activities in sessions_with_activities:
            if session.ended_at is None:
                continue  # Skip incomplete sessions

            features = self.extractor.extract_features(
                session_id=session.id,
                session_started_at=session.started_at,
                session_ended_at=session.ended_at,
                activities=activities,
            )
            feature_vectors.append(features)

        return feature_vectors

    def extract_as_matrix(
        self,
        sessions_with_activities: Sequence[tuple],
    ) -> tuple[list[list[float]], list[str]]:
        """
        Extract features and return as matrix format suitable for ML libraries.

        Returns:
            (feature_matrix, session_ids)
            where feature_matrix is list of lists (each row is a session's features)
        """
        feature_vectors = self.extract_batch(sessions_with_activities)
        feature_matrix = [fv.as_array() for fv in feature_vectors]
        session_ids = [fv.session_id for fv in feature_vectors]
        return feature_matrix, session_ids
