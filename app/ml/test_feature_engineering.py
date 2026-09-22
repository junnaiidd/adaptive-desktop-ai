"""
Comprehensive tests for Phase 2A: ML Feature Engineering Foundation

Tests cover:
- Normal multi-app sessions
- Single-app sessions
- Empty sessions
- Missing application/process values
- Deterministic output
- Feature alignment
- Temporal feature extraction
- Application entropy calculation
- Edge cases and error handling
"""

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import pytest

from app.ml.feature_engineering import FeatureExtractor, FeatureVector, BatchFeatureExtractor


def _timestamp(seconds: int = 0, hour: int = 12) -> datetime:
    """Create a timezone-aware test timestamp."""
    return datetime(2026, 1, 1, hour=hour, tzinfo=timezone.utc) + timedelta(seconds=seconds)


@dataclass(frozen=True)
class MockActivity:
    """Mock activity object matching StoredActivity interface."""

    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    application: str | None
    process_name: str | None
    window_title: str | None = None

    def __hash__(self):
        return hash(self.started_at)


# ============================================================================
# Feature Vector Tests
# ============================================================================


def test_feature_vector_creation_valid() -> None:
    """Test creating a valid feature vector."""
    fv = FeatureVector(
        session_id="session-1",
        feature_names=("duration", "count"),
        feature_values=(10.0, 5.0),
        metadata={"test": "value"},
    )

    assert fv.session_id == "session-1"
    assert len(fv.feature_names) == 2
    assert len(fv.feature_values) == 2
    assert fv.to_dict() == {"duration": 10.0, "count": 5.0}
    assert fv.as_array() == [10.0, 5.0]


def test_feature_vector_creation_name_value_mismatch() -> None:
    """Test that feature vector rejects mismatched names and values."""
    with pytest.raises(ValueError, match="Feature count mismatch"):
        FeatureVector(
            session_id="session-1",
            feature_names=("duration", "count", "extra"),
            feature_values=(10.0, 5.0),
            metadata={},
        )


# ============================================================================
# Basic Feature Extraction Tests
# ============================================================================


def test_feature_extractor_single_activity() -> None:
    """Test feature extraction from a single-activity session."""
    extractor = FeatureExtractor()
    activity = MockActivity(
        started_at=_timestamp(0),
        ended_at=_timestamp(300),
        duration_seconds=300.0,
        application="VSCode",
        process_name="Code.exe",
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(300),
        activities=[activity],
    )

    assert features.session_id == "session-1"
    assert len(features.feature_names) == len(FeatureExtractor.FEATURE_NAMES)
    assert len(features.feature_values) == len(FeatureExtractor.FEATURE_NAMES)

    # Verify expected values
    feature_dict = features.to_dict()
    assert feature_dict["session_duration_minutes"] == 5.0  # 300 seconds = 5 minutes
    assert feature_dict["activity_segment_count"] == 1.0
    assert feature_dict["unique_application_count"] == 1.0
    assert feature_dict["dominant_app_percentage"] == 1.0  # 100% in one app
    assert feature_dict["application_switch_count"] == 0.0  # No transitions


def test_feature_extractor_multiple_activities_same_app() -> None:
    """Test feature extraction with multiple activities in same application."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(100),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(100),
            ended_at=_timestamp(200),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(200),
            ended_at=_timestamp(300),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(300),
        activities=activities,
    )

    feature_dict = features.to_dict()
    assert feature_dict["session_duration_minutes"] == 5.0
    assert feature_dict["activity_segment_count"] == 3.0
    assert feature_dict["unique_application_count"] == 1.0
    assert feature_dict["dominant_app_percentage"] == 1.0
    assert feature_dict["application_switch_count"] == 0.0  # No transitions


def test_feature_extractor_multiple_apps() -> None:
    """Test feature extraction with multiple different applications."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(100),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(100),
            ended_at=_timestamp(200),
            duration_seconds=100.0,
            application="Chrome",
            process_name="chrome.exe",
        ),
        MockActivity(
            started_at=_timestamp(200),
            ended_at=_timestamp(400),
            duration_seconds=200.0,
            application="VSCode",
            process_name="Code.exe",
        ),
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(400),
        activities=activities,
    )

    feature_dict = features.to_dict()
    assert feature_dict["session_duration_minutes"] == pytest.approx(400 / 60.0)
    assert feature_dict["activity_segment_count"] == 3.0
    assert feature_dict["unique_application_count"] == 2.0  # VSCode and Chrome
    assert feature_dict["dominant_app_percentage"] == pytest.approx(0.75, abs=0.01)  # VSCode 300s/400s = 75%
    assert feature_dict["application_switch_count"] == 2.0  # VSCode → Chrome, Chrome → VSCode


# ============================================================================
# Application Entropy Tests
# ============================================================================


def test_application_entropy_single_app() -> None:
    """Test that entropy is 0 when all time is in one application."""
    extractor = FeatureExtractor()
    activity = MockActivity(
        started_at=_timestamp(0),
        ended_at=_timestamp(600),
        duration_seconds=600.0,
        application="VSCode",
        process_name="Code.exe",
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(600),
        activities=[activity],
    )

    feature_dict = features.to_dict()
    assert feature_dict["app_diversity_entropy"] == 0.0  # No diversity


def test_application_entropy_equal_distribution() -> None:
    """Test entropy with equally distributed app usage."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(100),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(100),
            ended_at=_timestamp(200),
            duration_seconds=100.0,
            application="Chrome",
            process_name="chrome.exe",
        ),
        MockActivity(
            started_at=_timestamp(200),
            ended_at=_timestamp(300),
            duration_seconds=100.0,
            application="Terminal",
            process_name="powershell.exe",
        ),
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(300),
        activities=activities,
    )

    feature_dict = features.to_dict()
    # Equal distribution should have high entropy (close to 1.0 normalized)
    assert feature_dict["app_diversity_entropy"] > 0.8
    assert feature_dict["app_diversity_entropy"] <= 1.0


def test_application_entropy_unequal_distribution() -> None:
    """Test entropy with unequal app usage."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(450),
            duration_seconds=450.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(450),
            ended_at=_timestamp(500),
            duration_seconds=50.0,
            application="Chrome",
            process_name="chrome.exe",
        ),
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(500),
        activities=activities,
    )

    feature_dict = features.to_dict()
    # Unequal distribution (90/10 split) should have lower entropy
    assert 0.0 < feature_dict["app_diversity_entropy"] < 0.5


# ============================================================================
# Transition Tests
# ============================================================================


def test_application_transitions_no_changes() -> None:
    """Test transition counting with no app changes."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(100),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(100),
            ended_at=_timestamp(200),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(200),
        activities=activities,
    )

    feature_dict = features.to_dict()
    assert feature_dict["application_switch_count"] == 0.0


def test_application_transitions_frequent_switching() -> None:
    """Test transition counting with frequent app switches."""
    extractor = FeatureExtractor()
    apps = ["VSCode", "Chrome", "Terminal", "Slack"]
    activities = []
    for i, app in enumerate(apps * 2):  # Repeat twice
        start = _timestamp(i * 50)
        end = _timestamp((i + 1) * 50)
        activities.append(
            MockActivity(
                started_at=start,
                ended_at=end,
                duration_seconds=50.0,
                application=app,
                process_name=f"{app.lower()}.exe",
            )
        )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(400),
        activities=activities,
    )

    feature_dict = features.to_dict()
    # 8 activities, 7 transitions (first activity has no transition)
    assert feature_dict["application_switch_count"] == 7.0
    assert feature_dict["application_switch_rate"] > 0.0


# ============================================================================
# Temporal Feature Tests
# ============================================================================


def test_temporal_features_morning_session() -> None:
    """Test temporal feature extraction for a morning session."""
    extractor = FeatureExtractor()
    # Create explicit UTC timestamps for a 9 AM session lasting 1 hour
    start_time = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    activity = MockActivity(
        started_at=start_time,
        ended_at=end_time,
        duration_seconds=3600.0,
        application="VSCode",
        process_name="Code.exe",
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=start_time,
        session_ended_at=end_time,
        activities=[activity],
    )

    feature_dict = features.to_dict()
    assert feature_dict["session_start_hour"] == 9.0
    assert feature_dict["session_end_hour"] == 10.0
    assert feature_dict["is_business_hours"] == 1.0  # 9 AM is business hours


def test_temporal_features_evening_session() -> None:
    """Test temporal feature extraction for an evening session."""
    extractor = FeatureExtractor()
    activity = MockActivity(
        started_at=_timestamp(0, hour=18),
        ended_at=_timestamp(3600, hour=18),
        duration_seconds=3600.0,
        application="VSCode",
        process_name="Code.exe",
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0, hour=18),
        session_ended_at=_timestamp(3600, hour=19),
        activities=[activity],
    )

    feature_dict = features.to_dict()
    assert feature_dict["session_start_hour"] == 18.0
    assert feature_dict["is_business_hours"] == 0.0  # 6 PM is not business hours


def test_temporal_features_day_of_week() -> None:
    """Test that day-of-week feature is extracted."""
    extractor = FeatureExtractor()
    # Wednesday, 2026-01-01 is a day 2 (Wednesday = 2 in weekday())
    activity = MockActivity(
        started_at=_timestamp(0),
        ended_at=_timestamp(3600),
        duration_seconds=3600.0,
        application="VSCode",
        process_name="Code.exe",
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(3600),
        activities=[activity],
    )

    feature_dict = features.to_dict()
    # 2026-01-01 is a Thursday (weekday = 3)
    assert 0.0 <= feature_dict["session_start_day_of_week"] <= 6.0


# ============================================================================
# Empty and Missing Data Tests
# ============================================================================


def test_feature_extraction_empty_session() -> None:
    """Test feature extraction with no activities."""
    extractor = FeatureExtractor()

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(300),
        activities=[],
    )

    feature_dict = features.to_dict()
    assert feature_dict["activity_segment_count"] == 0.0
    assert feature_dict["unique_application_count"] == 0.0
    assert feature_dict["dominant_app_percentage"] == 0.0
    assert feature_dict["application_switch_count"] == 0.0
    # Temporal features should still be valid
    assert 0.0 <= feature_dict["session_start_hour"] <= 23.0


def test_feature_extraction_missing_application_name() -> None:
    """Test feature extraction when application name is None."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(100),
            duration_seconds=100.0,
            application=None,
            process_name="unknown.exe",
        ),
        MockActivity(
            started_at=_timestamp(100),
            ended_at=_timestamp(200),
            duration_seconds=100.0,
            application="Chrome",
            process_name="chrome.exe",
        ),
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(200),
        activities=activities,
    )

    # Should gracefully handle None application
    assert features is not None
    assert len(features.feature_values) == len(FeatureExtractor.FEATURE_NAMES)


def test_feature_extraction_missing_process_name() -> None:
    """Test feature extraction when process_name is None."""
    extractor = FeatureExtractor()
    activity = MockActivity(
        started_at=_timestamp(0),
        ended_at=_timestamp(100),
        duration_seconds=100.0,
        application="Chrome",
        process_name=None,
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(100),
        activities=[activity],
    )

    assert features is not None
    assert len(features.feature_values) == len(FeatureExtractor.FEATURE_NAMES)


# ============================================================================
# Determinism Tests
# ============================================================================


def test_feature_extraction_deterministic() -> None:
    """Test that feature extraction is deterministic."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(100),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(100),
            ended_at=_timestamp(200),
            duration_seconds=100.0,
            application="Chrome",
            process_name="chrome.exe",
        ),
    ]

    features1 = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(200),
        activities=activities,
    )

    features2 = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(200),
        activities=activities,
    )

    # Exact same features
    assert features1.feature_values == features2.feature_values
    assert features1.to_dict() == features2.to_dict()


# ============================================================================
# Feature Alignment Tests
# ============================================================================


def test_feature_names_and_values_aligned() -> None:
    """Test that feature names align with feature values."""
    extractor = FeatureExtractor()
    activity = MockActivity(
        started_at=_timestamp(0),
        ended_at=_timestamp(300),
        duration_seconds=300.0,
        application="VSCode",
        process_name="Code.exe",
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(300),
        activities=[activity],
    )

    # Convert to dict and back to list
    feature_dict = features.to_dict()
    reconstructed = [feature_dict[name] for name in features.feature_names]

    assert reconstructed == list(features.feature_values)


def test_feature_count_matches_specification() -> None:
    """Test that actual feature count matches the specification."""
    extractor = FeatureExtractor()
    activity = MockActivity(
        started_at=_timestamp(0),
        ended_at=_timestamp(300),
        duration_seconds=300.0,
        application="VSCode",
        process_name="Code.exe",
    )

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(300),
        activities=[activity],
    )

    expected_count = len(FeatureExtractor.FEATURES)
    assert len(features.feature_names) == expected_count
    assert len(features.feature_values) == expected_count


# ============================================================================
# Batch Feature Extraction Tests
# ============================================================================


def test_batch_feature_extractor_multiple_sessions() -> None:
    """Test batch extraction of features from multiple sessions."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MockSession:
        id: str
        started_at: datetime
        ended_at: datetime | None

    batch_extractor = BatchFeatureExtractor()

    sessions_with_activities = [
        (
            MockSession("session-1", _timestamp(0), _timestamp(300)),
            [
                MockActivity(
                    started_at=_timestamp(0),
                    ended_at=_timestamp(300),
                    duration_seconds=300.0,
                    application="VSCode",
                    process_name="Code.exe",
                )
            ],
        ),
        (
            MockSession("session-2", _timestamp(400), _timestamp(700)),
            [
                MockActivity(
                    started_at=_timestamp(400),
                    ended_at=_timestamp(500),
                    duration_seconds=100.0,
                    application="Chrome",
                    process_name="chrome.exe",
                ),
                MockActivity(
                    started_at=_timestamp(500),
                    ended_at=_timestamp(700),
                    duration_seconds=200.0,
                    application="VSCode",
                    process_name="Code.exe",
                ),
            ],
        ),
    ]

    feature_vectors = batch_extractor.extract_batch(sessions_with_activities)

    assert len(feature_vectors) == 2
    assert feature_vectors[0].session_id == "session-1"
    assert feature_vectors[1].session_id == "session-2"


def test_batch_feature_extractor_as_matrix() -> None:
    """Test batch extraction returning matrix format."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MockSession:
        id: str
        started_at: datetime
        ended_at: datetime | None

    batch_extractor = BatchFeatureExtractor()

    sessions_with_activities = [
        (
            MockSession("session-1", _timestamp(0), _timestamp(300)),
            [
                MockActivity(
                    started_at=_timestamp(0),
                    ended_at=_timestamp(300),
                    duration_seconds=300.0,
                    application="VSCode",
                    process_name="Code.exe",
                )
            ],
        ),
        (
            MockSession("session-2", _timestamp(400), _timestamp(700)),
            [
                MockActivity(
                    started_at=_timestamp(400),
                    ended_at=_timestamp(700),
                    duration_seconds=300.0,
                    application="Chrome",
                    process_name="chrome.exe",
                )
            ],
        ),
    ]

    feature_matrix, session_ids = batch_extractor.extract_as_matrix(sessions_with_activities)

    assert len(feature_matrix) == 2
    assert len(session_ids) == 2
    assert session_ids == ["session-1", "session-2"]
    # Each row should have same number of features
    assert len(feature_matrix[0]) == len(FeatureExtractor.FEATURE_NAMES)
    assert len(feature_matrix[1]) == len(FeatureExtractor.FEATURE_NAMES)


# ============================================================================
# Edge Cases and Validation Tests
# ============================================================================


def test_feature_extraction_with_zero_duration_activity() -> None:
    """Test handling of activities with zero duration."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(0),
            duration_seconds=0.0,
            application="VSCode",
            process_name="Code.exe",
        )
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(1),
        activities=activities,
    )

    assert features is not None
    assert all(v >= 0.0 for v in features.feature_values)


def test_feature_extraction_timezone_aware_required() -> None:
    """Test that timezone-naive timestamps raise an error."""
    extractor = FeatureExtractor()
    naive_time = datetime(2026, 1, 1, 12, 0, 0)  # No timezone

    with pytest.raises(ValueError, match="timezone-aware"):
        extractor.extract_features(
            session_id="session-1",
            session_started_at=naive_time,
            session_ended_at=naive_time,
            activities=[],
        )


def test_feature_values_in_reasonable_ranges() -> None:
    """Test that extracted features are in reasonable ranges."""
    extractor = FeatureExtractor()
    activities = [
        MockActivity(
            started_at=_timestamp(0),
            ended_at=_timestamp(100),
            duration_seconds=100.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=_timestamp(100),
            ended_at=_timestamp(200),
            duration_seconds=100.0,
            application="Chrome",
            process_name="chrome.exe",
        ),
    ]

    features = extractor.extract_features(
        session_id="session-1",
        session_started_at=_timestamp(0),
        session_ended_at=_timestamp(200),
        activities=activities,
    )

    feature_dict = features.to_dict()

    # Verify ranges
    assert feature_dict["session_duration_minutes"] >= 0
    assert feature_dict["activity_segment_count"] >= 0
    assert feature_dict["unique_application_count"] >= 0
    assert 0 <= feature_dict["dominant_app_percentage"] <= 1.0
    assert 0 <= feature_dict["app_diversity_entropy"] <= 1.0
    assert feature_dict["application_switch_count"] >= 0
    assert feature_dict["application_switch_rate"] >= 0
    assert 0 <= feature_dict["session_start_hour"] <= 23
    assert 0 <= feature_dict["session_end_hour"] <= 23
    assert 0 <= feature_dict["session_start_day_of_week"] <= 6
    assert feature_dict["is_business_hours"] in (0.0, 1.0)


# ============================================================================
# Real-World Scenario Tests
# ============================================================================


def test_coding_session_scenario() -> None:
    """Test feature extraction for a typical coding session."""
    extractor = FeatureExtractor()
    # Coding session: VS Code dominant, with occasional Chrome and Terminal
    # Session: 9:00 AM to 9:40 AM (2400 seconds = 40 minutes)
    start_time = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    
    activities = [
        MockActivity(
            started_at=start_time,
            ended_at=start_time + timedelta(seconds=600),
            duration_seconds=600.0,
            application="VSCode",
            process_name="Code.exe",
        ),
        MockActivity(
            started_at=start_time + timedelta(seconds=600),
            ended_at=start_time + timedelta(seconds=900),
            duration_seconds=300.0,
            application="Chrome",
            process_name="chrome.exe",
        ),
        MockActivity(
            started_at=start_time + timedelta(seconds=900),
            ended_at=start_time + timedelta(seconds=1200),
            duration_seconds=300.0,
            application="Terminal",
            process_name="powershell.exe",
        ),
        MockActivity(
            started_at=start_time + timedelta(seconds=1200),
            ended_at=start_time + timedelta(seconds=2400),
            duration_seconds=1200.0,
            application="VSCode",
            process_name="Code.exe",
        ),
    ]

    end_time = start_time + timedelta(seconds=2400)

    features = extractor.extract_features(
        session_id="coding-session",
        session_started_at=start_time,
        session_ended_at=end_time,
        activities=activities,
    )

    feature_dict = features.to_dict()

    # Verify coding session characteristics
    assert feature_dict["unique_application_count"] == 3.0  # VSCode, Chrome, Terminal
    assert feature_dict["dominant_app_percentage"] > 0.6  # VSCode dominant (60%)
    assert feature_dict["application_switch_count"] == 3.0  # Three transitions
    assert feature_dict["is_business_hours"] == 1.0  # Morning is business hours
    assert feature_dict["session_duration_minutes"] == 40.0


def test_research_session_scenario() -> None:
    """Test feature extraction for a research browsing session."""
    extractor = FeatureExtractor()
    # Research session: many quick Chrome tab switches
    activities = []
    for i in range(8):
        activities.append(
            MockActivity(
                started_at=_timestamp(i * 150, hour=14),
                ended_at=_timestamp((i + 1) * 150, hour=14),
                duration_seconds=150.0,
                application="Chrome",
                process_name="chrome.exe",
            )
        )

    features = extractor.extract_features(
        session_id="research-session",
        session_started_at=_timestamp(0, hour=14),
        session_ended_at=_timestamp(1200, hour=14),
        activities=activities,
    )

    feature_dict = features.to_dict()

    # Verify research session characteristics
    assert feature_dict["unique_application_count"] == 1.0  # Only Chrome
    assert feature_dict["dominant_app_percentage"] == 1.0  # 100% Chrome
    assert feature_dict["application_switch_count"] == 0.0  # No switches (same app)
    assert feature_dict["is_business_hours"] == 1.0  # Afternoon is business hours


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
