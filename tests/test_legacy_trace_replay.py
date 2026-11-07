"""Verify diagnostic legacy traces match saved original episode records."""

from evaluation.legacy_trace_replay import TRACE_CONDITIONS, replay_legacy_episode


def test_seed42_episode_zero_replays_match_original_csv_for_all_primary_models():
    assert len(TRACE_CONDITIONS) == 5
    for condition_id in TRACE_CONDITIONS:
        replay = replay_legacy_episode(condition_id, seed_group=42, episode_index=0)
        assert replay["replay_matches_historical_csv"] is True
        assert replay["historical_csv_score"] == replay["replayed_score"]
        assert replay["historical_csv_length"] == replay["replayed_length"]
        assert replay["trajectory"][-1]["ball_lost"] is True
        assert replay["trajectory"][-1]["collision_audit"]["miss_confirmed"] is True
        assert sum(replay["selected_action_counts"].values()) == replay["replayed_length"]
        assert sum(replay["executed_action_counts_after_sticky_handling"].values()) == replay["replayed_length"]
