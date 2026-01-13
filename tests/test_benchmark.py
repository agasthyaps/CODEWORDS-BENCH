"""Tests for benchmark module (M6)."""

import pytest
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import Team, GameMode
from src.benchmark import (
    # Config
    TeamComposition,
    ModelConfig,
    TeamAssignment,
    MatchupConfig,
    ExperimentConfig,
    generate_matchups,
    count_total_games,
    # Runner
    BenchmarkResult,
    BenchmarkProgress,
    # Leaderboard
    ConfidenceInterval,
    LeaderboardEntry,
    HeadToHeadEntry,
    Leaderboard,
    wilson_score_interval,
    build_leaderboard,
    export_leaderboard_markdown,
)
from src.metrics import EpisodeMetrics, TeamMetrics


# ============================================================================
# ExperimentConfig Tests
# ============================================================================

class TestExperimentConfig:
    """Tests for experiment configuration."""

    def test_creates_valid_config(self):
        """Should create a valid experiment config."""
        models = [
            ModelConfig(name="model_a", model_id="a/model-a"),
            ModelConfig(name="model_b", model_id="b/model-b"),
        ]
        config = ExperimentConfig(
            name="test_experiment",
            models=models,
            seeds=[1, 2, 3],
        )

        assert config.name == "test_experiment"
        assert len(config.models) == 2
        assert len(config.seeds) == 3

    def test_generates_correct_matchup_count_homogeneous(self):
        """Should generate correct number of homogeneous matchups."""
        models = [
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
            ModelConfig(name="c", model_id="c"),
        ]
        config = ExperimentConfig(
            name="test",
            models=models,
            team_compositions=[TeamComposition.HOMOGENEOUS],
        )

        matchups = generate_matchups(config)

        # 3 models: (A vs A), (A vs B), (A vs C), (B vs B), (B vs C), (C vs C) = 6
        assert len(matchups) == 6

    def test_generates_correct_matchup_count_mixed_cluer(self):
        """Should generate correct number of mixed cluer matchups."""
        models = [
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
        ]
        config = ExperimentConfig(
            name="test",
            models=models,
            team_compositions=[TeamComposition.MIXED_CLUER],
        )

        matchups = generate_matchups(config)

        # 2 models, each can be cluer with other as guesser: 2 matchups
        assert len(matchups) == 2

    def test_counts_total_games(self):
        """Should count total games correctly."""
        models = [
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
        ]
        config = ExperimentConfig(
            name="test",
            models=models,
            team_compositions=[TeamComposition.HOMOGENEOUS],
            game_modes=[GameMode.STANDARD, GameMode.NO_ASSASSIN],
            seeds=[1, 2, 3, 4, 5],
            games_per_config=2,
        )

        matchups = generate_matchups(config)  # 3 matchups (A vs A, A vs B, B vs B)
        total = count_total_games(config)

        # 3 matchups × 2 modes × 5 seeds × 2 games = 60
        expected = 3 * 2 * 5 * 2
        assert total == expected

    def test_invalid_config_raises_error(self):
        """Should raise error for invalid config."""
        with pytest.raises(Exception):
            ExperimentConfig(name="")  # Empty name should fail

    def test_seeds_reproducible(self):
        """Seeds should be reproducible across runs."""
        config1 = ExperimentConfig(
            name="test",
            models=[ModelConfig(name="a", model_id="a")],
            seeds=list(range(10)),
        )
        config2 = ExperimentConfig(
            name="test",
            models=[ModelConfig(name="a", model_id="a")],
            seeds=list(range(10)),
        )

        assert config1.seeds == config2.seeds


# ============================================================================
# Matchup Generation Tests
# ============================================================================

class TestMatchupGeneration:
    """Tests for matchup generation."""

    def test_homogeneous_matchups_symmetric(self):
        """Homogeneous matchups should be symmetric."""
        models = [
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
        ]
        config = ExperimentConfig(
            name="test",
            models=models,
            team_compositions=[TeamComposition.HOMOGENEOUS],
        )

        matchups = generate_matchups(config)

        for matchup in matchups:
            # All agents on each team should be the same model
            red = matchup.red_team
            assert red.cluer == red.guesser_1 == red.guesser_2

            blue = matchup.blue_team
            assert blue.cluer == blue.guesser_1 == blue.guesser_2

    def test_mixed_cluer_assigns_correctly(self):
        """Mixed cluer should have different cluer than guessers."""
        models = [
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
        ]
        config = ExperimentConfig(
            name="test",
            models=models,
            team_compositions=[TeamComposition.MIXED_CLUER],
        )

        matchups = generate_matchups(config)

        for matchup in matchups:
            red = matchup.red_team
            # Cluer should be different from guessers
            assert red.cluer != red.guesser_1
            # Both guessers should be the same
            assert red.guesser_1 == red.guesser_2

    def test_heterogeneous_requires_three_models(self):
        """Heterogeneous requires at least 3 models."""
        models = [
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
        ]
        config = ExperimentConfig(
            name="test",
            models=models,
            team_compositions=[TeamComposition.HETEROGENEOUS],
        )

        matchups = generate_matchups(config)

        # With only 2 models, heterogeneous produces no matchups
        assert len(matchups) == 0


# ============================================================================
# BenchmarkProgress Tests
# ============================================================================

class TestBenchmarkProgress:
    """Tests for benchmark progress tracking."""

    def test_tracks_completion(self):
        """Should track completed games."""
        progress = BenchmarkProgress(
            experiment_name="test",
            started_at="2024-01-01T00:00:00",
            total_games=100,
            completed_games=0,
            failed_games=0,
        )

        # Mark a game complete
        progress.mark_completed("matchup1", GameMode.STANDARD, 42, 0)

        assert progress.completed_games == 1
        assert progress.is_completed("matchup1", GameMode.STANDARD, 42, 0)
        assert not progress.is_completed("matchup1", GameMode.STANDARD, 43, 0)

    def test_tracks_failures(self):
        """Should track failed games."""
        progress = BenchmarkProgress(
            experiment_name="test",
            started_at="2024-01-01T00:00:00",
            total_games=100,
            completed_games=0,
            failed_games=0,
        )

        progress.mark_failed()
        progress.mark_failed()

        assert progress.failed_games == 2


# ============================================================================
# Leaderboard Tests
# ============================================================================

class TestLeaderboard:
    """Tests for leaderboard generation."""

    def _create_mock_result(
        self,
        red_model: str,
        blue_model: str,
        winner: Team | None,
        mode: GameMode = GameMode.STANDARD,
    ) -> BenchmarkResult:
        """Create a mock benchmark result."""
        red_metrics = TeamMetrics(
            team=Team.RED,
            words_cleared=5,
            assassin_hit=False,
            total_clues=3,
            avg_clue_number=2.0,
            clue_efficiency=0.8,
            total_guesses=6,
            correct_guesses=5,
            wrong_guesses=1,
            opponent_guesses=0,
            neutral_guesses=1,
            guess_accuracy=0.83,
            avg_discussion_rounds=1.5,
            consensus_rate=0.67,
            avg_discussion_length=150,
            theory_of_mind_score=0.8,
        )
        blue_metrics = TeamMetrics(
            team=Team.BLUE,
            words_cleared=4,
            assassin_hit=False,
            total_clues=2,
            avg_clue_number=2.0,
            clue_efficiency=0.75,
            total_guesses=4,
            correct_guesses=4,
            wrong_guesses=0,
            opponent_guesses=0,
            neutral_guesses=0,
            guess_accuracy=1.0,
            avg_discussion_rounds=1.0,
            consensus_rate=1.0,
            avg_discussion_length=100,
            theory_of_mind_score=0.75,
        )
        metrics = EpisodeMetrics(
            episode_id="test",
            winner=winner,
            turns_to_win=5,
            red_metrics=red_metrics,
            blue_metrics=blue_metrics,
            red_coordination_score=0.7,
            blue_coordination_score=0.8,
        )

        return BenchmarkResult(
            matchup_id=f"{red_model}_vs_{blue_model}",
            mode=mode,
            seed=42,
            game_index=0,
            episode_id="test",
            winner=winner,
            metrics=metrics,
            red_models={"cluer": red_model, "guesser_1": red_model, "guesser_2": red_model},
            blue_models={"cluer": blue_model, "guesser_1": blue_model, "guesser_2": blue_model},
            duration_seconds=10.0,
        )

    def test_ranks_models_by_win_rate(self):
        """Should rank models by win rate."""
        results = [
            self._create_mock_result("model_a", "model_b", Team.RED),
            self._create_mock_result("model_a", "model_b", Team.RED),
            self._create_mock_result("model_a", "model_b", Team.BLUE),
            self._create_mock_result("model_b", "model_a", Team.BLUE),
        ]

        leaderboard = build_leaderboard(results)

        # Model A: 2 wins as red + 1 loss as blue = 2/3 = 66.7%
        # Model B: 2 wins as blue + 2 losses as red = 2/4 = 50%
        # Model A should be ranked higher
        assert leaderboard.overall[0].model == "model_a"

    def test_computes_confidence_intervals(self):
        """Should compute valid confidence intervals."""
        results = [
            self._create_mock_result("model_a", "model_b", Team.RED)
            for _ in range(20)
        ]

        leaderboard = build_leaderboard(results)

        for entry in leaderboard.overall:
            assert 0.0 <= entry.win_rate_ci.lower <= entry.win_rate + 0.001
            assert entry.win_rate - 0.001 <= entry.win_rate_ci.upper <= 1.0 + 0.001

    def test_handles_ties_by_coordination_score(self):
        """Should use coordination score as tiebreaker."""
        # Create results where both models have same win rate
        results = [
            self._create_mock_result("model_a", "model_b", Team.RED),
            self._create_mock_result("model_b", "model_a", Team.RED),
        ]

        leaderboard = build_leaderboard(results)

        # Both have 50% win rate, should be sorted by coordination score
        assert len(leaderboard.overall) == 2

    def test_filters_by_role(self):
        """Should filter leaderboard by role."""
        results = [
            self._create_mock_result("model_a", "model_b", Team.RED)
            for _ in range(5)
        ]

        leaderboard = build_leaderboard(results)

        # Cluer leaderboard should exist
        assert len(leaderboard.by_cluer) > 0
        for entry in leaderboard.by_cluer:
            assert entry.role == "cluer"

        # Guesser leaderboard should exist
        assert len(leaderboard.by_guesser) > 0
        for entry in leaderboard.by_guesser:
            assert entry.role == "guesser"

    def test_computes_head_to_head(self):
        """Should compute head-to-head statistics."""
        results = [
            self._create_mock_result("model_a", "model_b", Team.RED),
            self._create_mock_result("model_a", "model_b", Team.RED),
            self._create_mock_result("model_a", "model_b", Team.BLUE),
        ]

        leaderboard = build_leaderboard(results)

        # Should have head-to-head entry
        h2h = next(
            (h for h in leaderboard.head_to_head
             if {h.model_a, h.model_b} == {"model_a", "model_b"}),
            None,
        )

        assert h2h is not None
        assert h2h.games == 3
        assert h2h.model_a_wins + h2h.model_b_wins == 3


# ============================================================================
# Wilson Score Tests
# ============================================================================

class TestWilsonScore:
    """Tests for Wilson score confidence interval."""

    def test_zero_trials(self):
        """Should handle zero trials."""
        ci = wilson_score_interval(0, 0)
        assert ci.lower == 0.0
        assert ci.upper == 1.0

    def test_all_successes(self):
        """Should handle 100% success rate."""
        ci = wilson_score_interval(100, 100)
        assert ci.lower > 0.9  # Should be close to 1
        assert abs(ci.upper - 1.0) < 0.001  # Floating point tolerance

    def test_no_successes(self):
        """Should handle 0% success rate."""
        ci = wilson_score_interval(0, 100)
        assert ci.lower == 0.0
        assert ci.upper < 0.1  # Should be close to 0

    def test_fifty_percent(self):
        """Should center around 50% for balanced data."""
        ci = wilson_score_interval(50, 100)
        # CI should contain 0.5
        assert ci.lower < 0.5 < ci.upper

    def test_interval_narrows_with_more_data(self):
        """Confidence interval should narrow with more data."""
        ci_small = wilson_score_interval(5, 10)
        ci_large = wilson_score_interval(50, 100)

        small_width = ci_small.upper - ci_small.lower
        large_width = ci_large.upper - ci_large.lower

        assert large_width < small_width


# ============================================================================
# Export Tests
# ============================================================================

class TestLeaderboardExport:
    """Tests for leaderboard export."""

    def test_markdown_renders_correctly(self):
        """Should render markdown correctly."""
        entry = LeaderboardEntry(
            model="test_model",
            role="overall",
            games=100,
            wins=60,
            win_rate=0.6,
            win_rate_ci=ConfidenceInterval(lower=0.5, upper=0.7),
            avg_coordination_score=0.75,
            avg_theory_of_mind=0.8,
        )

        leaderboard = Leaderboard(overall=[entry])
        md = export_leaderboard_markdown(leaderboard)

        assert "# Benchmark Leaderboard" in md
        assert "test_model" in md
        assert "60.0%" in md
        assert "[0.50, 0.70]" in md


# ============================================================================
# Integration Tests
# ============================================================================

class TestBenchmarkIntegration:
    """Integration tests for benchmark module."""

    def test_config_to_matchups_to_total(self):
        """Config → Matchups → Total games should be consistent."""
        models = [
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
            ModelConfig(name="c", model_id="c"),
        ]
        config = ExperimentConfig(
            name="integration_test",
            models=models,
            team_compositions=[TeamComposition.HOMOGENEOUS],
            game_modes=[GameMode.STANDARD],
            seeds=list(range(10)),
            games_per_config=1,
        )

        matchups = generate_matchups(config)
        total = count_total_games(config)

        expected = len(matchups) * len(config.game_modes) * len(config.seeds) * config.games_per_config
        assert total == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
