"""Tests for metrics computation and export (M5)."""

import json
import pytest
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import (
    Team, GameConfig, GameMode, Phase, CardType,
    create_game, Board,
)
from src.agents import AgentConfig, CluerAgent, GuesserAgent, MockProvider
from src.runner import (
    TeamConfig, TeamAgents, GhostTeam, GhostMode, TurnTraces,
    ExtendedEpisodeRecord, run_episode,
)
from src.metrics import (
    TeamMetrics, EpisodeMetrics, AggregateMetrics,
    compute_team_metrics, compute_coordination_score,
    compute_episode_metrics, compute_aggregate_metrics,
    export_metrics, export_episode_markdown, export_aggregate_markdown,
)
from src.engine.models import AgentTrace


def create_mock_team(team: Team, game_state, unique_prefix: str = ""):
    """Create a mock team that plays reasonably."""
    team_key = team.value.lower()
    team_words = list(game_state.board.key_by_category[team_key])

    cluer_config = AgentConfig(
        model="mock", role="cluer", team=team,
        agent_id=f"{team_key}_cluer", temperature=0.7,
    )
    guesser1_config = AgentConfig(
        model="mock", role="guesser", team=team,
        agent_id=f"{team_key}_guesser_1", temperature=0.7,
    )
    guesser2_config = AgentConfig(
        model="mock", role="guesser", team=team,
        agent_id=f"{team_key}_guesser_2", temperature=0.7,
    )

    clue_words = [
        f"{unique_prefix}ALPHA", f"{unique_prefix}BRAVO", f"{unique_prefix}CHARLIE",
        f"{unique_prefix}DELTA", f"{unique_prefix}ECHO", f"{unique_prefix}FOXTROT",
        f"{unique_prefix}GOLF", f"{unique_prefix}HOTEL", f"{unique_prefix}INDIA",
        f"{unique_prefix}JULIET", f"{unique_prefix}KILO", f"{unique_prefix}LIMA",
    ]
    cluer_responses = [f"CLUE: {w}\nNUMBER: 2\nREASONING: Test." for w in clue_words]
    cluer_provider = MockProvider(responses=cluer_responses)

    guesser_responses = []
    for i in range(0, len(team_words), 2):
        words = team_words[i:i+2]
        guesser_responses.append(f"Let's guess.\nCONSENSUS: YES\nTOP: {', '.join(words)}")
        guesser_responses.append(f"GUESSES: {', '.join(words)}\nREASONING: Go.")

    guesser1_provider = MockProvider(responses=guesser_responses)
    guesser2_provider = MockProvider(responses=[
        "Agreed.\nCONSENSUS: YES\nTOP: word" for _ in range(20)
    ])

    return TeamAgents(
        cluer=CluerAgent(cluer_config, cluer_provider),
        guesser_1=GuesserAgent(guesser1_config, guesser1_provider),
        guesser_2=GuesserAgent(guesser2_config, guesser2_provider),
    )


# ============================================================================
# Episode Metrics Tests
# ============================================================================

class TestEpisodeMetrics:
    """Tests for episode metrics computation."""

    @pytest.mark.asyncio
    async def test_correct_winner_extracted(self):
        """Winner should be correctly extracted from episode."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        red_team = create_mock_team(Team.RED, state, "R")
        blue_team = create_mock_team(Team.BLUE, state, "B")

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=blue_team,
            max_turns=30,
        )

        metrics = compute_episode_metrics(episode)

        assert metrics.winner == episode.winner
        assert metrics.winner is not None

    @pytest.mark.asyncio
    async def test_turn_count_matches_transcript(self):
        """Turn count should match episode total_turns."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        red_team = create_mock_team(Team.RED, state, "R")
        blue_team = GhostTeam(Team.BLUE, GhostMode.PASS)

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=blue_team,
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)

        assert metrics.turns_to_win == episode.total_turns

    @pytest.mark.asyncio
    async def test_clue_efficiency_calculation(self):
        """Clue efficiency should be correct_guesses / sum(clue_numbers)."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        # Use standard mock team - it provides enough unique clue words
        red_team = create_mock_team(Team.RED, state, "EFFC")

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)

        # Red should have won by clearing 9 words
        assert metrics.winner == Team.RED
        # With 9 correct guesses and multiple clues of 2 each
        # efficiency = correct / sum(clue_numbers)
        # 9 correct / (5 clues * 2 each) = 9/10 = 0.9 approximately
        assert metrics.red_metrics.clue_efficiency > 0
        assert metrics.red_metrics.clue_efficiency <= 1.0

    @pytest.mark.asyncio
    async def test_guess_accuracy_calculation(self):
        """Guess accuracy should be correct_guesses / total_guesses."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        red_team = create_mock_team(Team.RED, state, "R")

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)

        # All guesses should be correct for our mock team
        assert metrics.red_metrics.guess_accuracy == 1.0
        assert metrics.red_metrics.correct_guesses == metrics.red_metrics.total_guesses


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases in metrics computation."""

    @pytest.mark.asyncio
    async def test_perfect_game_zero_wrong_guesses(self):
        """Episode with 0 wrong guesses should have perfect accuracy."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        red_team = create_mock_team(Team.RED, state, "R")

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)

        assert metrics.red_metrics.wrong_guesses == 0
        assert metrics.red_metrics.guess_accuracy == 1.0

    @pytest.mark.asyncio
    async def test_assassin_hit_turn_one(self):
        """Episode ending on turn 1 with assassin hit."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        assassin_word = list(state.board.key_by_category["assassin"])[0]

        cluer_config = AgentConfig(
            model="mock", role="cluer", team=Team.RED,
            agent_id="red_cluer", temperature=0.7,
        )
        guesser1_config = AgentConfig(
            model="mock", role="guesser", team=Team.RED,
            agent_id="red_guesser_1", temperature=0.7,
        )
        guesser2_config = AgentConfig(
            model="mock", role="guesser", team=Team.RED,
            agent_id="red_guesser_2", temperature=0.7,
        )

        cluer_provider = MockProvider(responses=["CLUE: DANGER\nNUMBER: 1\nREASONING: Bad."])
        guesser1_provider = MockProvider(responses=[
            f"Oops.\nCONSENSUS: YES\nTOP: {assassin_word}",
            f"GUESSES: {assassin_word}\nREASONING: Bad choice.",
        ])
        guesser2_provider = MockProvider(responses=["Sure.\nCONSENSUS: YES\nTOP: word"])

        red_team = TeamAgents(
            cluer=CluerAgent(cluer_config, cluer_provider),
            guesser_1=GuesserAgent(guesser1_config, guesser1_provider),
            guesser_2=GuesserAgent(guesser2_config, guesser2_provider),
        )

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=5,
        )

        metrics = compute_episode_metrics(episode)

        # Blue wins, Red hit assassin
        assert metrics.winner == Team.BLUE
        assert metrics.red_metrics.assassin_hit is True
        assert metrics.turns_to_win == 1

    @pytest.mark.asyncio
    async def test_no_consensus_achieved(self):
        """Episode with no consensus - all timeouts."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        red_words = list(state.board.key_by_category["red"])

        cluer_config = AgentConfig(
            model="mock", role="cluer", team=Team.RED,
            agent_id="red_cluer", temperature=0.7,
        )
        guesser1_config = AgentConfig(
            model="mock", role="guesser", team=Team.RED,
            agent_id="red_guesser_1", temperature=0.7,
        )
        guesser2_config = AgentConfig(
            model="mock", role="guesser", team=Team.RED,
            agent_id="red_guesser_2", temperature=0.7,
        )

        # Many unique clue words without digits
        clue_words = [
            "NOCONAA", "NOCONAB", "NOCONAC", "NOCONAD", "NOCONAE", "NOCONAF",
            "NOCONAG", "NOCONAH", "NOCONAI", "NOCONAJ", "NOCONAK", "NOCONAL",
            "NOCONAM", "NOCONAN", "NOCONAO", "NOCONAP", "NOCONAQ", "NOCONAR",
            "NOCONAS", "NOCONAT",
        ]
        cluer_responses = [
            f"CLUE: {w}\nNUMBER: 2\nREASONING: Test." for w in clue_words
        ]
        cluer_provider = MockProvider(responses=cluer_responses)

        # Guessers never agree on consensus in discussion
        # But guesser_1 still needs to make guesses after discussion ends
        discussion_responses = [
            f"I think word.\nCONSENSUS: NO\nTOP: {red_words[i % len(red_words)]}"
            for i in range(60)
        ]
        guess_responses = [
            f"GUESSES: {red_words[i % len(red_words)]}\nREASONING: Go."
            for i in range(20)
        ]

        # Interleave discussion and guess responses
        guesser1_responses = []
        for i in range(20):
            # 3 discussion messages per turn (max rounds)
            guesser1_responses.append(discussion_responses[i * 3] if i * 3 < len(discussion_responses) else discussion_responses[0])
            guesser1_responses.append(discussion_responses[i * 3 + 1] if i * 3 + 1 < len(discussion_responses) else discussion_responses[0])
            guesser1_responses.append(discussion_responses[i * 3 + 2] if i * 3 + 2 < len(discussion_responses) else discussion_responses[0])
            # Then a guess
            guesser1_responses.append(guess_responses[i % len(guess_responses)])

        guesser1_provider = MockProvider(responses=guesser1_responses)
        guesser2_provider = MockProvider(responses=[
            f"Disagree.\nCONSENSUS: NO\nTOP: {red_words[1]}" for _ in range(100)
        ])

        red_team = TeamAgents(
            cluer=CluerAgent(cluer_config, cluer_provider),
            guesser_1=GuesserAgent(guesser1_config, guesser1_provider),
            guesser_2=GuesserAgent(guesser2_config, guesser2_provider),
        )

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)

        # Consensus rate should be low (0 or very low)
        assert metrics.red_metrics.consensus_rate < 0.5


# ============================================================================
# Aggregate Metrics Tests
# ============================================================================

class TestAggregateMetrics:
    """Tests for aggregate metrics computation."""

    @pytest.mark.asyncio
    async def test_win_rates_sum_to_one(self):
        """Win rates should sum to 1.0 (including draws)."""
        # Use letter prefixes instead of R0, R1 (which contain digits)
        red_prefixes = ["WRRA", "WRRB", "WRRC"]
        blue_prefixes = ["WRBA", "WRBB", "WRBC"]

        episodes = []
        for i in range(3):
            state = create_game(config=GameConfig(seed=42 + i))
            ep = await run_episode(
                config=GameConfig(seed=42 + i),
                red_team=create_mock_team(Team.RED, state, red_prefixes[i]),
                blue_team=create_mock_team(Team.BLUE, state, blue_prefixes[i]),
                max_turns=30,
            )
            episodes.append(ep)

        aggregate = compute_aggregate_metrics(episodes)

        assert abs(aggregate.win_rate_red + aggregate.win_rate_blue + aggregate.draw_rate - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_averages_computed_correctly(self):
        """Averages should be computed correctly."""
        # Use letter prefixes
        prefixes = ["AVGA", "AVGB", "AVGC"]

        episodes = []
        for i in range(3):
            state = create_game(config=GameConfig(seed=42 + i))
            ep = await run_episode(
                config=GameConfig(seed=42 + i),
                red_team=create_mock_team(Team.RED, state, prefixes[i]),
                blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
                max_turns=20,
            )
            episodes.append(ep)

        aggregate = compute_aggregate_metrics(episodes)

        # Should have 3 episodes
        assert aggregate.episodes == 3

        # Avg turns should be reasonable
        assert aggregate.avg_turns_to_win > 0

    @pytest.mark.asyncio
    async def test_single_episode_input(self):
        """Should handle single-episode input."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        episode = await run_episode(
            config=config,
            red_team=create_mock_team(Team.RED, state, "R"),
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        aggregate = compute_aggregate_metrics([episode])

        assert aggregate.episodes == 1
        assert aggregate.win_rate_red == 1.0 or aggregate.win_rate_blue == 1.0

    def test_empty_episodes_list(self):
        """Should handle empty episodes list."""
        aggregate = compute_aggregate_metrics([])

        assert aggregate.episodes == 0
        assert aggregate.win_rate_red == 0.0
        assert aggregate.win_rate_blue == 0.0


# ============================================================================
# Coordination Score Tests
# ============================================================================

class TestCoordinationScore:
    """Tests for coordination score computation."""

    def test_coordination_score_components(self):
        """Coordination score should be weighted sum of components."""
        metrics = TeamMetrics(
            team=Team.RED,
            words_cleared=9,
            assassin_hit=False,
            total_clues=3,
            avg_clue_number=3.0,
            clue_efficiency=1.0,  # Perfect
            total_guesses=9,
            correct_guesses=9,
            wrong_guesses=0,
            opponent_guesses=0,
            neutral_guesses=0,
            guess_accuracy=1.0,  # Perfect
            avg_discussion_rounds=1.0,  # Fast consensus
            consensus_rate=1.0,  # Always consensus
            avg_discussion_length=100,
        )

        score = compute_coordination_score(metrics)

        # Perfect metrics should give high score (close to 1.0)
        assert score > 0.9

    def test_coordination_score_capped(self):
        """Discussion speed component should be capped at 1.0."""
        metrics = TeamMetrics(
            team=Team.RED,
            words_cleared=5,
            assassin_hit=False,
            total_clues=2,
            avg_clue_number=2.0,
            clue_efficiency=0.5,
            total_guesses=5,
            correct_guesses=5,
            wrong_guesses=0,
            opponent_guesses=0,
            neutral_guesses=0,
            guess_accuracy=1.0,
            avg_discussion_rounds=0.5,  # Very fast - should cap at 1.0
            consensus_rate=0.5,
            avg_discussion_length=50,
        )

        score = compute_coordination_score(metrics)

        # Score should be in reasonable range
        assert 0.0 <= score <= 1.0


# ============================================================================
# Export Tests
# ============================================================================

class TestExport:
    """Tests for metrics export functionality."""

    @pytest.mark.asyncio
    async def test_json_roundtrip(self):
        """JSON export should roundtrip correctly."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        episode = await run_episode(
            config=config,
            red_team=create_mock_team(Team.RED, state, "R"),
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)
        json_str = export_metrics(metrics, format="json")

        # Parse back
        data = json.loads(json_str)

        assert data["episode_id"] == metrics.episode_id
        assert data["turns_to_win"] == metrics.turns_to_win
        assert "red_metrics" in data
        assert "blue_metrics" in data

    @pytest.mark.asyncio
    async def test_csv_has_correct_headers(self):
        """CSV export should have correct headers."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        episode = await run_episode(
            config=config,
            red_team=create_mock_team(Team.RED, state, "CSVH"),
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)
        csv_str = export_metrics(metrics, format="csv")

        # Handle potential \r\n line endings
        lines = csv_str.replace("\r", "").strip().split("\n")
        headers = lines[0].split(",")

        assert "episode_id" in headers
        assert "winner" in headers
        assert "turns_to_win" in headers
        assert "red_words_cleared" in headers
        assert "blue_coordination_score" in headers

    @pytest.mark.asyncio
    async def test_markdown_renders_as_table(self):
        """Markdown export should render as table."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        episode = await run_episode(
            config=config,
            red_team=create_mock_team(Team.RED, state, "MDTBL"),
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=20,
        )

        metrics = compute_episode_metrics(episode)
        md_str = export_metrics(metrics, format="markdown")

        # Should contain table markers
        assert "|" in md_str
        assert "---" in md_str
        assert "Red" in md_str
        assert "Blue" in md_str
        assert "Coordination Score" in md_str

    @pytest.mark.asyncio
    async def test_aggregate_json_export(self):
        """Aggregate metrics should export to JSON."""
        # Use letter-only prefixes
        prefixes = ["AGJA", "AGJB"]

        episodes = []
        for i in range(2):
            state = create_game(config=GameConfig(seed=42 + i))
            ep = await run_episode(
                config=GameConfig(seed=42 + i),
                red_team=create_mock_team(Team.RED, state, prefixes[i]),
                blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
                max_turns=20,
            )
            episodes.append(ep)

        aggregate = compute_aggregate_metrics(episodes)
        json_str = export_metrics(aggregate, format="json")

        data = json.loads(json_str)

        assert data["episodes"] == 2
        assert "win_rate_red" in data
        assert "avg_coordination_score_red" in data

    @pytest.mark.asyncio
    async def test_aggregate_markdown_export(self):
        """Aggregate metrics should export to Markdown."""
        # Use letter-only prefixes
        prefixes = ["AGMA", "AGMB"]

        episodes = []
        for i in range(2):
            state = create_game(config=GameConfig(seed=42 + i))
            ep = await run_episode(
                config=GameConfig(seed=42 + i),
                red_team=create_mock_team(Team.RED, state, prefixes[i]),
                blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
                max_turns=20,
            )
            episodes.append(ep)

        aggregate = compute_aggregate_metrics(episodes)
        md_str = export_aggregate_markdown(aggregate)

        assert "# Aggregate Metrics" in md_str
        assert "Win Rates" in md_str
        assert "Red" in md_str

    @pytest.mark.asyncio
    async def test_multiple_episodes_csv(self):
        """Multiple episodes should export to CSV correctly."""
        # Use letter-only prefixes
        prefixes = ["MCVA", "MCVB", "MCVC"]

        episodes = []
        for i in range(3):
            state = create_game(config=GameConfig(seed=42 + i))
            ep = await run_episode(
                config=GameConfig(seed=42 + i),
                red_team=create_mock_team(Team.RED, state, prefixes[i]),
                blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
                max_turns=20,
            )
            episodes.append(ep)

        metrics_list = [compute_episode_metrics(ep) for ep in episodes]
        csv_str = export_metrics(metrics_list, format="csv")

        # Handle potential \r\n line endings
        lines = csv_str.replace("\r", "").strip().split("\n")
        # Header + 3 data rows
        assert len(lines) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
