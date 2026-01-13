"""Tests for two-team adversarial play (M4)."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import (
    Team, GameConfig, GameMode, Phase, create_game,
    DiscussionMessage,
)
from src.agents import AgentConfig, CluerAgent, GuesserAgent, MockProvider
from src.runner import (
    TeamConfig, TeamAgents, GhostTeam, GhostMode,
    ExtendedEpisodeRecord, run_episode,
)


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

    # Unique clue words
    clue_words = [
        f"{unique_prefix}ALPHA", f"{unique_prefix}BRAVO", f"{unique_prefix}CHARLIE",
        f"{unique_prefix}DELTA", f"{unique_prefix}ECHO", f"{unique_prefix}FOXTROT",
        f"{unique_prefix}GOLF", f"{unique_prefix}HOTEL", f"{unique_prefix}INDIA",
        f"{unique_prefix}JULIET", f"{unique_prefix}KILO", f"{unique_prefix}LIMA",
    ]
    cluer_responses = [f"CLUE: {w}\nNUMBER: 2\nREASONING: Test." for w in clue_words]
    cluer_provider = MockProvider(responses=cluer_responses)

    # Guessers guess their words
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
# Two-Team Game Tests
# ============================================================================

class TestTwoTeamGame:
    """Tests for two-team adversarial play."""

    @pytest.mark.asyncio
    async def test_both_teams_take_turns(self):
        """Both teams should take turns."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        red_team = create_mock_team(Team.RED, state, "R")
        blue_team = create_mock_team(Team.BLUE, state, "B")

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=blue_team,
            max_turns=6,
        )

        # Check that both teams took turns
        red_turns = [t for t in episode.turn_traces if t.team == Team.RED]
        blue_turns = [t for t in episode.turn_traces if t.team == Team.BLUE]

        assert len(red_turns) >= 1
        assert len(blue_turns) >= 1

    @pytest.mark.asyncio
    async def test_game_completes_with_winner(self):
        """Game should complete with a winner."""
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

        assert episode.winner is not None
        assert episode.winner in [Team.RED, Team.BLUE]

    @pytest.mark.asyncio
    async def test_turn_count_reasonable(self):
        """Turn count should be reasonable (not infinite loops)."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        red_team = create_mock_team(Team.RED, state, "R")
        blue_team = create_mock_team(Team.BLUE, state, "B")

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=blue_team,
            max_turns=50,
        )

        # Should complete in reasonable number of turns
        assert episode.total_turns < 50
        assert episode.total_turns >= 1


# ============================================================================
# Visibility Enforcement Tests
# ============================================================================

class TestVisibilityEnforcement:
    """Tests for visibility rules enforcement."""

    @pytest.mark.asyncio
    async def test_guesser_visible_state_has_no_key(self):
        """Guesser's visible_state should not contain key field."""
        from src.engine import get_visible_state

        state = create_game(config=GameConfig(seed=42))

        guesser_visible = get_visible_state(state, "red_guesser_1")

        # Key should be absent, not null
        assert "key" not in guesser_visible

    @pytest.mark.asyncio
    async def test_cluer_visible_state_has_key(self):
        """Cluer's visible_state should contain key field."""
        from src.engine import get_visible_state

        state = create_game(config=GameConfig(seed=42))

        cluer_visible = get_visible_state(state, "red_cluer")

        assert "key" in cluer_visible
        assert "red" in cluer_visible["key"]
        assert "blue" in cluer_visible["key"]

    @pytest.mark.asyncio
    async def test_cluer_sees_opponent_discussion(self):
        """Cluer's visible_state should include opponent's discussion."""
        from src.engine import get_visible_state, apply_clue, add_discussion_message

        state = create_game(config=GameConfig(seed=42))

        # Red gives clue and guesses
        state = apply_clue(state, "TESTING", 2)
        state = add_discussion_message(state, "red_guesser_1", "I think WORD1 works")
        state = add_discussion_message(state, "red_guesser_2", "Agreed")

        # Now it's Blue's turn - their cluer should see Red's discussion
        # (In reality, turn would have switched, but transcript is always visible)
        blue_cluer_visible = get_visible_state(state, "blue_cluer")

        # Check transcript contains Red's discussion
        transcript = blue_cluer_visible["public_transcript"]
        discussion_events = [e for e in transcript if e.get("event_type") == "discussion"]

        assert len(discussion_events) >= 2
        assert any("red_guesser" in e.get("agent_id", "") for e in discussion_events)


# ============================================================================
# Game Mode Tests
# ============================================================================

class TestGameModes:
    """Tests for different game modes."""

    @pytest.mark.asyncio
    async def test_standard_mode_assassin_ends_game(self):
        """In STANDARD mode, hitting assassin should end game."""
        config = GameConfig.for_mode(GameMode.STANDARD, seed=42)
        state = create_game(config=config)

        # Find assassin word
        assassin_word = list(state.board.key_by_category["assassin"])[0]

        # Create team that will hit assassin
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

        cluer_provider = MockProvider(responses=["CLUE: DANGER\nNUMBER: 1\nREASONING: Test."])
        guesser1_provider = MockProvider(responses=[
            f"Bad choice.\nCONSENSUS: YES\nTOP: {assassin_word}",
            f"GUESSES: {assassin_word}\nREASONING: Oops.",
        ])
        guesser2_provider = MockProvider(responses=[f"Sure.\nCONSENSUS: YES\nTOP: {assassin_word}"])

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

        # Blue wins because Red hit assassin
        assert episode.winner == Team.BLUE

    @pytest.mark.asyncio
    async def test_no_assassin_mode_distribution(self):
        """NO_ASSASSIN mode should have correct distribution."""
        config = GameConfig.for_mode(GameMode.NO_ASSASSIN, seed=42)

        assert config.red_count == 9
        assert config.blue_count == 8
        assert config.neutral_count == 8
        assert config.assassin_count == 0

        state = create_game(config=config)

        assert len(state.board.key_by_category["red"]) == 9
        assert len(state.board.key_by_category["blue"]) == 8
        assert len(state.board.key_by_category["neutral"]) == 8
        assert len(state.board.key_by_category["assassin"]) == 0

    @pytest.mark.asyncio
    async def test_single_guesser_mode_no_discussion(self):
        """SINGLE_GUESSER mode should skip discussion phase."""
        config = GameConfig.for_mode(GameMode.SINGLE_GUESSER, seed=42)
        state = create_game(config=config)

        red_team = create_mock_team(Team.RED, state, "R")
        blue_team = GhostTeam(Team.BLUE, GhostMode.PASS)

        episode = await run_episode(
            config=config,
            red_team=red_team,
            blue_team=blue_team,
            max_turns=10,
        )

        # Check that there are no discussion traces
        for turn_trace in episode.turn_traces:
            if turn_trace.team == Team.RED:
                assert len(turn_trace.discussion_traces) == 0

        # Check no discussion events in transcript
        discussion_events = [
            e for e in episode.public_transcript
            if e.get("event_type") == "discussion"
        ]
        # Only ghost discussion messages (if any) - real team should have none
        red_discussions = [
            e for e in discussion_events
            if "red" in e.get("agent_id", "").lower()
        ]
        assert len(red_discussions) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_team_clears_on_bonus_guess(self):
        """Team should be able to clear all words on bonus guess."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        # Get red words - we'll have team guess all of them
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

        # Give clue that allows many guesses
        cluer_provider = MockProvider(responses=[
            "CLUE: EVERYTHING\nNUMBER: 9\nREASONING: Go for all."
        ])
        # Guess all red words
        guesser1_provider = MockProvider(responses=[
            f"All of them.\nCONSENSUS: YES\nTOP: {', '.join(red_words)}",
            f"GUESSES: {', '.join(red_words)}\nREASONING: Win!",
        ])
        guesser2_provider = MockProvider(responses=[
            f"Go for it.\nCONSENSUS: YES\nTOP: {', '.join(red_words)}"
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
            max_turns=5,
        )

        # Red should win by clearing all words
        assert episode.winner == Team.RED
        assert episode.total_turns == 1  # Won on first turn


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
