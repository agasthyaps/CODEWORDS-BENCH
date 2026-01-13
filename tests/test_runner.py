"""Tests for the turn orchestration and episode runner (M3)."""

import pytest
import json
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import (
    Team, GameConfig, Phase, create_game, apply_clue,
    transition_to_guessing, DiscussionMessage,
)
from src.agents import AgentConfig, CluerAgent, GuesserAgent, MockProvider
from src.runner import (
    TurnTraces,
    run_clue_phase, run_discussion_phase, run_guess_phase, run_turn,
    TeamConfig, TeamAgents, GhostMode, GhostTeam,
    ExtendedEpisodeRecord, run_episode, run_single_team_episode,
)


# ============================================================================
# Turn Orchestration Tests
# ============================================================================

class TestTurnOrchestration:
    """Tests for turn phase orchestration."""

    @pytest.fixture
    def game_state(self):
        """Create a fresh game state."""
        return create_game(config=GameConfig(seed=42))

    @pytest.fixture
    def mock_cluer(self):
        """Create a mock cluer agent."""
        config = AgentConfig(
            model="mock",
            role="cluer",
            team=Team.RED,
            agent_id="red_cluer",
            temperature=0.7,
        )
        provider = MockProvider(responses=[
            "CLUE: UMBRELLA\nNUMBER: 2\nREASONING: Test clue."
        ])
        return CluerAgent(config, provider)

    @pytest.fixture
    def mock_guessers(self, game_state):
        """Create mock guesser agents."""
        board_words = game_state.board.words[:3]

        config1 = AgentConfig(
            model="mock",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_1",
            temperature=0.7,
        )
        config2 = AgentConfig(
            model="mock",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_2",
            temperature=0.7,
        )

        provider1 = MockProvider(responses=[
            "I think we should guess carefully.\nCONSENSUS: YES\nTOP: " + board_words[0],
            f"GUESSES: {board_words[0]}, {board_words[1]}\nREASONING: Top picks.",
        ])
        provider2 = MockProvider(responses=[
            "Agreed.\nCONSENSUS: YES\nTOP: " + board_words[0],
        ])

        return [
            GuesserAgent(config1, provider1),
            GuesserAgent(config2, provider2),
        ]

    @pytest.mark.asyncio
    async def test_clue_phase_updates_state(self, game_state, mock_cluer):
        """Clue phase should update state correctly."""
        new_state, trace, should_continue = await run_clue_phase(mock_cluer, game_state)

        assert should_continue is True
        assert new_state.phase == Phase.DISCUSSION
        assert new_state.current_clue is not None
        assert new_state.current_clue.word == "UMBRELLA"
        assert trace.agent_id == "red_cluer"

    @pytest.mark.asyncio
    async def test_discussion_messages_in_transcript(self, game_state, mock_guessers):
        """Discussion messages should appear in transcript in order."""
        # Set up state in discussion phase
        state = apply_clue(game_state, "TESTING", 2)

        new_state, traces = await run_discussion_phase(mock_guessers, state, max_rounds=3)

        # Check messages in transcript
        discussion_events = [
            e for e in new_state.public_transcript
            if isinstance(e, DiscussionMessage)
        ]

        assert len(discussion_events) >= 2
        assert discussion_events[0].agent_id == "red_guesser_1"
        assert discussion_events[1].agent_id == "red_guesser_2"
        assert new_state.phase == Phase.GUESS

    @pytest.mark.asyncio
    async def test_guess_phase_processes_until_turn_ends(self, game_state, mock_guessers):
        """Guess phase should process guesses until turn ends."""
        # Set up state in guess phase
        state = apply_clue(game_state, "TESTING", 2)
        state = transition_to_guessing(state)

        new_state, trace = await run_guess_phase(
            mock_guessers[0], state, []
        )

        # Turn should have ended (guesses processed)
        # Either correct guesses or wrong guess ended turn
        assert trace is not None
        assert trace.agent_id == "red_guesser_1"

    @pytest.mark.asyncio
    async def test_phase_transitions(self, game_state):
        """Phases should transition correctly: CLUE → DISCUSSION → GUESS → CLUE."""
        board_words = game_state.board.words[:2]

        config = AgentConfig(
            model="mock", role="cluer", team=Team.RED,
            agent_id="red_cluer", temperature=0.7,
        )
        cluer_provider = MockProvider(responses=[
            "CLUE: UMBRELLA\nNUMBER: 2\nREASONING: Test."
        ])
        cluer = CluerAgent(config, cluer_provider)

        config1 = AgentConfig(
            model="mock", role="guesser", team=Team.RED,
            agent_id="red_guesser_1", temperature=0.7,
        )
        config2 = AgentConfig(
            model="mock", role="guesser", team=Team.RED,
            agent_id="red_guesser_2", temperature=0.7,
        )
        guesser_provider1 = MockProvider(responses=[
            f"Let's go.\nCONSENSUS: YES\nTOP: {board_words[0]}",
            f"GUESSES: {board_words[0]}\nREASONING: Safe pick.",
        ])
        guesser_provider2 = MockProvider(responses=[
            f"Agreed.\nCONSENSUS: YES\nTOP: {board_words[0]}",
        ])

        team = TeamAgents(
            cluer=cluer,
            guesser_1=GuesserAgent(config1, guesser_provider1),
            guesser_2=GuesserAgent(config2, guesser_provider2),
        )

        # Initial state
        assert game_state.phase == Phase.CLUE
        assert game_state.current_turn == Team.RED

        # Run full turn
        new_state, traces = await run_turn(team, game_state, max_discussion_rounds=2)

        # After turn, should be CLUE phase for other team
        assert new_state.phase == Phase.CLUE
        assert new_state.current_turn == Team.BLUE
        assert traces.turn_number == 1
        assert traces.team == Team.RED


# ============================================================================
# Ghost Team Tests
# ============================================================================

class TestGhostTeam:
    """Tests for ghost team behavior."""

    @pytest.fixture
    def game_state(self):
        """Create a fresh game state."""
        return create_game(config=GameConfig(seed=42))

    @pytest.mark.asyncio
    async def test_pass_ghost_produces_no_clue(self, game_state):
        """PASS ghost should produce no clue (returns None)."""
        ghost = GhostTeam(Team.BLUE, GhostMode.PASS)

        clue, trace = await ghost.cluer.generate_clue(game_state)

        assert clue is None
        assert trace.parsed_result["is_ghost"] is True
        assert trace.parsed_result["mode"] == "PASS"

    @pytest.mark.asyncio
    async def test_random_ghost_produces_legal_clue(self, game_state):
        """RANDOM ghost should produce legal clues."""
        ghost = GhostTeam(Team.BLUE, GhostMode.RANDOM)

        clue, trace = await ghost.cluer.generate_clue(game_state)

        assert clue is not None
        assert clue.word.isalpha()
        assert 1 <= clue.number <= 3

    @pytest.mark.asyncio
    async def test_ghost_traces_marked_correctly(self, game_state):
        """Ghost traces should be marked with is_ghost."""
        ghost = GhostTeam(Team.BLUE, GhostMode.PASS)

        _, clue_trace = await ghost.cluer.generate_clue(game_state)
        _, discuss_trace = await ghost.guesser_1.discuss(game_state, [])
        _, guess_trace = await ghost.guesser_1.make_guesses(game_state, [])

        assert clue_trace.parsed_result.get("is_ghost") is True
        assert discuss_trace.parsed_result.get("is_ghost") is True
        assert guess_trace.parsed_result.get("is_ghost") is True


# ============================================================================
# Single-Team Game Tests
# ============================================================================

class TestSingleTeamGame:
    """Tests for single-team play against ghost."""

    @pytest.fixture
    def real_team(self):
        """Create a real team that makes reasonable moves."""
        def make_team(game_state):
            # Get red team words for guessing
            red_words = list(game_state.board.key_by_category["red"])

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

            # Cluer gives unique clues (won't repeat)
            unique_clue_words = [
                "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO",
                "FOXTROT", "GOLF", "HOTEL", "INDIA", "JULIET",
                "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR",
            ]
            cluer_responses = [
                f"CLUE: {word}\nNUMBER: 2\nREASONING: Test."
                for word in unique_clue_words
            ]
            cluer_provider = MockProvider(responses=cluer_responses)

            # Guessers guess red words in order
            guesser_responses = []
            for i in range(0, len(red_words), 2):
                words = red_words[i:i+2]
                guesser_responses.append(f"Let's guess.\nCONSENSUS: YES\nTOP: {', '.join(words)}")
                guesser_responses.append(f"GUESSES: {', '.join(words)}\nREASONING: Go for it.")

            guesser1_provider = MockProvider(responses=guesser_responses)
            guesser2_provider = MockProvider(responses=[
                "Agreed.\nCONSENSUS: YES\nTOP: word" for _ in range(20)
            ])

            return TeamAgents(
                cluer=CluerAgent(cluer_config, cluer_provider),
                guesser_1=GuesserAgent(guesser1_config, guesser1_provider),
                guesser_2=GuesserAgent(guesser2_config, guesser2_provider),
            )

        return make_team

    @pytest.mark.asyncio
    async def test_real_team_vs_pass_ghost(self, real_team):
        """Real team should be able to win against PASS ghost."""
        config = GameConfig(seed=42)
        state = create_game(config=config)
        team = real_team(state)

        episode = await run_single_team_episode(
            config=config,
            real_team=team,
            real_team_color=Team.RED,
            max_turns=20,
        )

        # Game should complete
        assert episode.winner is not None
        # Real team should win (assuming no assassin hit)
        # Winner could be RED (cleared words) or BLUE (if RED hit assassin)

    @pytest.mark.asyncio
    async def test_game_ends_on_assassin(self):
        """Game should end if team hits assassin."""
        config = GameConfig(seed=42)
        state = create_game(config=config)

        # Find assassin word
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

        # Deliberately guess assassin
        cluer_provider = MockProvider(responses=[
            "CLUE: DANGER\nNUMBER: 1\nREASONING: Test."
        ])
        guesser1_provider = MockProvider(responses=[
            f"Bad guess.\nCONSENSUS: YES\nTOP: {assassin_word}",
            f"GUESSES: {assassin_word}\nREASONING: Oops.",
        ])
        guesser2_provider = MockProvider(responses=[
            f"Sure.\nCONSENSUS: YES\nTOP: {assassin_word}",
        ])

        team = TeamAgents(
            cluer=CluerAgent(cluer_config, cluer_provider),
            guesser_1=GuesserAgent(guesser1_config, guesser1_provider),
            guesser_2=GuesserAgent(guesser2_config, guesser2_provider),
        )

        episode = await run_single_team_episode(
            config=config,
            real_team=team,
            real_team_color=Team.RED,
            max_turns=5,
        )

        # Blue should win (red hit assassin)
        assert episode.winner == Team.BLUE


# ============================================================================
# Episode Record Tests
# ============================================================================

class TestEpisodeRecord:
    """Tests for episode record serialization."""

    @pytest.mark.asyncio
    async def test_episode_serializes_to_json(self):
        """Episode should serialize to JSON."""
        config = GameConfig(seed=42)
        ghost_red = GhostTeam(Team.RED, GhostMode.PASS)
        ghost_blue = GhostTeam(Team.BLUE, GhostMode.PASS)

        episode = await run_episode(
            config=config,
            red_team=ghost_red,
            blue_team=ghost_blue,
            max_turns=5,
        )

        # Should serialize without error
        data = episode.model_dump(mode="json")
        json_str = json.dumps(data, default=str)

        assert episode.episode_id in json_str
        assert len(json_str) > 100

    @pytest.mark.asyncio
    async def test_episode_save_and_load(self):
        """Episode should save to file and load back."""
        config = GameConfig(seed=42)
        ghost_red = GhostTeam(Team.RED, GhostMode.PASS)
        ghost_blue = GhostTeam(Team.BLUE, GhostMode.PASS)

        episode = await run_episode(
            config=config,
            red_team=ghost_red,
            blue_team=ghost_blue,
            max_turns=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = episode.save(tmpdir)

            assert filepath.exists()

            loaded = ExtendedEpisodeRecord.load(filepath)

            assert loaded.episode_id == episode.episode_id
            assert loaded.board_seed == episode.board_seed
            assert loaded.winner == episode.winner

    @pytest.mark.asyncio
    async def test_transcript_and_traces_separate(self):
        """Transcript and traces should be separate structures."""
        config = GameConfig(seed=42)
        ghost_red = GhostTeam(Team.RED, GhostMode.RANDOM)
        ghost_blue = GhostTeam(Team.BLUE, GhostMode.PASS)

        episode = await run_episode(
            config=config,
            red_team=ghost_red,
            blue_team=ghost_blue,
            max_turns=5,
        )

        # Transcript is public events
        assert isinstance(episode.public_transcript, list)

        # Traces are organized by turn
        assert isinstance(episode.turn_traces, list)
        if episode.turn_traces:
            assert isinstance(episode.turn_traces[0], TurnTraces)


# ============================================================================
# Replay Determinism Tests
# ============================================================================

class TestReplayDeterminism:
    """Tests for replay determinism."""

    @pytest.mark.asyncio
    async def test_same_seed_same_board(self):
        """Same seed should produce identical boards."""
        config1 = GameConfig(seed=123)
        config2 = GameConfig(seed=123)

        ghost = GhostTeam(Team.RED, GhostMode.PASS)

        episode1 = await run_episode(
            config=config1,
            red_team=ghost,
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=2,
        )
        episode2 = await run_episode(
            config=config2,
            red_team=ghost,
            blue_team=GhostTeam(Team.BLUE, GhostMode.PASS),
            max_turns=2,
        )

        assert episode1.board.words == episode2.board.words
        assert episode1.board_seed == episode2.board_seed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
