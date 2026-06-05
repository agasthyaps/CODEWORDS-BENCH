from __future__ import annotations

import pytest

from src.agents import AgentConfig, CluerAgent, GuesserAgent, MockProvider
from src.benchmark import (
    BenchmarkConditionName,
    ExperimentConfig,
    ModelConfig,
    build_run_manifest,
    sparse_family_matchup_subset,
)
from src.core.state import AgentStateManager
from src.decrypto.models import (
    ActionLog,
    ClueSet,
    ConsensusGuess,
    GuesserIndependent,
    GuesserShare,
    RoundCounters,
    RoundLog,
    RoundStateTag,
)
from src.engine import GameConfig, GameMode, Team, create_game
from src.hanabi.game import apply_action, create_game as create_hanabi_game
from src.hanabi.models import DiscardAction, HanabiConfig
from src.hanabi.orchestrator import run_turn as run_hanabi_turn
from src.runner import TeamAgents, run_clue_phase, run_discussion_phase, run_episode


def _agent_config(role: str, team: Team, agent_id: str) -> AgentConfig:
    return AgentConfig(
        model="mock",
        role=role,
        team=team,
        agent_id=agent_id,
        temperature=0.0,
    )


def test_default_condition_is_human_table_scratchpad():
    config = ExperimentConfig(
        name="protocol-test",
        models=[
            ModelConfig(name="a", model_id="a"),
            ModelConfig(name="b", model_id="b"),
        ],
    )

    assert config.condition.name == BenchmarkConditionName.HUMAN_TABLE_SCRATCHPAD
    assert config.condition.scratchpad_enabled
    assert config.condition.invalid_action_policy == "human_table"


def test_run_manifest_records_reproducibility_fields():
    manifest = build_run_manifest(
        game_type="codenames",
        models={"red": "mock-a", "blue": "mock-b"},
        provider_parameters={"temperature": 0.0},
        seed_schedule=[1, 2, 3],
    )

    assert manifest.condition.name == BenchmarkConditionName.HUMAN_TABLE_SCRATCHPAD
    assert manifest.game_type == "codenames"
    assert manifest.seed_schedule == [1, 2, 3]
    assert manifest.prompt_hashes
    assert "temperature" in manifest.provider_parameters


def test_sparse_family_matchup_subset_is_bounded():
    models = [
        ModelConfig(name="gpt", model_id="openai/gpt", family="openai"),
        ModelConfig(name="claude", model_id="anthropic/claude", family="anthropic"),
        ModelConfig(name="gemini", model_id="google/gemini", family="google"),
        ModelConfig(name="qwen", model_id="qwen/qwen", family="open_weight"),
        ModelConfig(name="other-openai", model_id="openai/other", family="openai"),
    ]

    subset = sparse_family_matchup_subset(models)

    assert subset == [
        ("openai/gpt", "anthropic/claude"),
        ("openai/gpt", "google/gemini"),
        ("anthropic/claude", "google/gemini"),
        ("qwen/qwen", "openai/gpt"),
        ("qwen/qwen", "anthropic/claude"),
    ]


def test_decrypto_round_log_retains_round_state():
    counters = {
        "red": RoundCounters(
            own_interceptions=0,
            own_miscommunications=0,
            opp_interceptions=0,
            opp_miscommunications=0,
        ),
        "blue": RoundCounters(
            own_interceptions=0,
            own_miscommunications=0,
            opp_interceptions=0,
            opp_miscommunications=0,
        ),
    }
    state_tags = {
        "red": RoundStateTag(interceptions_state="tied", danger=False),
        "blue": RoundStateTag(interceptions_state="tied", danger=False),
    }
    independent = GuesserIndependent(
        agent_id="red_guesser_1",
        guess=(1, 2, 3),
        confidence=0.5,
        rationale="",
        parse_ok=True,
    )
    share = GuesserShare(agent_id="red_guesser_1", message="")
    consensus = ConsensusGuess(
        captain_id="red_guesser_1",
        guess=(1, 2, 3),
        confidence=0.5,
        rationale="",
        parse_ok=True,
    )
    action = ActionLog(
        kind="decode",
        team="red",
        opponent_team="blue",
        independent=(independent, independent),
        share=(share, share),
        consensus=consensus,
        correct=True,
    )

    round_log = RoundLog(
        round_number=1,
        counters_before=counters,
        counters_after=counters,
        round_state_at_clue_time=state_tags,
        public_clues={
            "red": ClueSet(clues=("A", "B", "C")),
            "blue": ClueSet(clues=("D", "E", "F")),
        },
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(action, action, action, action),
        round_state={"red": {"decode_consensus": {"guess": (1, 2, 3)}}},
    )

    assert round_log.round_state["red"]["decode_consensus"]["guess"] == (1, 2, 3)


@pytest.mark.asyncio
async def test_codenames_discussion_scratchpads_persist():
    state = create_game(config=GameConfig(seed=42))
    agent_states = AgentStateManager()
    cluer = CluerAgent(
        _agent_config("cluer", Team.RED, "red_cluer"),
        MockProvider(
            [
                "CLUE: UMBRELLA\nNUMBER: 1\nREASONING: test\nSCRATCHPAD: remember umbrella"
            ]
        ),
    )
    guesser_1 = GuesserAgent(
        _agent_config("guesser", Team.RED, "red_guesser_1"),
        MockProvider(
            [
                "I like one option.\nCONSENSUS: YES\nTOP: "
                f"{state.board.words[0]}\nSCRATCHPAD: g1 note"
            ]
        ),
    )
    guesser_2 = GuesserAgent(
        _agent_config("guesser", Team.RED, "red_guesser_2"),
        MockProvider(
            [
                "Agreed.\nCONSENSUS: YES\nTOP: "
                f"{state.board.words[0]}\nSCRATCHPAD: g2 note"
            ]
        ),
    )

    state, _, _ = await run_clue_phase(cluer, state, agent_states)
    state, _ = await run_discussion_phase(
        [guesser_1, guesser_2], state, max_rounds=1, agent_states=agent_states
    )

    scratchpads = {
        agent_id: agent_state.scratchpad
        for agent_id, agent_state in agent_states.get_all_states().items()
    }
    assert "remember umbrella" in scratchpads["red_cluer"]
    assert "g1 note" in scratchpads["red_guesser_1"]
    assert "g2 note" in scratchpads["red_guesser_2"]


@pytest.mark.asyncio
async def test_raw_chat_condition_disables_saved_scratchpads():
    state = create_game(config=GameConfig.for_mode(GameMode.SINGLE_GUESSER, seed=42))
    red_team = TeamAgents(
        cluer=CluerAgent(
            _agent_config("cluer", Team.RED, "red_cluer"),
            MockProvider(
                [
                    "CLUE: UMBRELLA\nNUMBER: 1\nREASONING: test\nSCRATCHPAD: should not persist"
                ]
            ),
        ),
        guesser_1=GuesserAgent(
            _agent_config("guesser", Team.RED, "red_guesser_1"),
            MockProvider(["GUESSES: PASS\nREASONING: done\nSCRATCHPAD: skip"]),
        ),
        guesser_2=None,
    )
    blue_team = red_team

    episode = await run_episode(
        config=state.config,
        red_team=red_team,
        blue_team=blue_team,
        max_turns=1,
        condition=BenchmarkConditionName.RAW_CHAT,
    )

    assert episode.agent_scratchpads == {}
    assert episode.run_manifest is not None
    assert episode.run_manifest.condition.name == BenchmarkConditionName.RAW_CHAT


def test_hanabi_human_table_invalid_action_consumes_turn():
    state = create_hanabi_game(HanabiConfig(num_players=3, seed=42))
    assert state.hint_tokens == state.config.max_hints

    new_state, result, turn_log = apply_action(
        state,
        "player_1",
        DiscardAction(card_position=0),
        invalid_policy="human_table",
    )

    assert not result.success
    assert "maximum hint" in result.message
    assert turn_log.result.success is False
    assert len(new_state.action_history) == 1
    assert new_state.current_player == "player_2"
    assert new_state.turn_number == 2


@pytest.mark.asyncio
async def test_hanabi_orchestrator_logs_invalid_action_penalty():
    class InvalidDiscardPlayer:
        player_id = "player_1"

        async def decide_action(self, state, scratchpad):
            return DiscardAction(card_position=0), "invalid at max hints", None

    state = create_hanabi_game(HanabiConfig(num_players=3, seed=42))
    events = []
    penalties = []

    new_state, turn_log, _ = await run_hanabi_turn(
        state,
        InvalidDiscardPlayer(),
        AgentStateManager(),
        condition=None,
        harness_events=events,
        benchmark_penalties=penalties,
    )

    assert not turn_log.result.success
    assert new_state.current_player == "player_2"
    assert {event.event_type for event in events} >= {
        "invalid_action_proposed",
        "human_table_correction_applied",
    }
    assert [penalty.penalty_type for penalty in penalties] == [
        "invalid_action",
        "moderator_correction",
    ]
