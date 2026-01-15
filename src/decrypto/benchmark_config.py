from __future__ import annotations

from dataclasses import dataclass

from src.benchmark.config import ModelConfig, MatchupConfig, TeamAssignment, TeamComposition


@dataclass(frozen=True)
class DecryptoExperimentConfig:
    name: str
    models: list[ModelConfig]
    seeds: list[int]
    games_per_config: int = 1
    temperature: float = 0.7
    output_dir: str = "benchmark_results"


def generate_decrypto_matchups(models: list[ModelConfig]) -> list[MatchupConfig]:
    """
    Decrypto spec: for each unordered model pair (A,B), generate 4 configs:
      homog-A, homog-B, mixed-A-clue, mixed-B-clue

    No side counterbalancing: do NOT emit swapped duplicates.
    """
    model_by_id = {m.model_id: m for m in models}
    ids = sorted(model_by_id.keys())
    matchups: list[MatchupConfig] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            A = model_by_id[a_id]
            B = model_by_id[b_id]
            pair_key = f"{A.model_id}|{B.model_id}"

            def _mk(red: TeamAssignment, blue: TeamAssignment, config_type: str) -> MatchupConfig:
                return MatchupConfig(
                    red_team=red,
                    blue_team=blue,
                    composition=TeamComposition.HOMOGENEOUS,
                    pair_key=pair_key,
                    config_type=config_type,
                    direction=None,
                )

            # 1) homog-A: RED=A,A,A  BLUE=B,B,B
            matchups.append(
                _mk(
                    TeamAssignment(cluer=A, guesser_1=A, guesser_2=A),
                    TeamAssignment(cluer=B, guesser_1=B, guesser_2=B),
                    "homog-A",
                )
            )
            # 2) homog-B: RED=B,B,B  BLUE=A,A,A
            matchups.append(
                _mk(
                    TeamAssignment(cluer=B, guesser_1=B, guesser_2=B),
                    TeamAssignment(cluer=A, guesser_1=A, guesser_2=A),
                    "homog-B",
                )
            )
            # 3) mixed-A-clue: RED=(A,B,B) BLUE=(B,A,A)
            matchups.append(
                _mk(
                    TeamAssignment(cluer=A, guesser_1=B, guesser_2=B),
                    TeamAssignment(cluer=B, guesser_1=A, guesser_2=A),
                    "mixed-A-clue",
                )
            )
            # 4) mixed-B-clue: RED=(B,A,A) BLUE=(A,B,B)
            matchups.append(
                _mk(
                    TeamAssignment(cluer=B, guesser_1=A, guesser_2=A),
                    TeamAssignment(cluer=A, guesser_1=B, guesser_2=B),
                    "mixed-B-clue",
                )
            )

    return matchups

