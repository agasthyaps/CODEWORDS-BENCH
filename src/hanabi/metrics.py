"""Metrics calculation for Hanabi games."""

from __future__ import annotations

from typing import Any

from .models import (
    HanabiEpisodeRecord,
    HanabiState,
    HintAction,
    PlayAction,
    DiscardAction,
    TurnLog,
)


def compute_episode_metrics(episode: HanabiEpisodeRecord) -> dict[str, Any]:
    """
    Compute metrics for a completed Hanabi episode.
    
    Returns dict with:
    - score: Final score (0-25)
    - score_percentage: Score as percentage of max (25)
    - total_turns: Number of turns played
    - hints_given: Total hint actions
    - plays_attempted: Total play actions
    - plays_successful: Successful plays
    - plays_failed: Failed plays (fuse losses)
    - discards: Total discard actions
    - hint_efficiency: Plays per hint given
    - critical_discards: Discards of critical cards
    - fuses_lost: Number of fuse tokens lost
    - per_player: Per-player breakdown
    """
    turns = episode.turns
    
    # Count action types
    hints_given = 0
    plays_attempted = 0
    plays_successful = 0
    plays_failed = 0
    discards = 0
    
    # Per-player stats
    per_player: dict[str, dict[str, int]] = {}
    for pid in episode.config.num_players * ["player_"]:
        # Initialize later when we see actual player IDs
        pass
    
    for turn in turns:
        pid = turn.player_id
        if pid not in per_player:
            per_player[pid] = {
                "hints": 0,
                "plays": 0,
                "plays_successful": 0,
                "plays_failed": 0,
                "discards": 0,
            }
        
        action = turn.action
        if isinstance(action, HintAction):
            hints_given += 1
            per_player[pid]["hints"] += 1
        elif isinstance(action, PlayAction):
            plays_attempted += 1
            per_player[pid]["plays"] += 1
            if turn.result.was_playable:
                plays_successful += 1
                per_player[pid]["plays_successful"] += 1
            else:
                plays_failed += 1
                per_player[pid]["plays_failed"] += 1
        elif isinstance(action, DiscardAction):
            discards += 1
            per_player[pid]["discards"] += 1
    
    # Compute derived metrics
    hint_efficiency = plays_successful / hints_given if hints_given > 0 else 0.0
    play_success_rate = plays_successful / plays_attempted if plays_attempted > 0 else 0.0
    
    return {
        "score": episode.final_score,
        "score_percentage": round(episode.final_score / 25 * 100, 1),
        "max_possible_score": 25,
        "total_turns": len(turns),
        "game_over_reason": episode.game_over_reason,
        
        # Action counts
        "hints_given": hints_given,
        "plays_attempted": plays_attempted,
        "plays_successful": plays_successful,
        "plays_failed": plays_failed,
        "discards": discards,
        
        # Derived metrics
        "hint_efficiency": round(hint_efficiency, 3),
        "play_success_rate": round(play_success_rate, 3),
        "fuses_lost": plays_failed,
        
        # Per-color breakdown
        "stacks_completed": sum(1 for v in episode.final_played_cards.values() if v == 5),
        "per_color": {color: num for color, num in episode.final_played_cards.items()},
        
        # Per-player breakdown
        "per_player": per_player,
    }


def compute_hint_utilization(episode: HanabiEpisodeRecord) -> dict[str, Any]:
    """
    Analyze how hints were used in the game.
    
    Tracks:
    - Hints that led to plays within N turns
    - Hints that led to discards
    - "Wasted" hints (no visible follow-up)
    """
    turns = episode.turns
    
    hint_outcomes: list[dict[str, Any]] = []
    
    for i, turn in enumerate(turns):
        if not isinstance(turn.action, HintAction):
            continue
        
        hint_action = turn.action
        target = hint_action.target_player
        
        # Look for follow-up actions by the target player
        follow_up_found = False
        follow_up_turn = None
        follow_up_type = None
        
        for j in range(i + 1, min(i + 10, len(turns))):  # Look ahead up to 10 turns
            future_turn = turns[j]
            if future_turn.player_id == target:
                follow_up_turn = j - i
                if isinstance(future_turn.action, PlayAction):
                    follow_up_type = "play"
                    follow_up_found = True
                    break
                elif isinstance(future_turn.action, DiscardAction):
                    follow_up_type = "discard"
                    follow_up_found = True
                    break
        
        hint_outcomes.append({
            "turn": i,
            "hinter": turn.player_id,
            "target": target,
            "hint_type": hint_action.hint_type,
            "hint_value": hint_action.hint_value,
            "follow_up_found": follow_up_found,
            "follow_up_turns": follow_up_turn,
            "follow_up_type": follow_up_type,
        })
    
    # Summarize
    total_hints = len(hint_outcomes)
    hints_with_follow_up = sum(1 for h in hint_outcomes if h["follow_up_found"])
    hints_to_plays = sum(1 for h in hint_outcomes if h["follow_up_type"] == "play")
    hints_to_discards = sum(1 for h in hint_outcomes if h["follow_up_type"] == "discard")
    
    return {
        "total_hints": total_hints,
        "hints_with_follow_up": hints_with_follow_up,
        "hints_to_plays": hints_to_plays,
        "hints_to_discards": hints_to_discards,
        "utilization_rate": round(hints_with_follow_up / total_hints, 3) if total_hints > 0 else 0.0,
        "details": hint_outcomes,
    }


def score_category(score: int) -> str:
    """Categorize a Hanabi score."""
    if score == 25:
        return "perfect"
    elif score >= 21:
        return "excellent"
    elif score >= 16:
        return "good"
    elif score >= 11:
        return "mediocre"
    elif score >= 6:
        return "poor"
    else:
        return "terrible"


def compute_coordination_metrics(episode: HanabiEpisodeRecord) -> dict[str, Any]:
    """
    Compute metrics related to agent coordination and theory of mind.
    
    Looks at:
    - Hint interpretation accuracy (did target act on hint correctly?)
    - Play confidence (did agents play known-safe cards?)
    - Communication efficiency
    """
    turns = episode.turns
    
    # Track hint-to-action chains
    hint_chains: list[dict[str, Any]] = []
    
    for i, turn in enumerate(turns):
        if not isinstance(turn.action, HintAction):
            continue
        
        hint = turn.action
        target = hint.target_player
        
        # Find target's next action
        for j in range(i + 1, len(turns)):
            if turns[j].player_id == target:
                next_action = turns[j]
                
                chain = {
                    "hint_turn": i,
                    "action_turn": j,
                    "gap": j - i,
                    "hint_type": hint.hint_type,
                    "hint_value": hint.hint_value,
                    "action_type": next_action.action.action_type,
                    "action_successful": None,
                }
                
                if isinstance(next_action.action, PlayAction):
                    chain["action_successful"] = next_action.result.was_playable
                
                hint_chains.append(chain)
                break
    
    # Compute summary stats
    immediate_plays = [c for c in hint_chains if c["gap"] == 1 and c["action_type"] == "play"]
    immediate_play_success = sum(1 for c in immediate_plays if c["action_successful"]) / len(immediate_plays) if immediate_plays else 0.0
    
    return {
        "hint_chains": len(hint_chains),
        "immediate_plays_after_hint": len(immediate_plays),
        "immediate_play_success_rate": round(immediate_play_success, 3),
        "average_gap_to_action": round(sum(c["gap"] for c in hint_chains) / len(hint_chains), 2) if hint_chains else 0.0,
    }
