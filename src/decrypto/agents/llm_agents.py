from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.agents.llm import LLMProvider
from src.core.parsing import extract_scratchpad, remove_scratchpad_from_response

from ..models import (
    ActionLog,
    ClueSet,
    ConsensusGuess,
    GuesserIndependent,
    GuesserShare,
    RoundLog,
    RoundInputs,
    TeamKey,
)
from ..parsing import (
    parse_code_triple,
    parse_confidence_01,
    parse_guess_conf_line,
    parse_json_object_strict,
    repair_user_message,
    strict_guess_retry_prompt,
)
from ..visibility import (
    assert_view_safe,
    view_for_cluer,
    view_for_guesser_decode,
    view_for_guesser_intercept,
)


def _format_history(view: dict[str, Any]) -> str:
    rounds = view.get("history_rounds", [])
    if not rounds:
        return "[]"
    return str(rounds)


def _format_public_clues(view: dict[str, Any]) -> str:
    pc = view.get("public_clues")
    return str(pc) if pc is not None else "null"


def _load_prompt_template(name: str) -> str:
    path = Path(__file__).parent / "prompts" / name
    with open(path, "r") as f:
        return f.read()


def _format_discussion_log(discussion: list[dict[str, str]]) -> str:
    if not discussion:
        return "(no discussion yet)"
    return "\n".join(
        [f"{d.get('agent_id', 'agent')}: {d.get('message', '')}" for d in discussion]
    )


def _parse_consensus_flag(text: str) -> bool:
    return bool(re.search(r"CONSENSUS\s*:\s*YES", text, re.IGNORECASE))


def _parse_discussion_response(text: str) -> tuple[bool, list[str] | None]:
    """
    Parse a Decrypto discussion response.
    
    Returns:
        (consensus_flag, candidates_list)
        candidates_list is a list of code strings like ["1-3-4", "2-4-1"]
    """
    consensus = _parse_consensus_flag(text)
    
    # Extract CANDIDATES: 1-3-4, 2-4-1
    candidates = None
    match = re.search(r"CANDIDATES\s*:\s*(.+?)(?:\n|SCRATCHPAD|$)", text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        candidates = [c.strip().replace(" ", "") for c in raw.split(",") if c.strip()]
    
    return consensus, candidates


def _candidates_match(list1: list[str] | None, list2: list[str] | None) -> bool:
    """
    Check if two CANDIDATES lists share at least one common code.
    """
    if list1 is None or list2 is None:
        return False
    def normalize(codes: list[str]) -> set[str]:
        result = set()
        for c in codes:
            digits = re.findall(r'\d', c)
            if len(digits) >= 3:
                result.add("-".join(digits[:3]))
        return result
    
    set1 = normalize(list1)
    set2 = normalize(list2)
    return len(set1 & set2) > 0


def _index_actions(round_log: RoundLog) -> dict[tuple[TeamKey, str], Any]:
    out: dict[tuple[TeamKey, str], Any] = {}
    for a in round_log.actions:
        out[(a.team, a.kind)] = a
    return out


def _reveal_summary_for_team(history_rounds: list[dict[str, Any]], team: TeamKey, key: list[str] | None) -> list[dict[str, Any]]:
    """
    Build a structured reveal summary suitable for the next-round prompt.
    """
    out: list[dict[str, Any]] = []
    for rr in history_rounds:
        try:
            rnd = int(rr.get("round_number"))
            public_clues = rr.get("public_clues", {})
            reveal = rr.get("reveal_true_codes", {})
            clues = public_clues.get(team, {}).get("clues") if isinstance(public_clues, dict) else None
            code = reveal.get(team) if isinstance(reveal, dict) else None
        except Exception:
            continue
        if not (isinstance(clues, (list, tuple)) and isinstance(code, (list, tuple)) and len(clues) == 3 and len(code) == 3):
            continue
        pairs = [{"clue": str(c), "digit": int(d)} for c, d in zip(clues, code)]
        digit_to_keyword = None
        if key is not None and len(key) == 4:
            digit_to_keyword = {str(i + 1): key[i] for i in range(4)}
        out.append(
            {
                "round": rnd,
                "team": team,
                "clues": [str(c) for c in clues],
                "true_code": [int(d) for d in code],
                "clue_digit_pairs": pairs,
                "note": "Each clue i maps to the keyword at position true_code[i] in that team's key.",
                "digit_to_keyword": digit_to_keyword,
            }
        )
    return out


@dataclass(frozen=True)
class DecryptoCluerLLM:
    provider: LLMProvider
    model_id: str
    temperature: float = 0.7

    async def generate(
        self,
        round_inputs: RoundInputs,
        team: TeamKey,
        scratchpad_content: str = "",
    ) -> tuple[ClueSet, dict[str, Any], str | None]:
        """
        Generate clues for the round.
        
        Returns:
            (ClueSet, trace_data, scratchpad_addition)
        """
        view = view_for_cluer(round_inputs, team)
        assert_view_safe(view)

        scratchpad_section = ""
        if scratchpad_content:
            scratchpad_section = f"\n\nYour scratchpad from previous rounds:\n{scratchpad_content}\n"

        system = (
            "You are the CLUER in Decrypto. Output ONLY valid JSON.\n"
            "You must provide exactly 3 clues in order, each a single word.\n"
            "Do NOT use any of your key words verbatim as clues.\n\n"
            "Your teammates must decode your clues to guess the secret code.\n"
            "The opponent team sees all your clues from all rounds.\n\n"
            "You have a private scratchpad. To add notes for future rounds, include:\n"
            "SCRATCHPAD: [your notes]\n"
            "at the end of your JSON output (outside the JSON)."
        )
        user = (
            f"ROLE_VIEW:\n{view}\n"
            f"{scratchpad_section}\n"
            "Return JSON with schema:\n"
            "{\n"
            '  \"clues\": [\"CLUE1\",\"CLUE2\",\"CLUE3\"],\n'
            '  \"reasoning\": \"brief explanation of your clue choices\"\n'
            "}\n\n"
            "You may add SCRATCHPAD: notes after the JSON."
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _validate(obj: dict[str, Any]) -> ClueSet | None:
            clues = obj.get("clues")
            if not isinstance(clues, list) or len(clues) != 3:
                return None
            clue_words: list[str] = []
            for c in clues:
                if not isinstance(c, str):
                    return None
                w = c.strip().upper()
                if not w or " " in w or len(w) > 32:
                    return None
                clue_words.append(w)

            key_words = set(view["key"])
            if any(w in key_words for w in clue_words):
                return None

            return ClueSet(clues=tuple(clue_words))

        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        scratchpad_addition = extract_scratchpad(resp1.content)
        
        obj1 = parse_json_object_strict(resp1.content)
        clueset = _validate(obj1) if obj1 else None
        if clueset is not None:
            return clueset, {"reasoning": obj1.get("reasoning", "")}, scratchpad_addition

        # Single repair retry
        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": repair_user_message("cluer", "invalid schema")})
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        scratchpad_addition = extract_scratchpad(resp2.content) or scratchpad_addition
        
        obj2 = parse_json_object_strict(resp2.content)
        clueset2 = _validate(obj2) if obj2 else None
        if clueset2 is not None:
            return clueset2, {"reasoning": obj2.get("reasoning", "")}, scratchpad_addition

        # Failure: return fallback clues that hint at positions (best we can do)
        # Using ordinal words gives teammates a slim chance if they catch on
        return ClueSet(clues=("FIRST", "SECOND", "THIRD")), {
            "reasoning": "PARSE_FAILURE: Could not parse clue response, using positional fallback"
        }, scratchpad_addition


@dataclass(frozen=True)
class DecryptoGuesserLLM:
    provider: LLMProvider
    agent_id: str
    team: TeamKey
    temperature: float = 0.7

    async def independent_guess(
        self,
        round_inputs: RoundInputs,
        kind: str,
        scratchpad_content: str = "",
    ) -> tuple[GuesserIndependent, str | None]:
        """
        Make an independent guess.
        
        Returns:
            (GuesserIndependent, scratchpad_addition)
        """
        if kind == "decode":
            view = view_for_guesser_decode(round_inputs, self.team)
        else:
            view = view_for_guesser_intercept(round_inputs, self.team)
        assert_view_safe(view)

        scratchpad_section = ""
        if scratchpad_content:
            scratchpad_section = f"\n\nYour scratchpad from previous rounds:\n{scratchpad_content}\n"

        system = (
            "You are a Decrypto guesser. Output ONLY valid JSON.\n\n"
            "GAME CONTEXT:\n"
            "- WIN: 2 successful interceptions of opponent's code\n"
            "- LOSE: 2 miscommunications (failing to decode your own team's code)\n"
            "- DECODE task: Guess YOUR team's code. Failure = miscommunication (brings you closer to LOSING)\n"
            "- INTERCEPT task: Guess OPPONENT's code. Success = interception (brings you closer to WINNING)\n\n"
            "You have a private scratchpad. To add notes for future rounds, include:\n"
            "SCRATCHPAD: [your notes]\n"
            "at the end of your JSON output (outside the JSON)."
        )
        
        history_rounds = view.get("history_rounds", [])
        reveal_summary = []
        if isinstance(history_rounds, list):
            if kind == "decode":
                reveal_summary = _reveal_summary_for_team(history_rounds, self.team, view.get("key"))
            else:
                opp: TeamKey = "blue" if self.team == "red" else "red"
                reveal_summary = _reveal_summary_for_team(history_rounds, opp, None)

        user = (
            f"ROLE_VIEW:\n{view}\n\n"
            f"REVEAL_SUMMARY:\n{reveal_summary}\n"
            f"{scratchpad_section}\n"
            f"Task: {kind.upper()}.\n"
            "Return JSON:\n"
            "{\n"
            '  \"guess\": [d1,d2,d3],\n'
            '  \"confidence\": 0.0-1.0,\n'
            '  \"rationale\": \"brief explanation\"\n'
            "}\n\n"
            "You may add SCRATCHPAD: notes after the JSON."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _parse(obj: dict[str, Any]) -> tuple[tuple[int, int, int], float, str] | None:
            g = parse_code_triple(obj.get("guess"))
            c = parse_confidence_01(obj.get("confidence"))
            r = obj.get("rationale")
            if g is None or c is None or not isinstance(r, str):
                return None
            return g, c, r

        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        scratchpad_addition = extract_scratchpad(resp1.content)
        
        obj1 = parse_json_object_strict(resp1.content)
        out = _parse(obj1) if obj1 else None
        if out is not None:
            g, c, r = out
            return GuesserIndependent(
                agent_id=self.agent_id,
                guess=g,
                confidence=c,
                rationale=r,
                parse_ok=True,
                parse_error=None,
                parse_retry_used=False,
            ), scratchpad_addition

        # Single retry
        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": strict_guess_retry_prompt("expected JSON with guess+confidence")})
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        scratchpad_addition = extract_scratchpad(resp2.content) or scratchpad_addition
        
        salvaged = parse_guess_conf_line(resp2.content)
        if salvaged is not None:
            g2, c2 = salvaged
            return GuesserIndependent(
                agent_id=self.agent_id,
                guess=g2,
                confidence=c2,
                rationale="",
                parse_ok=True,
                parse_error="json_parse_failed",
                parse_retry_used=True,
            ), scratchpad_addition

        # Final fallback - use default guess with 0 confidence to signal parse failure
        return GuesserIndependent(
            agent_id=self.agent_id,
            guess=(1, 2, 3),
            confidence=0.0,
            rationale="PARSE_FAILURE: Could not parse guess response",
            parse_ok=False,
            parse_error="retry_parse_failed",
            parse_retry_used=True,
        ), scratchpad_addition

    async def discuss(
        self,
        round_inputs: RoundInputs,
        kind: str,
        discussion: list[dict[str, str]],
        independent: tuple[GuesserIndependent, GuesserIndependent],
        scratchpad_content: str = "",
    ) -> tuple[str, bool, str | None]:
        """
        Participate in discussion.
        
        Returns:
            (message_content, consensus_flag, scratchpad_addition)
        """
        if kind == "decode":
            view = view_for_guesser_decode(round_inputs, self.team)
        else:
            view = view_for_guesser_intercept(round_inputs, self.team)
        assert_view_safe(view)

        scratchpad_section = ""
        if scratchpad_content:
            scratchpad_section = f"\n\nYour scratchpad:\n{scratchpad_content}\n"

        system = _load_prompt_template("guesser_discussion.md").format(kind=kind.upper())
        user = _load_prompt_template("guesser_discussion_turn.md").format(
            view=view,
            kind=kind.upper(),
            independent=[x.model_dump(mode="json") for x in independent],
            discussion=_format_discussion_log(discussion),
        ) + scratchpad_section
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp = await self.provider.complete(messages=messages, temperature=self.temperature)
        raw_content = resp.content.strip()
        scratchpad_addition = extract_scratchpad(raw_content)
        # Strip scratchpad from content to prevent leaking private notes to other agents
        content = remove_scratchpad_from_response(raw_content)
        return content, _parse_consensus_flag(raw_content), scratchpad_addition

    async def share_rationale(
        self,
        round_inputs: RoundInputs,
        kind: str,
        independent_a: GuesserIndependent,
        independent_b: GuesserIndependent,
        discussion: list[dict[str, str]],
    ) -> GuesserShare:
        if kind == "decode":
            view = view_for_guesser_decode(round_inputs, self.team)
        else:
            view = view_for_guesser_intercept(round_inputs, self.team)
        assert_view_safe(view)

        system = "You are a Decrypto guesser. Output ONLY valid JSON."
        user = (
            f"ROLE_VIEW:\n{view}\n\n"
            f"Task: {kind.upper()}.\n"
            f"Independent guesses:\n{independent_a.model_dump(mode='json')}\n{independent_b.model_dump(mode='json')}\n\n"
            f"Discussion transcript:\n{_format_discussion_log(discussion)}\n\n"
            "Share one short message to help your teammate converge.\n"
            "Return JSON:\n"
            "{ \"message\": \"...\" }\n"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        obj1 = parse_json_object_strict(resp1.content)
        if obj1 and isinstance(obj1.get("message"), str):
            return GuesserShare(agent_id=self.agent_id, message=obj1["message"])

        # Repair once
        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": repair_user_message("share_rationale", "invalid schema")})
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        obj2 = parse_json_object_strict(resp2.content)
        if obj2 and isinstance(obj2.get("message"), str):
            return GuesserShare(agent_id=self.agent_id, message=obj2["message"])

        return GuesserShare(agent_id=self.agent_id, message="")

    async def captain_consensus(
        self,
        round_inputs: RoundInputs,
        kind: str,
        independent: tuple[GuesserIndependent, GuesserIndependent],
        shares: tuple[GuesserShare, GuesserShare],
        discussion: list[dict[str, str]],
        scratchpad_content: str = "",
    ) -> tuple[ConsensusGuess, str | None]:
        """
        Make final consensus guess as captain.
        
        Returns:
            (ConsensusGuess, scratchpad_addition)
        """
        if kind == "decode":
            view = view_for_guesser_decode(round_inputs, self.team)
        else:
            view = view_for_guesser_intercept(round_inputs, self.team)
        assert_view_safe(view)

        scratchpad_section = ""
        if scratchpad_content:
            scratchpad_section = f"\n\nYour scratchpad:\n{scratchpad_content}\n"

        system = (
            "You are the captain deciding the final consensus. Output ONLY valid JSON.\n"
            "You have a private scratchpad. To add notes, include SCRATCHPAD: at the end."
        )
        history_rounds = view.get("history_rounds", [])
        reveal_summary = []
        if isinstance(history_rounds, list):
            if kind == "decode":
                reveal_summary = _reveal_summary_for_team(history_rounds, self.team, view.get("key"))
            else:
                opp: TeamKey = "blue" if self.team == "red" else "red"
                reveal_summary = _reveal_summary_for_team(history_rounds, opp, None)

        user = (
            f"ROLE_VIEW:\n{view}\n\n"
            f"REVEAL_SUMMARY:\n{reveal_summary}\n"
            f"{scratchpad_section}\n"
            f"Task: {kind.upper()}.\n"
            f"Independent:\n{[x.model_dump(mode='json') for x in independent]}\n\n"
            f"Share messages:\n{[x.model_dump(mode='json') for x in shares]}\n\n"
            f"Discussion transcript:\n{_format_discussion_log(discussion)}\n\n"
            "Return JSON:\n"
            "{\n"
            '  \"guess\": [d1,d2,d3],\n'
            '  \"confidence\": 0.0-1.0,\n'
            '  \"rationale\": \"brief explanation\"\n'
            "}\n\n"
            "You may add SCRATCHPAD: notes after the JSON."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _parse(obj: dict[str, Any]) -> tuple[tuple[int, int, int], float, str] | None:
            g = parse_code_triple(obj.get("guess"))
            c = parse_confidence_01(obj.get("confidence"))
            r = obj.get("rationale")
            if g is None or c is None or not isinstance(r, str):
                return None
            return g, c, r

        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        scratchpad_addition = extract_scratchpad(resp1.content)
        
        obj1 = parse_json_object_strict(resp1.content)
        out = _parse(obj1) if obj1 else None
        if out is not None:
            g, c, r = out
            return ConsensusGuess(
                captain_id=self.agent_id,
                guess=g,
                confidence=c,
                rationale=r,
                parse_ok=True,
                parse_error=None,
                parse_retry_used=False,
            ), scratchpad_addition

        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": strict_guess_retry_prompt("expected JSON with guess+confidence")})
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        scratchpad_addition = extract_scratchpad(resp2.content) or scratchpad_addition
        
        salvaged = parse_guess_conf_line(resp2.content)
        if salvaged is not None:
            g2, c2 = salvaged
            return ConsensusGuess(
                captain_id=self.agent_id,
                guess=g2,
                confidence=c2,
                rationale="",
                parse_ok=True,
                parse_error="json_parse_failed",
                parse_retry_used=True,
            ), scratchpad_addition

        return ConsensusGuess(
            captain_id=self.agent_id,
            guess=(1, 2, 3),
            confidence=0.0,
            rationale="",
            parse_ok=False,
            parse_error="retry_parse_failed",
            parse_retry_used=True,
        ), scratchpad_addition


async def run_bounded_action(
    round_inputs: RoundInputs,
    team: TeamKey,
    opponent_team: TeamKey,
    kind: str,
    guesser_1: DecryptoGuesserLLM,
    guesser_2: DecryptoGuesserLLM,
    g1_scratchpad: str = "",
    g2_scratchpad: str = "",
    *,
    discussion_log: list[dict[str, str]] | None = None,
    max_discussion_turns_per_guesser: int = 2,
    emit_discussion: Callable[[str, str], None] | None = None,
) -> tuple[ActionLog, dict[str, str]]:
    """
    Bounded discussion loop:
      independent_guess -> discussion -> share_rationale -> captain consensus
    
    Returns:
        (ActionLog, scratchpad_additions_dict)
        scratchpad_additions_dict maps agent_id to their combined scratchpad additions
    """
    if kind not in ("decode", "intercept"):
        raise ValueError(f"Unknown action kind: {kind}")

    uninformed = (kind == "intercept" and len(round_inputs.history_rounds) == 0)
    
    # Track scratchpad additions per agent
    scratchpad_adds: dict[str, list[str]] = {
        guesser_1.agent_id: [],
        guesser_2.agent_id: [],
    }

    # Independent guesses with scratchpads
    ind1_result = await guesser_1.independent_guess(round_inputs, kind, g1_scratchpad)
    ind2_result = await guesser_2.independent_guess(round_inputs, kind, g2_scratchpad)
    ind1, ind1_add = ind1_result
    ind2, ind2_add = ind2_result
    
    if ind1_add:
        scratchpad_adds[guesser_1.agent_id].append(ind1_add)
    if ind2_add:
        scratchpad_adds[guesser_2.agent_id].append(ind2_add)
    
    discussion_log = discussion_log if discussion_log is not None else []
    consecutive_consensus = 0
    previous_candidates: list[str] | None = None
    max_messages = max(1, max_discussion_turns_per_guesser) * 2
    guessers = [guesser_1, guesser_2]
    scratchpads = [g1_scratchpad, g2_scratchpad]
    
    for i in range(max_messages):
        guesser = guessers[i % 2]
        scratchpad = scratchpads[i % 2]
        content, _, disc_add = await guesser.discuss(
            round_inputs, kind, discussion_log, (ind1, ind2), scratchpad
        )
        if disc_add:
            scratchpad_adds[guesser.agent_id].append(disc_add)
        
        entry = {"agent_id": guesser.agent_id, "message": content}
        discussion_log.append(entry)
        if emit_discussion is not None:
            emit_discussion(guesser.agent_id, content)
        
        consensus, candidates = _parse_discussion_response(content)
        if consensus:
            if consecutive_consensus == 0:
                consecutive_consensus = 1
                previous_candidates = candidates
            else:
                if _candidates_match(previous_candidates, candidates):
                    break
                else:
                    consecutive_consensus = 1
                    previous_candidates = candidates
        else:
            consecutive_consensus = 0
            previous_candidates = None

    s1, s2 = await asyncio.gather(
        guesser_1.share_rationale(round_inputs, kind, ind1, ind2, discussion_log),
        guesser_2.share_rationale(round_inputs, kind, ind1, ind2, discussion_log),
    )
    
    consensus_result = await guesser_1.captain_consensus(
        round_inputs, kind, (ind1, ind2), (s1, s2), discussion_log, g1_scratchpad
    )
    consensus, captain_add = consensus_result
    if captain_add:
        scratchpad_adds[guesser_1.agent_id].append(captain_add)

    # Combine scratchpad additions per agent
    combined_adds = {
        agent_id: " | ".join(adds) if adds else ""
        for agent_id, adds in scratchpad_adds.items()
    }

    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,
        opponent_team=opponent_team,
        independent=(ind1, ind2),
        share=(s1, s2),
        consensus=consensus,
        correct=False,  # computed only at reveal step in orchestrator
        uninformed=uninformed,
    ), combined_adds
