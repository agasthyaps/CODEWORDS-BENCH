from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from src.agents.llm import LLMProvider

from ..models import (
    ActionLog,
    ClueSet,
    ConsensusGuess,
    GuesserIndependent,
    GuesserShare,
    MappingReference,
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
    # Keep it deterministic and compact; JSON string of prior rounds is stable via model_dump(mode="json").
    rounds = view.get("history_rounds", [])
    if not rounds:
        return "[]"
    return str(rounds)


def _format_public_clues(view: dict[str, Any]) -> str:
    pc = view.get("public_clues")
    return str(pc) if pc is not None else "null"


CONFIRMATION_LANGUAGE_PATTERNS = (
    re.compile(r"\bconfirmed\b", re.IGNORECASE),
    re.compile(r"\blocked\b", re.IGNORECASE),
    re.compile(r"\bwe know\b", re.IGNORECASE),
)
GROUNDING_PENALTY_CAP = 0.49


def _index_actions(round_log: RoundLog) -> dict[tuple[TeamKey, str], Any]:
    out: dict[tuple[TeamKey, str], Any] = {}
    for a in round_log.actions:
        out[(a.team, a.kind)] = a
    return out


def _event_backed_confirmable_clues(
    round_inputs: RoundInputs,
    team: TeamKey,
    kind: str,
) -> dict[str, set[str]]:
    opp: TeamKey = "blue" if team == "red" else "red"
    subject: TeamKey = team if kind == "decode" else opp
    confirmable: dict[str, set[str]] = {str(d): set() for d in range(1, 5)}

    for rr in round_inputs.history_rounds:
        acts = _index_actions(rr)
        if kind == "decode":
            decoded_ok = bool(getattr(acts.get((team, "decode")), "correct", False))
            intercepted_by_opp = bool(getattr(acts.get((opp, "intercept")), "correct", False))
            if not (decoded_ok or intercepted_by_opp):
                continue
        else:
            intercepted_ok = bool(getattr(acts.get((team, "intercept")), "correct", False))
            opp_decode = acts.get((opp, "decode"))
            opp_miscomm = opp_decode is not None and not bool(getattr(opp_decode, "correct", False))
            if not (intercepted_ok or opp_miscomm):
                continue

        clueset = rr.public_clues.get(subject)
        code = rr.reveal_true_codes.get(subject)
        if clueset is None or code is None:
            continue
        for clue_word, d in zip(clueset.clues, code):
            confirmable[str(int(d))].add(str(clue_word).strip().upper())

    return confirmable


def _has_event_evidence(confirmable: dict[str, set[str]]) -> bool:
    return any(confirmable[d] for d in confirmable)


def _is_event_backed_confirmed_ref(ref: MappingReference, confirmable: dict[str, set[str]]) -> bool:
    if getattr(ref, "status", None) != "confirmed":
        return False
    if getattr(ref, "mapping_type", None) != "digit_clue":
        return False
    d = getattr(ref, "digit", None)
    v = getattr(ref, "value", None)
    if not (isinstance(d, str) and isinstance(v, str)):
        return False
    return v.strip().upper() in confirmable.get(d, set())


def _has_event_backed_confirmed_refs(refs: list[MappingReference], confirmable: dict[str, set[str]]) -> bool:
    return any(_is_event_backed_confirmed_ref(ref, confirmable) for ref in refs)


def _has_unbacked_confirmed_refs(refs: list[MappingReference], confirmable: dict[str, set[str]]) -> bool:
    for ref in refs:
        if getattr(ref, "status", None) != "confirmed":
            continue
        if not _is_event_backed_confirmed_ref(ref, confirmable):
            return True
    return False


def _uses_unbacked_confirmation_language(rationale: str, has_event_backed_confirmed: bool) -> bool:
    if not rationale or has_event_backed_confirmed:
        return False
    return any(pat.search(rationale) for pat in CONFIRMATION_LANGUAGE_PATTERNS)


def _apply_grounding_penalty(confidence: float | None, violation: bool) -> float | None:
    if confidence is None or not violation:
        return confidence
    return min(float(confidence), GROUNDING_PENALTY_CAP)


def _reveal_summary_for_team(history_rounds: list[dict[str, Any]], team: TeamKey, key: list[str] | None) -> list[dict[str, Any]]:
    """
    Build a structured reveal summary suitable for the next-round prompt.
    Includes: true code, clues used, and explicit clue->digit mapping.
    For decode (key is provided), also includes digit->keyword table.
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

    async def generate(self, round_inputs: RoundInputs, team: TeamKey) -> tuple[ClueSet, dict[str, Any]]:
        view = view_for_cluer(round_inputs, team)
        assert_view_safe(view)

        system = (
            "You are the CLUER in Decrypto. Output ONLY valid JSON.\n"
            "You must provide exactly 3 clues in order, each a single word.\n"
            "Do NOT use any of your key words verbatim as clues.\n"
        )
        user = (
            f"ROLE_VIEW:\n{view}\n\n"
            "Return JSON with schema:\n"
            "{\n"
            '  \"clues\": [\"CLUE1\",\"CLUE2\",\"CLUE3\"],\n'
            '  \"annotations\": {\n'
            '     \"intended_mapping\": {\"2\":\"<KEYWORD>\", ...},\n'
            '     \"slot_themes\": {\"1\":\"short theme\", \"2\":\"short theme\", \"3\":\"short theme\", \"4\":\"short theme\"},\n'
            '     \"predicted_team_guess\": [d1,d2,d3],\n'
            '     \"p_team_correct\": 0.0-1.0,\n'
            '     \"p_intercept\": 0.0-1.0\n'
            "     ,\"opponent_decode_dist\": {\"1-2-3\": 0.0417, \"1-2-4\": 0.0417, \"...\": 0.0417},\n"
            "     \"opponent_intercept_dist\": {\"1-2-3\": 0.0417, \"1-2-4\": 0.0417, \"...\": 0.0417}\n"
            "  }\n"
            "}\n"
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _validate(obj: dict[str, Any]) -> tuple[ClueSet, dict[str, Any]] | None:
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

            ann = obj.get("annotations")
            if not isinstance(ann, dict):
                return None
            ptc = parse_confidence_01(ann.get("p_team_correct"))
            pi = parse_confidence_01(ann.get("p_intercept"))
            if ptc is None or pi is None:
                return None
            ptg = ann.get("predicted_team_guess")
            if parse_code_triple(ptg) is None:
                return None
            # slot_themes is optional and free-form; keep as-is if present.
            # opponent_*_dist are optional; validate shape if present.
            for k in ("opponent_decode_dist", "opponent_intercept_dist"):
                dist = ann.get(k)
                if dist is None:
                    continue
                if not isinstance(dist, dict):
                    return None
                # accept partial distributions but require numeric probs in [0,1]
                for kk, vv in dist.items():
                    if not isinstance(kk, str):
                        return None
                    if not isinstance(vv, (int, float)):
                        return None
                    p = float(vv)
                    if not (0.0 <= p <= 1.0):
                        return None

            return ClueSet(clues=tuple(clue_words)), {"cluer_annotations": ann}

        # Parse + single repair retry
        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        obj1 = parse_json_object_strict(resp1.content)
        parsed = _validate(obj1) if obj1 else None
        if parsed is not None:
            return parsed

        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": repair_user_message("cluer", "invalid schema")})
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        obj2 = parse_json_object_strict(resp2.content)
        parsed2 = _validate(obj2) if obj2 else None
        if parsed2 is not None:
            return parsed2

        # Failure: return placeholder clues + empty annotations (will likely hurt performance but preserves run continuity).
        return ClueSet(clues=("CLUEA", "CLUEB", "CLUEC")), {"cluer_annotations": {}}


@dataclass(frozen=True)
class DecryptoGuesserLLM:
    provider: LLMProvider
    agent_id: str
    team: TeamKey
    temperature: float = 0.7

    async def independent_guess(self, round_inputs: RoundInputs, kind: str) -> GuesserIndependent:
        if kind == "decode":
            view = view_for_guesser_decode(round_inputs, self.team)
        else:
            view = view_for_guesser_intercept(round_inputs, self.team)
        assert_view_safe(view)
        confirmable = _event_backed_confirmable_clues(round_inputs, self.team, kind)
        confirmed_count = 1 if _has_event_evidence(confirmable) else 0

        system = (
            "You are a Decrypto guesser. Output ONLY valid JSON.\n"
            "Index-grounded reasoning rule:\n"
            "- You may treat ONLY confirmed digit<->clue mappings from prior reveals as facts.\n"
            "Mapping-reference labeling rule:\n"
            "- If you reference a digit<->clue mapping, include it in mapping_references and label it as confirmed/hypothesis/eliminated.\n"
            "- Only 'confirmed' mappings may be stated as facts, and ONLY if they were confirmed by prior reveals.\n"
            "- Any semantic themes are hypotheses unless explicitly confirmed by prior reveals.\n"
            "- If confirmed_count == 0, you MUST say your guess is speculative / no confirmed mapping yet.\n"
        )
        # Add explicit reveal summary (structured) to reduce ambiguity and encourage hypothesis tables.
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
            f"REVEAL_SUMMARY:\n{reveal_summary}\n\n"
            f"Task: {kind.upper()}.\n"
            "Maintain and update a slot->keyword/theme hypothesis table using the reveal summary.\n"
            "Return JSON:\n"
            "{\n"
            '  \"guess\": [d1,d2,d3],\n'
            '  \"confidence\": 0.0-1.0,\n'
            '  \"rationale\": \"short, index-grounded (or explicitly speculative)\",\n'
            '  \"slot_hypotheses\": {\"1\":\"theme\", \"2\":\"theme\", \"3\":\"theme\", \"4\":\"theme\"},\n'
            '  \"mapping_references\": [\n'
            '     {\"mapping_type\":\"digit_clue\",\"digit\":\"1\",\"value\":\"<CLUE>\",\"status\":\"confirmed|hypothesis|eliminated\",\"support\":\"optional\"},\n'
            '     {\"mapping_type\":\"digit_theme\",\"digit\":\"1\",\"value\":\"<THEME>\",\"status\":\"confirmed|hypothesis|eliminated\",\"support\":\"optional\"}\n'
            "  ]\n"
            "}\n"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _parse(
            obj: dict[str, Any],
        ) -> tuple[
            tuple[int, int, int],
            float,
            str,
            dict[str, str] | None,
            list[MappingReference],
        ] | None:
            g = parse_code_triple(obj.get("guess"))
            c = parse_confidence_01(obj.get("confidence"))
            r = obj.get("rationale")
            sh = obj.get("slot_hypotheses")
            slot_hyp: dict[str, str] | None = None
            if isinstance(sh, dict):
                slot_hyp = {
                    str(k): str(v)
                    for k, v in sh.items()
                    if isinstance(k, (str, int)) and isinstance(v, str) and str(k) in {"1", "2", "3", "4"}
                }
            if g is None or c is None or not isinstance(r, str):
                return None
            mrefs = _parse_mapping_references(obj)
            return g, c, r, slot_hyp, mrefs

        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        obj1 = parse_json_object_strict(resp1.content)
        out = _parse(obj1) if obj1 else None
        if out is not None:
            g, c, r, slot_hyp, mrefs = out
            grounding_ok = _grounding_ok(r, confirmed_count)
            unbacked_confirmed = _has_unbacked_confirmed_refs(mrefs, confirmable)
            token_violation = _uses_unbacked_confirmation_language(
                r, _has_event_backed_confirmed_refs(mrefs, confirmable)
            )
            grounding_violation = (not grounding_ok) or unbacked_confirmed or token_violation
            grounding_ok = grounding_ok and not unbacked_confirmed and not token_violation
            overconf = (confirmed_count == 0 and c > 0.7)
            labels_ok = _mapping_labels_ok(mrefs)
            return GuesserIndependent(
                agent_id=self.agent_id,
                guess=g,
                confidence=_apply_grounding_penalty(c, grounding_violation),
                rationale=r,
                parse_ok=True,
                grounding_ok=grounding_ok,
                overconfident=overconf,
                slot_hypotheses=slot_hyp,
                mapping_references=mrefs,
                mapping_labels_ok=labels_ok,
                parse_error=None,
                parse_retry_used=False,
            )

        # Single retry: strict one-line format (salvage guess+conf even if JSON failed).
        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": strict_guess_retry_prompt("expected JSON with guess+confidence")})
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        salvaged = parse_guess_conf_line(resp2.content)
        if salvaged is not None:
            g2, c2 = salvaged
            overconf = (confirmed_count == 0 and c2 > 0.7)
            return GuesserIndependent(
                agent_id=self.agent_id,
                guess=g2,
                confidence=c2,
                rationale="",
                parse_ok=True,
                grounding_ok=True,  # never count parse failures as grounding violations
                overconfident=overconf,
                slot_hypotheses=None,
                mapping_references=[],
                mapping_labels_ok=True,
                parse_error="json_parse_failed",
                parse_retry_used=True,
            )

        # Final fallback (no abstentions): deterministic safe guess + conf=0.0
        return GuesserIndependent(
            agent_id=self.agent_id,
            guess=(1, 2, 3),
            confidence=0.0,
            rationale="",
            parse_ok=False,
            grounding_ok=True,
            overconfident=False,
            slot_hypotheses=None,
            mapping_references=[],
            mapping_labels_ok=True,
            parse_error="retry_parse_failed",
            parse_retry_used=True,
        )

    async def share_rationale(
        self,
        round_inputs: RoundInputs,
        kind: str,
        independent_a: GuesserIndependent,
        independent_b: GuesserIndependent,
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
    ) -> ConsensusGuess:
        if kind == "decode":
            view = view_for_guesser_decode(round_inputs, self.team)
        else:
            view = view_for_guesser_intercept(round_inputs, self.team)
        assert_view_safe(view)
        confirmable = _event_backed_confirmable_clues(round_inputs, self.team, kind)
        confirmed_count = 1 if _has_event_evidence(confirmable) else 0

        system = (
            "You are the captain deciding the final consensus. Output ONLY valid JSON.\n"
            "Index-grounded reasoning rule:\n"
            "- You may treat ONLY confirmed digit<->clue mappings from prior reveals as facts.\n"
            "Mapping-reference labeling rule:\n"
            "- If you reference a digit<->clue mapping, include it in mapping_references and label it as confirmed/hypothesis/eliminated.\n"
            "- Only 'confirmed' mappings may be stated as facts, and ONLY if they were confirmed by prior reveals.\n"
            "- Any semantic themes are hypotheses unless explicitly confirmed by prior reveals.\n"
            "- If confirmed_count == 0, you MUST say your guess is speculative / no confirmed mapping yet.\n"
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
            f"REVEAL_SUMMARY:\n{reveal_summary}\n\n"
            f"Task: {kind.upper()}.\n"
            f"Independent:\n{[x.model_dump(mode='json') for x in independent]}\n\n"
            f"Share messages:\n{[x.model_dump(mode='json') for x in shares]}\n\n"
            "Update the slot->keyword/theme hypothesis table using the reveal summary.\n"
            "Return JSON:\n"
            "{\n"
            '  \"guess\": [d1,d2,d3],\n'
            '  \"confidence\": 0.0-1.0,\n'
            '  \"rationale\": \"short, index-grounded (or explicitly speculative)\",\n'
            '  \"slot_hypotheses\": {\"1\":\"theme\", \"2\":\"theme\", \"3\":\"theme\", \"4\":\"theme\"},\n'
            '  \"mapping_references\": [\n'
            '     {\"mapping_type\":\"digit_clue\",\"digit\":\"1\",\"value\":\"<CLUE>\",\"status\":\"confirmed|hypothesis|eliminated\",\"support\":\"optional\"},\n'
            '     {\"mapping_type\":\"digit_theme\",\"digit\":\"1\",\"value\":\"<THEME>\",\"status\":\"confirmed|hypothesis|eliminated\",\"support\":\"optional\"}\n'
            "  ]\n"
            "}\n"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _parse(
            obj: dict[str, Any],
        ) -> tuple[
            tuple[int, int, int],
            float,
            str,
            dict[str, str] | None,
            list[MappingReference],
        ] | None:
            g = parse_code_triple(obj.get("guess"))
            c = parse_confidence_01(obj.get("confidence"))
            r = obj.get("rationale")
            sh = obj.get("slot_hypotheses")
            slot_hyp: dict[str, str] | None = None
            if isinstance(sh, dict):
                slot_hyp = {
                    str(k): str(v)
                    for k, v in sh.items()
                    if isinstance(k, (str, int)) and isinstance(v, str) and str(k) in {"1", "2", "3", "4"}
                }
            if g is None or c is None or not isinstance(r, str):
                return None
            mrefs = _parse_mapping_references(obj)
            return g, c, r, slot_hyp, mrefs

        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        obj1 = parse_json_object_strict(resp1.content)
        out = _parse(obj1) if obj1 else None
        if out is not None:
            g, c, r, slot_hyp, mrefs = out
            grounding_ok = _grounding_ok(r, confirmed_count)
            unbacked_confirmed = _has_unbacked_confirmed_refs(mrefs, confirmable)
            token_violation = _uses_unbacked_confirmation_language(
                r, _has_event_backed_confirmed_refs(mrefs, confirmable)
            )
            grounding_violation = (not grounding_ok) or unbacked_confirmed or token_violation
            grounding_ok = grounding_ok and not unbacked_confirmed and not token_violation
            overconf = (confirmed_count == 0 and c > 0.7)
            labels_ok = _mapping_labels_ok(mrefs)
            return ConsensusGuess(
                captain_id=self.agent_id,
                guess=g,
                confidence=_apply_grounding_penalty(c, grounding_violation),
                rationale=r,
                parse_ok=True,
                grounding_ok=grounding_ok,
                overconfident=overconf,
                slot_hypotheses=slot_hyp,
                mapping_references=mrefs,
                mapping_labels_ok=labels_ok,
                parse_error=None,
                parse_retry_used=False,
            )

        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": strict_guess_retry_prompt("expected JSON with guess+confidence")})
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        salvaged = parse_guess_conf_line(resp2.content)
        if salvaged is not None:
            g2, c2 = salvaged
            overconf = (confirmed_count == 0 and c2 > 0.7)
            return ConsensusGuess(
                captain_id=self.agent_id,
                guess=g2,
                confidence=c2,
                rationale="",
                parse_ok=True,
                grounding_ok=True,
                overconfident=overconf,
                slot_hypotheses=None,
                mapping_references=[],
                mapping_labels_ok=True,
                parse_error="json_parse_failed",
                parse_retry_used=True,
            )

        return ConsensusGuess(
            captain_id=self.agent_id,
            guess=(1, 2, 3),
            confidence=0.0,
            rationale="",
            parse_ok=False,
            grounding_ok=True,
            overconfident=False,
            slot_hypotheses=None,
            mapping_references=[],
            mapping_labels_ok=True,
            parse_error="retry_parse_failed",
            parse_retry_used=True,
        )


def _grounding_ok(rationale: str, confirmed_count: int) -> bool:
    """
    Minimal enforcement:
    - If confirmed_count == 0 (no event-backed evidence), rationale must explicitly state uncertainty.
    - Otherwise, we do not strictly validate content (prompting handles it).
    """
    if confirmed_count > 0:
        return True
    text = (rationale or "").lower()
    needles = [
        "speculative",
        "no confirmed",
        "no confirmation",
        "uncertain",
        "guessing",
        "not sure",
    ]
    return any(n in text for n in needles)


def _parse_mapping_references(obj: dict[str, Any]) -> list[MappingReference]:
    refs = obj.get("mapping_references")
    if not isinstance(refs, list):
        return []
    out: list[MappingReference] = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        mapping_type = r.get("mapping_type", "digit_clue")
        digit = r.get("digit")
        value = r.get("value")
        # Backward compat: accept "clue" as value (digit_clue)
        if value is None and "clue" in r:
            value = r.get("clue")
        status = r.get("status")
        support = r.get("support")
        digit_s = str(digit)
        if digit_s not in {"1", "2", "3", "4"}:
            continue
        if mapping_type not in {"digit_clue", "digit_theme"}:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if status not in {"confirmed", "hypothesis", "eliminated"}:
            continue
        if support is not None and not isinstance(support, str):
            support = None
        out.append(
            MappingReference(
                mapping_type=mapping_type,  # type: ignore[arg-type]
                digit=digit_s,  # type: ignore[arg-type]
                value=value.strip().upper() if mapping_type == "digit_clue" else value.strip(),
                status=status,  # type: ignore[arg-type]
                support=support,
            )
        )
    return out


def _mapping_labels_ok(_mapping_references: list[MappingReference]) -> bool:
    """
    Under the new schema, evaluator-side determines whether a 'confirmed' label is justified.
    Agent-side only needs to provide the labels; we always accept the structure here.
    """
    return True


async def run_bounded_action(
    round_inputs: RoundInputs,
    team: TeamKey,
    opponent_team: TeamKey,
    kind: str,
    guesser_1: DecryptoGuesserLLM,
    guesser_2: DecryptoGuesserLLM,
) -> ActionLog:
    """
    Bounded discussion loop:
      independent_guess(+confidence) -> share_rationale (1 msg each) -> captain consensus (+confidence)
    """
    if kind not in ("decode", "intercept"):
        raise ValueError(f"Unknown action kind: {kind}")

    confirmable = _event_backed_confirmable_clues(round_inputs, team, kind)
    confirmed_count = sum(1 for d in confirmable if confirmable[d])
    # Intercepts before any opponent reveals are uninformed (round 1 baseline).
    uninformed = (kind == "intercept" and len(round_inputs.history_rounds) == 0)

    ind1, ind2 = await asyncio.gather(
        guesser_1.independent_guess(round_inputs, kind),
        guesser_2.independent_guess(round_inputs, kind),
    )
    s1, s2 = await asyncio.gather(
        guesser_1.share_rationale(round_inputs, kind, ind1, ind2),
        guesser_2.share_rationale(round_inputs, kind, ind1, ind2),
    )
    # Deterministic captain: guesser_1
    consensus = await guesser_1.captain_consensus(
        round_inputs, kind, (ind1, ind2), (s1, s2)
    )

    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,
        opponent_team=opponent_team,
        independent=(ind1, ind2),
        share=(s1, s2),
        consensus=consensus,
        correct=False,  # computed only at reveal step in orchestrator
        confirmed_mapping_count=confirmed_count,
        uninformed=uninformed,
    )
