#!/usr/bin/env python3
"""Quick test script to verify LLM integration works."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.engine import Team, GameConfig, create_game, apply_clue, transition_to_guessing
from src.agents import (
    AgentConfig, CluerAgent, GuesserAgent, run_discussion,
    create_provider,
)


async def test_cluer(provider, model: str):
    """Test clue generation."""
    print("\n" + "=" * 60)
    print("TESTING CLUER AGENT")
    print("=" * 60)

    config = AgentConfig(
        model=model,
        role="cluer",
        team=Team.RED,
        agent_id="red_cluer",
        temperature=0.7,
        max_retries=3,
    )

    agent = CluerAgent(config, provider)
    state = create_game(config=GameConfig(seed=42))

    print(f"\nBoard words: {state.board.words}")
    print(f"\nRed team words: {sorted(state.board.key_by_category['red'])}")

    clue, trace = await agent.generate_clue(state)

    print(f"\n✓ Clue generated: {clue.word} ({clue.number})")
    print(f"  Reasoning: {trace.parsed_result.get('reasoning', 'N/A')[:200]}...")
    print(f"  Latency: {trace.latency_ms:.0f}ms")
    print(f"  Tokens: {trace.input_tokens} in, {trace.output_tokens} out")

    return state, clue


async def test_guessers(provider, model: str, state, clue):
    """Test discussion and guessing."""
    print("\n" + "=" * 60)
    print("TESTING GUESSER AGENTS")
    print("=" * 60)

    # Apply the clue to move to discussion phase
    state = apply_clue(state, clue.word, clue.number)

    config1 = AgentConfig(
        model=model,
        role="guesser",
        team=Team.RED,
        agent_id="red_guesser_1",
        temperature=0.7,
    )
    config2 = AgentConfig(
        model=model,
        role="guesser",
        team=Team.RED,
        agent_id="red_guesser_2",
        temperature=0.7,
    )

    guessers = [
        GuesserAgent(config1, provider),
        GuesserAgent(config2, provider),
    ]

    print(f"\nClue: {clue.word} ({clue.number})")
    print("\n--- Discussion ---")

    messages, traces, final_state = await run_discussion(guessers, state, max_rounds=2)

    for msg in messages:
        print(f"\n{msg.agent_id}:")
        print(f"  {msg.content[:300]}{'...' if len(msg.content) > 300 else ''}")

    # Make final guesses
    print("\n--- Final Guesses ---")
    guesses, trace = await guessers[0].make_guesses(final_state, messages)

    print(f"\n✓ Guesses: {guesses}")
    print(f"  Reasoning: {trace.parsed_result.get('reasoning', 'N/A')[:200]}...")

    # Check if guesses are correct
    red_words = state.board.key_by_category["red"]
    correct = [g for g in guesses if g in red_words]
    print(f"\n  Correct guesses: {correct}")


async def main():
    # Check for API keys
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        print("Using OpenRouter API")
        provider = create_provider("openrouter", model="anthropic/claude-3.5-sonnet")
        model = "anthropic/claude-3.5-sonnet"
    elif anthropic_key:
        print("Using Anthropic API")
        provider = create_provider("anthropic", model="claude-sonnet-4-20250514")
        model = "claude-sonnet-4-20250514"
    elif openai_key:
        print("Using OpenAI API")
        provider = create_provider("openai", model="gpt-4o")
        model = "gpt-4o"
    else:
        print("ERROR: No API key found!")
        print("\nSet one of these environment variables:")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    try:
        state, clue = await test_cluer(provider, model)
        await test_guessers(provider, model, state, clue)
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
