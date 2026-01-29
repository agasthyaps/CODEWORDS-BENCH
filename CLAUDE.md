# Claude Code Notes for CODEWORDS-BENCH

## Production Persistent Volume Structure (Railway)

The production server uses a Railway persistent volume mounted at `/app/benchmark_results/`.

### Directory Layout

```
/app/benchmark_results/
├── leaderboard.json          # Main leaderboard data
├── lost+found/               # Railway volume system directory (ignore)
├── sessions/                 # NEW consolidated session storage
│   ├── batches/              # Batch run sessions
│   ├── codenames/            # Individual codenames game sessions
│   ├── decrypto/             # Individual decrypto game sessions
│   ├── hanabi/               # Individual hanabi game sessions
│   └── stats/                # Session statistics
├── benchmark_*/              # Named benchmark runs (legacy pattern)
│   ├── benchmark_state.json  # Benchmark progress/state
│   └── config.json           # Benchmark configuration
├── test/                     # Test benchmark (old)
│   ├── codenames/
│   ├── decrypto/
│   ├── benchmark_state.json
│   └── config.json
├── test2/                    # Another test benchmark (old)
│   ├── codenames/
│   ├── decrypto/
│   ├── hanabi/
│   ├── benchmark_state.json
│   └── config.json
└── hanabi_addl/              # Hanabi additional benchmark
    ├── hanabi/
    ├── findings/
    ├── benchmark_state.json
    └── config.json
```

### Key Notes

1. **`sessions/` directory**: This is the NEW consolidated storage for individual game sessions (not benchmarks). It's organized by game type with subdirectories for batches and stats.

2. **Benchmark directories**: Each benchmark run creates its own directory (e.g., `benchmark_2026-01-27_1946`, `hanabi_addl`, `test`, `test2`). These contain:
   - `benchmark_state.json`: Tracks progress, completed games, etc.
   - `config.json`: The benchmark configuration
   - Game-type subdirectories (`codenames/`, `decrypto/`, `hanabi/`) containing individual game episode results

3. **`lost+found/`**: This is a standard Linux filesystem directory created by ext4. Ignore it - it's just part of how Railway provisions the persistent volume.

4. **`leaderboard.json`**: The main leaderboard data file, stored at the root level.

### Snapshot from 2026-01-29

```
benchmark_2026-01-27_1946/    # Contains benchmark_state.json, config.json
hanabi_addl/                   # Contains benchmark_state.json, config.json, findings/, hanabi/
sessions/batches/              # Batch session storage
sessions/codenames/            # Individual codenames sessions
sessions/decrypto/             # Individual decrypto sessions
sessions/hanabi/               # Individual hanabi sessions
sessions/stats/                # Session statistics
test/                          # Legacy test benchmark
test2/                         # Legacy test benchmark (larger, 43KB state file)
```
