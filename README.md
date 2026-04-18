# Stem Agent

[![CI](https://github.com/qflen/stem-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/qflen/stem-agent/actions/workflows/ci.yml)

A self-specializing AI agent that evolves from an undifferentiated core into a task-specific specialist through guided differentiation.

Full writeup: [`docs/writeup.pdf`](docs/writeup.pdf) (4 pages).

https://github.com/user-attachments/assets/2da6098c-45eb-47ce-b338-03519fb83a08

The clip above replays a recorded run from `docs/example_run/journal.json` (no live API calls) and lands on the headline numbers: baseline F1 `0.000` → specialized F1 `0.778` on the 20-sample benchmark. Pausable; 15 seconds.

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/qflen/stem-agent.git
cd stem-agent

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Or with uv (faster)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"

# Or create a .env file
echo 'OPENAI_API_KEY=your-key-here' > .env
```

### Usage

```bash
# Run the full differentiation process
stem-agent differentiate --domain code_quality_analysis

# Review a Python file with the specialized agent
stem-agent review path/to/file.py

# View evaluation results
stem-agent evaluate

# Pretty-print the evolution journal
stem-agent journal --last
```

### Development

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make format

# Run full evaluation
make eval
```

## Architecture

The stem agent follows a biological differentiation metaphor:

```
UNDIFFERENTIATED → SENSING → DIFFERENTIATING → VALIDATING → SPECIALIZED → EXECUTING
                                    ▲              │
                                    │              │
                                    └── ROLLBACK ──┘
```

### Phases

1. **Sensing**: Queries an LLM to build structured domain knowledge
2. **Planning**: Selects capabilities and designs a multi-pass review architecture
3. **Specialization**: Assembles the specialized agent from prompt fragments and tools
4. **Validation**: Benchmarks against a ground-truth corpus with regression gates
5. **Execution**: The specialized agent reviews code

## Project Structure

```
src/stem_agent/
├── core/           # Agent, state machine, journal, config
├── phases/         # Sensing, planning, specialization, validation
├── capabilities/   # Registry, tools, prompt library
├── evaluation/     # Metrics, benchmark, comparator, fixtures
├── ports/          # LLM and storage protocols
└── adapters/       # OpenAI and JSON file implementations
```

## Evaluation

The benchmark corpus contains 20 Python code samples with ground-truth labels:
- 5 logic bugs (off-by-one, wrong operators, missing null checks)
- 4 security vulnerabilities (SQL injection, path traversal, hardcoded credentials)
- 4 code smells (deep nesting, god functions, dead code)
- 2 performance issues (N+1 queries, unnecessary copies)
- 5 clean code samples (adversarial true negatives that look suspicious but are correct)

Precision, recall, F1, and specificity, measured before and after specialization on the same corpus.

142 deterministic tests, no network.
