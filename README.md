# Stem Agent

[![CI](https://github.com/qflen/stem-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/qflen/stem-agent/actions/workflows/ci.yml)

A self-specializing AI agent that evolves from an undifferentiated core into a task-specific specialist through guided differentiation.

Full writeup: [`docs/writeup.pdf`](docs/writeup.pdf) (4 pages).

https://github.com/user-attachments/assets/caa352ec-5c6c-4c03-8fbd-0974e7d68d0a

The clip above replays a recorded run (no live API calls). The current
headline is computed across **3 seeds × 2 domains** (live OpenAI runs,
artifacts under [`docs/example_run/seeds/`](docs/example_run/seeds/)):

- Code-quality, 11-sample validation: F1 = **0.837 [0.800, 0.889]**, specificity 0.444.
- Security audit, 4-sample validation: F1 = **0.794 [0.667, 0.857]**, specificity 0.000.

Cross-domain 2×2 and Cohen's κ adjudicator results are in
[`docs/writeup.pdf`](docs/writeup.pdf) (Tables 2–3). Total live spend
across the multi-seed batch was ~$1.30.

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

# Multi-seed run (deterministic 11/4/4 partition by seed)
stem-agent differentiate --seeds 3 --domain code_quality_analysis --store-prompts

# Render the headline mean ± 95% CI table from a multi-seed dir
stem-agent report docs/example_run/seeds/cq

# Print the with-vs-without-capability-generation 2×4 ablation grid
stem-agent eval-ablation

# Replay a stored prompt by its journal hash
stem-agent replay <prompt_hash>

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

The benchmark corpus contains 19 Python code samples with ground-truth
labels, partitioned 11 / 4 / 4 into validation, holdout, and probe
slices by a stable seed-deterministic hash:

- 4 logic bugs (off-by-one, wrong operators, missing null checks, broken boolean precedence)
- 4 security vulnerabilities (SQL injection, path traversal, hardcoded credentials, unsafe eval)
- 4 code smells (deep nesting, god functions, dead code, magic-number stack)
- 2 performance issues (N+1 queries, unnecessary copies)
- 5 clean code samples (adversarial true negatives that look suspicious but are correct)

Precision, recall, F1, and specificity are measured on the validation
slice; the holdout drives the empirical-admission gate on
LLM-generated capabilities; the probe is fed unlabelled into sensing.

278 deterministic tests, no network.
