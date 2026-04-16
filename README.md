# sql-rl-env: A Reinforcement Learning Environment for SQL Query Generation

*Author: Kartik Munjal*

A Gymnasium-compatible reinforcement learning environment for SQL query generation via slot-filling, designed for studying reward signal design and reward hacking in structured task domains.

---

## Research Overview

This project implements a **slot-filling Markov decision process** for SQL query generation over a small e-commerce database. The environment is designed explicitly to study:

1. **Reward signal design**: Four reward functions with different trade-offs between semantic richness, signal density, and hacking vulnerability
2. **Reward hacking analysis**: Eight documented hacking scenarios with formal detection methods
3. **Curriculum difficulty**: Five tasks of increasing SQL complexity (simple SELECT → window functions)
4. **Baseline comparisons**: Three agents (random, rule-based, REINFORCE) across all task/reward combinations

The key research insight is that by using a **schema-grounded slot-filling** formulation (rather than free-text token generation), all reward hacking scenarios operate on *syntactically and schema-valid SQL*, making them analytically tractable.

---

## Environment Design

### State Space

```
obs["schema"]        float32 (20, 8)    — Column feature matrix: [table one-hot | dtype one-hot]
obs["nl_embedding"]  float32 (128,)     — Bag-of-words NL query encoding
obs["partial_sql"]   float32 (26, 32)   — One-hot slot fillings per phase
obs["current_phase"] int32   (1,)       — Current build phase index
```

### Action Space

```
Discrete(N_MAX_ACTIONS=32)
```

Actions are schema-grounded and phase-specific. The valid action set (provided via `info["action_mask"]`) changes at each step based on:
- The current build phase (SELECT_COLS, FROM_TABLE, WHERE_COL, etc.)
- Previously filled slots (join key masking, type-safe WHERE constraints)
- Schema structure (only existing tables/columns are valid actions)

### Reward Signals

| Name | R | Signal | Primary Use |
|------|---|--------|-------------|
| Exact Match | R1 | Binary {0, 1} | Evaluation |
| Execution Match | R2 | {-0.1, 0, 0.5, 1.0} | Evaluation + R4 input |
| Partial Credit | R3 | [0, 1] continuous | Dense training signal |
| Composite | R4 | [−0.1, 1] | Primary training signal |

R4 = 0.50·R2 + 0.30·R3 + 0.15·R1 − 0.05·efficiency\_penalty

---

## Five SQL Tasks

| Task | Complexity | SQL Construct | Episode Steps |
|------|-----------|---------------|---------------|
| 1: Simple SELECT | ★☆☆☆☆ | SELECT + WHERE | 5 |
| 2: Aggregation | ★★☆☆☆ | GROUP BY + HAVING + ORDER BY | 8 |
| 3: INNER JOIN | ★★★☆☆ | Two-table join with FK selection | 8 |
| 4: IN Subquery | ★★★★☆ | Nested sub-MDP with 10 phases | 10 |
| 5: Window Function | ★★★★★ | RANK/ROW_NUMBER + PARTITION BY | 9 |

All tasks use a shared e-commerce database: `customers`, `products`, `orders`, `order_items` (50, 20, 200, 600 rows respectively).

---

## Reward Hacking Analysis

Eight hacking scenarios are documented in [`research_notes/reward_hacking_report.md`](research_notes/reward_hacking_report.md):

| ID | Reward | Mechanism | Detection Signal |
|----|--------|-----------|-----------------|
| H1.1 | R1 | Cross-join passes alias normalisation | Row count KL drift |
| H1.2 | R1 | SELECT * exploitation | Column coverage trend |
| H2.1 | R2 | Large result set Jaccard inflation | Row distribution KL |
| H2.2 | R2 | Timeout stagnation (no-gradient trap) | Learning curve plateau |
| H3.1 | R3 | Column flooding for Jaccard numerator | Coverage Spearman r |
| H3.2 | R3 | WHERE >= 0 operator monoculture | Operator entropy |
| H4.1 | R4 | High w_partial local optimum | exec vs partial gap |
| H4.2 | R4 | LIMIT suppresses efficiency penalty | Pre-LIMIT COUNT(*) |

Hacking is detected in real-time by `src/analysis/reward_hacking_detector.py` using three statistical signals (KL divergence, Shannon entropy, Spearman rank correlation). A composite alert fires when 2/3 signals trigger simultaneously.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Seed the database

```bash
python scripts/seed_db.py
```

This creates `data/ecommerce.db` (deterministic, seed=42) and `configs/nl_vocab.json`.

### 3. Run a sanity check (random agent)

```bash
python scripts/run_experiment.py --dry_run
```

### 4. Train REINFORCE on Task 1

```bash
python scripts/run_experiment.py --task task_01_simple --reward composite --episodes 2000
```

### 5. Run full evaluation table

```bash
python scripts/evaluate_agents.py --episodes 500
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
sql-rl-env/
├── src/
│   ├── env/
│   │   ├── sql_env.py          # Gymnasium.Env — central contract
│   │   ├── action_space.py     # Schema-grounded hierarchical action space
│   │   ├── state.py            # SQLState + NLEncoder
│   │   ├── schema.py           # Schema loader + FK graph
│   │   └── executor.py         # Safe SQLite executor (timeout + pre-LIMIT COUNT)
│   ├── tasks/
│   │   └── base.py             # SQLTask + TaskRegistry + SQL assemblers
│   ├── rewards/
│   │   ├── exact_match.py      # R1: normalised string comparison
│   │   ├── execution_match.py  # R2: result-set Jaccard comparison
│   │   ├── partial_credit.py   # R3: component-wise scoring
│   │   └── composite.py        # R4: weighted combination + efficiency penalty
│   ├── agents/
│   │   ├── random_agent.py     # Lower-bound baseline
│   │   ├── rule_agent.py       # Keyword-matching non-learning upper bound
│   │   └── reinforce_agent.py  # REINFORCE + baseline network (PyTorch)
│   └── analysis/
│       └── reward_hacking_detector.py  # 3-signal statistical detector
├── configs/
│   ├── env_config.yaml
│   ├── task_config.yaml
│   ├── reward_config.yaml
│   ├── agent_config.yaml
│   └── tasks/                  # NL queries + gold SQL for each task
├── scripts/
│   ├── seed_db.py              # Deterministic database seeder
│   ├── run_experiment.py       # REINFORCE training loop
│   └── evaluate_agents.py      # Cross-agent comparison table
├── tests/
│   ├── test_rewards.py         # Reward function unit tests + H3.1 regression
│   ├── test_actions.py         # Action mask correctness
│   └── test_tasks.py           # SQL assembly and task loading
└── research_notes/
    ├── design_decisions.md     # Rationale for 8 architectural decisions
    └── reward_hacking_report.md  # Formal analysis of 8 hacking scenarios
```

---

## Key Design Decisions

Full rationale in [`research_notes/design_decisions.md`](research_notes/design_decisions.md).

**D1** — SQL domain chosen for executable semantics and rich partial-correctness structure.

**D2** — Slot-filling MDP (not token-level) for tractable masking, interpretable hacking, and clean separation from language model capabilities.

**D3** — Terminal-only reward with gamma=1.0 to avoid artificial decomposition of joint SQL decisions and penalisation of longer tasks.

**D4** — Schema-grounded masking at environment level to eliminate syntactic hacking and focus analysis on semantically valid but semantically wrong queries.

**D5** — REINFORCE (not PPO) for transparency: failure modes are directly observable and documentable, appropriate for a reward signal analysis project.

---

## Citing This Work

```bibtex
@misc{munjal2026sqlrlenv,
  title  = {sql-rl-env: A Gymnasium Environment for Reward Signal Design in SQL Query Generation},
  author = {Kartik Munjal},
  year   = {2026},
  url    = {https://github.com/kartikmunjal/rl-env}
}
```
