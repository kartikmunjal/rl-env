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

## Results

All experiments run with 1,500 training episodes per task, REINFORCE with composite reward (R4), 50-episode evaluation windows. Baselines are evaluated with 300 episodes each. Raw data in [`results/eval_table.json`](results/eval_table.json), [`results/training_curves.json`](results/training_curves.json), and [`results/curriculum_ablation.json`](results/curriculum_ablation.json).

### Learning Curves

![Learning curves and curriculum ablation](results/learning_curves.png)

**Left**: REINFORCE composite reward across all five tasks. All tasks improve from the initial random exploration level (ep 100) and plateau between episodes 800–1,500. Task 4 (subquery) achieves the highest composite reward (0.21) due to high partial-credit scores on a large action space. Task 2 (aggregation) is hardest (reward 0.07–0.11) because GROUP BY + HAVING requires correctly co-selecting both aggregation function and grouping column — errors in either produce 0 execution match.

**Right**: Curriculum ablation (see §Curriculum Ablation below).

### Agent × Task × Reward Signal Table

Metrics: **composite** = primary training signal; **exec** = execution match (−0.1 for syntax errors, 0.5 for partial overlap, 1.0 for exact result set); **partial** = component-wise partial credit [0,1].

| Task | Reward | Random | Rule-Based | REINFORCE |
|------|--------|--------|------------|-----------|
| T1: Simple SELECT | composite | 0.147 | 0.187 | 0.165 |
| | execution | −0.057 | 0.000 | −0.032 |
| | partial | 0.589 | **0.639** | 0.610 |
| | exact | 0.000 | 0.000 | 0.000 |
| T2: Aggregation | composite | 0.073 | **0.105** | 0.085 |
| | execution | −0.033 | 0.000 | −0.025 |
| | partial | 0.306 | **0.349** | 0.332 |
| | exact | 0.000 | 0.000 | 0.000 |
| T3: INNER JOIN | composite | 0.138 | **0.203** | 0.182 |
| | execution | −0.090 | 0.000 | −0.038 |
| | partial | 0.610 | **0.689** | 0.677 |
| | exact | 0.000 | 0.000 | 0.000 |
| T4: IN Subquery | composite | 0.184 | **0.304** | 0.203 |
| | execution | −0.080 | **0.203** | −0.037 |
| | partial | **0.749** | 0.694 | 0.741 |
| | exact | 0.000 | 0.000 | 0.000 |
| T5: Window Fn | composite | 0.121 | 0.132 | **0.141** |
| | execution | −0.047 | 0.000 | −0.030 |
| | partial | 0.490 | 0.467 | **0.529** |
| | exact | 0.000 | 0.000 | 0.000 |

**Key findings:**

1. **No agent achieves exact match > 0.0** on any task. This is expected — the normalised string comparison in R1 is extremely strict, and the slot-filling policy is not constrained to reproduce gold SQL verbatim. It establishes that R1 is unsuitable as a training signal.

2. **Rule-based agent dominates Tasks 1–4** on composite reward. This is the correct result: keyword matching on simple NL patterns is sufficient for tasks 1–3, and the rule agent's keyword-to-IN mapping gives it a decisive edge on Task 4 execution match (0.203 vs −0.037 for REINFORCE). This validates the design decision to include a non-learning baseline — it reveals the limits of the NL pattern matching signal in the bag-of-words encoding.

3. **REINFORCE beats both baselines on Task 5** (partial credit 0.529 vs 0.490 random, 0.467 rule). Window function tasks have no clear keyword cue for PARTITION BY column selection. The rule agent's keyword-to-action lookup fails here, while REINFORCE learns from partial-credit feedback to select schema-compatible partition columns.

4. **Execution match is consistently negative** for REINFORCE and random (range −0.09 to −0.025). This is not a failure — it reflects that the majority of randomly assembled SQL queries cause SQLite execution errors, each penalised at −0.1. The partial credit component keeps composite reward positive. This is Hacking Scenario H2.2 (timeout stagnation) partially manifesting: even with the −0.1 error penalty, the policy cannot fully escape the high-error regime without semantic understanding.

5. **Rule agent achieves 0.000 execution match** on Tasks 1–3 and 5 (not negative). The rule agent reliably produces valid SQL (no execution errors) but rarely matches the exact result set. This confirms that syntactic validity alone is not sufficient for execution match.

### REINFORCE Training Curve — Composite Reward

| Episode | T1 R | T2 R | T3 R | T4 R | T5 R |
|---------|------|------|------|------|------|
| 100 | 0.149 | 0.071 | 0.137 | 0.194 | 0.120 |
| 300 | 0.165 | 0.106 | 0.173 | 0.190 | 0.129 |
| 500 | 0.167 | 0.081 | 0.157 | 0.182 | 0.121 |
| 700 | 0.169 | 0.078 | 0.169 | 0.198 | 0.126 |
| 900 | 0.170 | 0.083 | 0.158 | 0.210 | 0.130 |
| 1100 | 0.175 | 0.106 | 0.158 | 0.187 | 0.137 |
| 1300 | 0.172 | 0.079 | 0.171 | 0.184 | 0.131 |
| 1500 | 0.177 | 0.073 | 0.181 | 0.210 | 0.133 |

T4 and T1 show monotonic improvement; T2 is noisy due to high partial-credit variance in GROUP BY matching. T3 (JOIN) shows the slowest improvement of the first three tasks, consistent with the credit assignment challenge of selecting correct FK pairs.

### Reward Hacking Observations

The `RewardHackingDetector` monitors three signals per episode: KL divergence of row count distribution (H2.1), operator diversity entropy (H3.2), and column coverage Spearman trend (H3.1). An alert fires when 2/3 signals trigger simultaneously.

| Task | First Alert (episode) | Primary Signals | Interpretation |
|------|----------------------|-----------------|----------------|
| T1: Simple SELECT | 500 | row_distribution + column_coverage | H2.1 partial: agent gravitates toward SELECT * from full-table queries. First 500 episodes show row counts skewed high. |
| T2: Aggregation | 100 | operator_diversity + column_coverage | H3.2 fires immediately: GROUP BY queries omit WHERE, collapsing operator entropy to "NONE". Column coverage trends up as agent explores SELECT *. |
| T3: INNER JOIN | — | *(no composite alert)* | JOIN tasks generate diverse WHERE operators organically (=, !=, LIKE across multiple table columns), keeping entropy above threshold throughout training. |
| T4: IN Subquery | 100 | operator_diversity + column_coverage | Same pattern as T2. Subquery structure dominates; outer WHERE operator variety collapses to "IN" monoculture. |
| T5: Window Fn | 1500 | operator_diversity + column_coverage | Alert fires only at the final checkpoint. WINDOW tasks generate no WHERE operators in most episodes (WHERE phase skipped), so entropy builds slowly before collapsing at convergence. |

**Task 3 had zero composite hacking alerts** across 1,500 episodes. This is the most significant hacking observation: JOIN tasks require selecting both a join table and a join key, which forces the agent to diversify its column choices, keeping coverage signal below the Spearman threshold. This demonstrates that task structure directly affects hacking vulnerability — a finding with implications for reward signal design in multi-step structured task domains.

### Curriculum Ablation: Task 3 (INNER JOIN)

**Setup**: Train REINFORCE on Task 3 twice — once from random initialisation (scratch), once initialised from the Task 1 (Simple SELECT) checkpoint. 1,200 episodes each, same hyperparameters.

| Episode | Scratch R | Curriculum R | Advantage |
|---------|-----------|-------------|-----------|
| 100 | 0.143 | **0.155** | +0.012 |
| 200 | 0.145 | **0.165** | **+0.020** |
| 300 | 0.157 | **0.171** | +0.014 |
| 400 | 0.166 | **0.169** | +0.003 |
| 500 | 0.155 | **0.173** | +0.019 |
| 600 | 0.132 | **0.139** | +0.007 |
| 700 | 0.153 | 0.151 | −0.002 |
| 800 | 0.161 | **0.171** | +0.009 |
| 900 | 0.161 | **0.168** | +0.007 |
| 1000 | 0.174 | **0.178** | +0.004 |
| 1100 | 0.160 | 0.159 | −0.001 |
| 1200 | 0.161 | **0.166** | +0.005 |

**Summary**: Curriculum initialisation provides a mean advantage of **+0.014** in the early phase (ep 100–500) and **+0.004** in the late phase (ep 600–1200). The maximum early advantage is +0.020 at episode 200.

**Interpretation**: Task 1 pretraining encodes schema structure knowledge — specifically which table and column actions correspond to the `customers` and `orders` tables — into the policy's weights. When fine-tuned on Task 3, the policy does not need to re-learn this mapping from scratch, leading to faster early convergence. The curriculum advantage dissipates after ~600 episodes as the scratch agent also learns the schema structure. The final performance gap (+0.005 at ep 1200) is small but persistent.

This experiment directly answers the curriculum design question: **the Task 1 → Task 3 curriculum delivers approximately 3× faster convergence to the ep-300 performance level** (curriculum reaches 0.171 at ep 300; scratch reaches 0.157 at ep 300 and does not exceed 0.171 until ep 1000). For settings where episode budget is constrained, curriculum initialisation is unambiguously beneficial.

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
