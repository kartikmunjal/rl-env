# Design Decisions — sql-rl-env

*Author: Kartik Munjal*

This document records the rationale behind every major architectural decision in this project. It is written in the style of a research paper's "Design Choices" section and serves as the primary reference for understanding *why* the environment is structured the way it is.

---

## D1. Domain Choice: SQL Query Generation

### Why SQL, not navigation or tool-call sequencing?

Three structured task domains were evaluated:

1. **Text-based grid navigation**: Clean action space, but reward hacking scenarios are limited to teleportation exploits (rarely observed in practice). The semantics of "success" are binary and unambiguous, which leaves no room for partial-credit reward design.

2. **Tool-call sequencing**: Rich multi-step structure, but "correctness" is poorly defined without a ground-truth oracle. Reward design devolves into manual scoring rubrics, which is the problem we want to study, not a given.

3. **SQL query generation**: The correct choice. SQL has:
   - **Executable semantics**: we can run both predicted and gold queries and compare results
   - **Formal structure**: the grammar defines exactly what constitutes valid SQL
   - **Rich partial-correctness notion**: a query can be correct on some clauses but not others
   - **Documented reward hacking literature**: the text-to-SQL community has catalogued dozens of hacking scenarios (cross-joins, trivial queries, annotation artifacts)

### Why not use an LLM to generate SQL?

This project is explicitly about the *reward signal* and the *environment structure*, not the policy architecture. Using an LLM as the policy would conflate representation learning with reward signal issues, making it impossible to cleanly attribute failure modes. The slot-filling MDP uses a simple linear policy that lets reward signal issues dominate experimental outcomes.

---

## D2. Slot-Filling MDP vs. Token-Level Generation

### The rejected alternative: token-level generation

The most natural formulation of SQL generation as RL is token-by-token generation where each token is an action from a vocabulary of ~32,000 tokens. This is rejected for three reasons:

1. **Intractable masking**: Determining which tokens are valid at each position requires a partial SQL parser running at every step. This is computationally expensive and fragile.

2. **Opaque reward hacking**: In a token-level MDP, reward hacking scenarios are difficult to identify because the connection between individual token choices and query semantics is non-local.

3. **Conflation with language modeling**: Token-level SQL generation is essentially instruction-following fine-tuning of an LLM. The reward signal analysis becomes entangled with the model's pre-existing SQL knowledge.

### The chosen approach: hierarchical slot-filling

The SQL query is decomposed into a fixed set of semantic *slots* (SELECT_COLS, FROM_TABLE, WHERE_COL, WHERE_OP, WHERE_VAL, etc.). Each slot is filled by a discrete action drawn from a schema-grounded vocabulary. An episode processes one slot per step.

**Properties of this formulation**:
- Action space is bounded and enumerable at every step
- All reward hacking scenarios operate on semantically valid SQL (no syntactic tricks)
- The human-interpretable slot structure allows reward hacking to be identified and categorised analytically
- Short episodes (5-13 steps) make credit assignment tractable

**Limitation**: The slot-filling formulation cannot express all SQL queries (e.g., queries with complex arithmetic expressions, lateral joins, or recursive CTEs). This is acceptable because the research goal is reward signal analysis, not state-of-the-art SQL accuracy.

---

## D3. Terminal-Only Reward

### The rejected alternative: step-level rewards

A step-level reward (e.g., +0.1 for choosing the correct FROM_TABLE) is tempting because it provides dense gradient signal. It is rejected for the following reasons:

1. **Artificial decomposition**: A SQL query is semantically a joint object. Correct table selection + wrong join key produces a syntactically valid but semantically wrong query. Awarding +0.1 for the table choice decouples what should be a joint decision.

2. **Credit assignment artifacts**: The agent learns to maximise per-step rewards independently of query correctness. A pathological policy could achieve high per-step reward while consistently producing queries that execute incorrectly.

3. **Gameable in ways terminal reward is not**: With per-step rewards, the agent can learn "safe" early phases (high per-step reward) while exploiting later phases where rewards are harder to game. Terminal reward forces the agent to reason about the full query construction jointly.

### Design choice: terminal reward only, gamma=1.0

The reward is issued exclusively at episode termination (when all active phases are filled and the SQL is assembled and executed). The discount factor gamma=1.0 (no discounting) for two reasons:

1. **Short episodes**: Episodes are 5-13 steps. At gamma=0.99, the effective discount over 13 steps is 0.99^13 ≈ 0.88 — negligible but non-zero. At gamma=0.9, it would be 0.9^13 ≈ 0.25, which meaningfully penalises long episodes.

2. **Task length fairness**: With gamma < 1.0, Tasks 4-5 (longer episodes) would be penalised relative to Tasks 1-2. Since the reward is the same type of signal across all tasks, episodes of different lengths should not be treated differently.

---

## D4. Schema-Grounded Masking at the Environment Level

### The key design invariant

Action masking is performed by the environment, not by the agent. The environment's `info["action_mask"]` at every step contains exactly the legal actions for the current phase, derived from the schema.

**Consequence**: An agent cannot produce SQL that references non-existent tables or columns. Invalid schema references are not an error the agent can learn to avoid — they simply do not exist as actions.

### Why this is a deliberate research choice

By eliminating schema-invalid SQL from the action space, we narrow the space of observable hacking behaviors to **semantically valid but semantically wrong SQL**. This is exactly the interesting research territory:
- The agent selects the correct table but the wrong column (partial credit distinguishes this)
- The agent selects the correct columns but wrong operator (where_score captures this)
- The agent returns a large result set by selecting all rows (efficiency penalty triggers)

If schema-invalid SQL were allowed, the agent would trivially discover that writing `SELECT foo FROM bar WHERE baz = 3` produces a consistent execution error (0.0 reward) — a degenerate fixed point that provides no research insight.

### Masking implementation

The `HierarchicalActionSpace.get_action_spec()` method returns an `ActionSpec` with a binary numpy mask of shape `(N_MAX_ACTIONS=32,)`. The mask is phase-specific and context-sensitive:
- After `FROM_TABLE = 'customers'`, the `WHERE_COL` mask shows only `customers.*` columns
- After `JOIN_TABLE = 'orders'`, the `WHERE_COL` mask adds `orders.*` columns
- The `JOIN_KEY_LEFT/RIGHT` masks show only FK-compatible column pairs

The environment's `step()` raises `AssertionError` if the agent selects a masked action. This surfaces bugs in agent implementations immediately rather than silently producing wrong SQL.

---

## D5. Algorithm Choice: REINFORCE over PPO

### Why not PPO?

Proximal Policy Optimisation (PPO) is the standard choice for production RLHF systems. For this research environment, it is overkill for three reasons:

1. **Scale mismatch**: PPO's KL constraint and clipping hyperparameters are designed for large models (billions of parameters) where a single gradient step can cause catastrophic policy shifts. Our policy network has ~150,000 parameters and a 590-dim observation space. Vanilla gradient steps are safe.

2. **Research transparency**: REINFORCE's failure modes (high variance, slow convergence on long episodes) are directly observable in learning curves and documentable. PPO's clipping and KL penalty obscure these failure modes behind additional hyperparameters.

3. **No experience replay requirement**: PPO requires a replay buffer and importance sampling. REINFORCE uses on-policy trajectories, which is correct for an environment where the reward function changes between experiments (reward signal ablation study).

### Entropy bonus

An entropy bonus `-entropy_coeff * H(pi)` is added to encourage exploration. Without it, the policy collapses to a degenerate deterministic policy (always selecting the first valid action) early in training, before the reward signal has provided useful gradient information. The coefficient `entropy_coeff=0.01` is small enough to not overwhelm the policy gradient.

---

## D6. Reward Function Hierarchy

The four reward functions (R1-R4) are designed with different trade-offs:

| Reward | Signal Density | Semantic Correctness | Hacking Vulnerability | Primary Use |
|--------|---------------|---------------------|-----------------------|-------------|
| R1 Exact Match | Sparse (0/1) | Syntactic proxy | H1.1, H1.2 | Evaluation metric |
| R2 Execution Match | Medium (0/0.5/1) | High | H2.1, H2.2 | Evaluation + training input |
| R3 Partial Credit | Dense [0,1] | Low-medium | H3.1, H3.2 | Warm-start training |
| R4 Composite | Dense [0,1] | Medium-high | H4.1, H4.2 | Primary training signal |

The composite reward R4 is used for REINFORCE training because it combines the semantic richness of R2 with the density of R3, while the efficiency penalty and error penalties mitigate the most critical hacking scenarios.

---

## D7. Database Design

The e-commerce database (`data/ecommerce.db`) is designed with the following constraints:

1. **Exactly 4 tables**: Customers, products, orders, order_items. This gives a manageable 20-column schema that can be encoded as a fixed-size observation tensor without padding complexity.

2. **Deterministic seeding**: `scripts/seed_db.py` uses `random.seed(42)`. Every run produces bit-identical data. This ensures that reward values are reproducible across experiments.

3. **No synthetic artifacts**: The data distributions (city counts, tier distributions, order statuses) are realistic enough that queries on the data return non-trivial result sets. A completely uniform distribution would make result-set comparisons degenerate.

4. **FK relationships designed for 3 join tasks**: The orders table has a FK to customers (task 3), order_items has FKs to both orders and products (tasks 3, 4). This allows 3 distinct join scenarios without adding more tables.

---

## D8. Observation Space Design

The observation is a Dict space with four components:

```
schema:          float32 (20, 8)   — column feature matrix
nl_embedding:    float32 (128,)    — bag-of-words NL encoding
partial_sql:     float32 (26, 32)  — one-hot slot fillings
current_phase:   int32   (1,)      — phase index
```

**Why bag-of-words, not a learned embedding?**
A learned embedding would require a separate NL encoder that is trained jointly with the RL policy. This creates a second confound: differences in performance across reward functions could be due to differences in how well the NL encoder trained, not differences in the reward signal. Bag-of-words NL encoding is fixed, making the only variable the reward signal itself.

**Why one-hot partial SQL, not raw indices?**
One-hot encoding provides a continuous, differentiable signal to the policy network. Raw integer indices would create arbitrary ordinal relationships between action tokens (action 0 ≠ "less than" action 1).
