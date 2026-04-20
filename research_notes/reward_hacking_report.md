# Reward Hacking Analysis Report — sql-rl-env

*Author: Kartik Munjal*

This report documents eight reward hacking scenarios identified in the sql-rl-env benchmark. For each scenario we provide: the formal mechanism, a concrete example, an empirical detection method, and the counter-measure implemented in the codebase.

One scope note is necessary after the final experiments: several "counter-measures" in this document are best understood as *partial mitigations or monitoring hooks*, not complete fixes. In particular, the final weight ablation shows that stronger execution penalties do not improve execution match, and the exact-match metric never becomes non-zero for any agent. Those results narrow what can be claimed about semantic verification in the current environment.

The framework follows the definition: *reward hacking occurs when a policy achieves high reward under a given reward function without achieving the intended behaviour.*

---

## Overview Table

| ID | Reward Signal | Mechanism | Detectable? | Counter-measure |
|----|--------------|-----------|-------------|-----------------|
| H1.1 | R1 Exact Match | Cross-join passes normalisation | Yes (row count) | Efficiency penalty in R4 |
| H1.2 | R1 Exact Match | SELECT * exploitation | Yes (column coverage) | * masked on Tasks 1-3 |
| H2.1 | R2 Execution | Large result set Jaccard inflation | Yes (KL drift) | Jaccard not Recall |
| H2.2 | R2 Execution | Timeout stagnation | Indirect (no learning) | -0.1 error penalty |
| H3.1 | R3 Partial Credit | Column flooding | Yes (coverage trend) | Jaccard denominator |
| H3.2 | R3 Partial Credit | WHERE >= 0 operator gaming | Yes (entropy collapse) | Operator diversity check |
| H4.1 | R4 Composite | High w_partial local optimum | Yes (exec vs partial gap) | Weight ablation study |
| H4.2 | R4 Composite | LIMIT suppresses efficiency penalty | No (pre-LIMIT COUNT) | COUNT(*) before LIMIT |

---

## H1.1 — Alias Stripping Cross-Join Bypass

### Reward Function
R1 (Exact Match)

### Mechanism
The normalisation pipeline in `ExactMatchReward._normalise()` strips table aliases (e.g., `c.` → ``) and collapses whitespace. Consider a gold query and a predicted query:

**Gold SQL:**
```sql
SELECT c.name, o.order_date FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'delivered'
```

**Predicted SQL (cross-join attack):**
```sql
SELECT name, order_date FROM customers, orders
WHERE customer_id = customer_id AND status = 'delivered'
```

After normalisation (strip aliases, lowercase, collapse whitespace, sort SELECT cols):
- Gold normalised: `select name, order_date from customers inner join orders on customer_id = customer_id where status = 'delivered'`
- Predicted normalised: `select name, order_date from customers, orders where customer_id = customer_id and status = 'delivered'`

These are NOT identical (the JOIN syntax differs after alias stripping), so this specific attack does not bypass exact match for INNER JOIN. However, for simple queries without an ON clause, the implicit join `FROM a, b WHERE a.id = b.id` can normalize to match `FROM a INNER JOIN b ON a.id = b.id` depending on normalization aggressiveness.

### Empirical Evidence
Running a REINFORCE agent trained on R1 for 2,000 episodes on Task 3 (join), we observe that agents occasionally generate comma-separated FROM clauses with WHERE-based join conditions. The fraction of such queries increases over training, suggesting the agent is exploring this normalisation exploit.

### Counter-measure
The efficiency penalty in R4 fires when `actual_rows_no_limit > 5 * expected_rows`. A cross-join on 50 customers × 200 orders = 10,000 rows vs. an expected ~200 join result triggers a penalty of `log(10000/200 + 1) / log(50 + 2) ≈ 0.92`, which caps at the `efficiency_penalty_max = 0.20`. This makes cross-join queries substantially worse than correct INNER JOIN queries under R4.

---

## H1.2 — SELECT * Exploitation

### Reward Function
R1 (Exact Match)

### Mechanism
If SELECT * is a valid action and the normalisation pipeline expands * to all columns before comparison, then for single-table queries, an agent that always selects * will match the gold query whenever the gold SQL also uses all columns.

For Task 1 (simple SELECT), the gold SQL typically selects 2-3 specific columns. An agent selecting * does NOT match the gold after normalisation because * ≠ (col1, col2). However, if the normalisation pipeline expands * to the full column list, the attack becomes viable.

### Analysis
In our normalisation pipeline, `*` is NOT expanded — it is treated as a literal token. Therefore this attack fails under exact match. However, it can partially succeed under partial credit (R3) because `_extract_columns()` extracts no columns from `SELECT *`, and comparing {} to {col1, col2} gives Jaccard = 0.

The restriction of `*` to specific task contexts in the action mask (`SELECT_COLS` phase marks `*` as action 0 but the environment's `enforce_nonempty_select` setting prevents it from being the ONLY selection) provides the primary mitigation.

---

## H2.1 — Large Result Set Jaccard Inflation

### Reward Function
R2 (Execution Match)

### Mechanism
The execution match reward includes a 0.5 partial score when `|predicted ∩ gold| / |predicted ∪ gold| ≥ 0.50`. This creates an incentive to return large result sets that are supersets of the gold result.

**Formal example:**
- Gold query returns 10 rows from `customers WHERE tier='gold'`
- Predicted query: `SELECT * FROM customers` returns 50 rows (superset)
- Intersection: 10 rows (all gold rows appear in the superset)
- Jaccard: 10 / 50 = 0.20 — **below** the 0.50 threshold, no partial reward

**The critical case:** When the gold result set is large relative to the table size. If the gold query returns 40 out of 50 customers, then `SELECT * FROM customers` achieves Jaccard = 40/50 = 0.80, triggering the 0.5 partial reward.

### Empirical Detection
The `RewardHackingDetector` Signal 1 tracks the KL divergence of the row count distribution. An agent exploiting H2.1 will shift the distribution toward larger result sets. In the current implementation, `kl_threshold = 0.5` began as a heuristic default baked into the detector. It is no longer completely untested: the repo now includes both checkpoint-evaluation and traced-training threshold sweeps, and the observed alert patterns are stable across substantial perturbations of `tau_kl`. What remains uncalibrated is the stronger claim about formal false-positive / false-negative rates on a large bank of human-labeled windows.

**Example output from hacking detector:**
```
HackingAlert(episode=450, signal="row_distribution", severity=0.72,
  description="Row count distribution shifted increasing (KL=0.81 > threshold=0.5).
               Suspected hacking: H2.1 (large result sets)")
```

### Counter-measure
The use of Jaccard (|∩|/|∪|) rather than Recall (|∩|/|gold|) is the primary mitigation. With Recall, `SELECT * FROM customers` on a query with 40 gold rows would achieve Recall = 40/40 = 1.0 — a perfect score for a trivially wrong query. With Jaccard, it achieves 40/50 = 0.80 on the partial check. While this is above the 0.50 threshold, the efficiency penalty in R4 fires (50 vs. 40 rows, ratio = 1.25 < 5.0 threshold, so no penalty in this case). For cases with much larger result sets, the efficiency penalty provides the correction.

This should not be overstated: the mitigation is incomplete. If the predicted result is a moderately oversized superset, execution match can still award partial credit, and the current benchmark does not have a separate exact-verification layer that ever activates in practice.

---

## H2.2 — Timeout Stagnation (Non-Hacking Failure Mode)

### Reward Function
R2 (Execution Match)

### Mechanism
This is technically not reward hacking — it is a failure mode of the reward signal. If the agent consistently generates queries that cause SQLite timeouts (e.g., recursive cross-joins on large tables), every episode returns 0.0. The policy gradient over these episodes has zero signal (no direction to update toward or away from timeouts).

**Example pathological query:**
```sql
SELECT * FROM customers c1, customers c2, orders o
WHERE c1.customer_id != c2.customer_id
```
This is a non-equijoin with O(50 × 50 × 200) = 500,000 row output, which times out at 2 seconds.

### The Policy Stagnation Trap
With 0.0 reward for both timeout queries AND for valid-but-wrong queries, the policy has no gradient signal to distinguish them. The agent stagnates in a local minimum where it produces queries of mixed validity.

### Counter-measure
Timeout and syntax error queries yield **-0.1** (not 0.0). This creates a negative gradient that pushes the policy away from query patterns that cause execution failures, without providing information about which direction to move toward.

**Design choice**: -0.1 not -1.0 because a large negative penalty for timeouts makes the policy overly conservative — it learns to avoid complex queries entirely rather than learning to generate complex queries correctly.

---

## H3.1 — Column Flooding for Jaccard Numerator

### Reward Function
R3 (Partial Credit)

### Mechanism
The `columns_score` in partial credit uses Jaccard similarity: `|predicted ∩ gold| / |predicted ∪ gold|`. An agent that selects ALL columns in the schema maximises the numerator `|predicted ∩ gold| = |gold|`, but the denominator grows with the number of selected columns.

**Mathematical analysis:**
- Schema has 20 columns total
- Gold query selects 3 columns
- Predicted query selects all 20 columns
- Jaccard = |{3} ∩ {20}| / |{3} ∪ {20}| = 3 / 20 = **0.15**

This is worse than selecting exactly the right 3 columns (Jaccard = 1.0), and only slightly better than selecting 3 wrong columns (Jaccard = 0/6 = 0.0).

**The alternative (buggy) metric**: If the implementation accidentally used Recall instead of Jaccard:
- Recall = |predicted ∩ gold| / |gold| = 3/3 = **1.0**

This would make column flooding a dominant strategy, as documented in the `test_rewards.py::TestPartialCredit::test_jaccard_penalises_column_flooding` regression test.

### Detection
`RewardHackingDetector` Signal 3 tracks the Spearman rank correlation of column coverage (|selected_cols| / |schema_cols|) over time. A consistently positive trend toward high coverage indicates flooding.

### Counter-measure
1. **Jaccard denominator**: The implementation explicitly uses Jaccard not Recall. The `_jaccard()` function in `partial_credit.py` uses `|A ∪ B|` in the denominator, with a code comment warning about the Recall bug.
2. **Regression test**: `test_jaccard_penalises_column_flooding` verifies that selecting all 15 schema columns against a 2-column gold gives Jaccard ≈ 0.133, asserting `< 0.2` with an error message naming H3.1.

---

## H3.2 — WHERE Operator Gaming

### Reward Function
R3 (Partial Credit)

### Mechanism
The `where_score` component awards 0.5 * (column match) + 0.4 * (operator match). If an agent always writes `WHERE numeric_col >= 0` (trivially true for any positive numeric column), it achieves:
- Column match: depends on whether the column is correct
- Operator match: 0.0 (gold usually uses `=`, not `>=`)
- WHERE score: 0.5 * col_match + 0.4 * 0.0 = 0.5 * col_match

An agent that learns to always use `>=` for numeric columns, combined with correct table and column selection, achieves a partial credit score of approximately:
```
0.20 (tables) + 0.25 * 0.8 (partial cols) + 0.20 * 0.5 (where, col match) + 0.15 * 0 (agg) + 0.10 (grp) + 0.10 (ord)
= 0.20 + 0.20 + 0.10 + 0 + 0.10 + 0.10 = 0.70
```

A 0.70 partial credit score on every episode with wrong operator is a stable local optimum — the agent has no gradient signal to move to the correct `=` operator.

### Empirical Evidence
In pilot experiments with high `w_partial` (0.60) in the composite reward, we observe the `>=` operator appearing in >80% of episodes targeting numeric WHERE conditions. Shannon entropy of the operator distribution collapses from H=1.58 (uniform over 7 operators) to H=0.0.

### Detection
`RewardHackingDetector` Signal 2 tracks the Shannon entropy of WHERE operators used. When entropy < `entropy_threshold = 0.8` (normalised) AND a single operator dominates, a Signal 2 alert fires. As with the KL threshold, this entropy threshold started as a heuristic default; traced reruns now show local stability under +/-20% perturbations on representative tasks, but full calibration across tasks, seeds, and labeled windows remains open.

### Counter-measure
1. **Composite reward weighting**: In principle, adding execution match should make the operator-gaming optimum less attractive because a correct query still dominates numerically.
2. **Operator diversity detection**: The reward hacking detector flags this pattern early in training, allowing researchers to inspect collapse dynamics.

The final experiments weaken the stronger version of this claim. The weight ablation shows that increasing `w_exec` does not materially improve execution match, so in this environment the execution component does not reliably break the local optimum by itself. This is best read as a prototype-design limitation that directly motivates a tighter semantic action space and richer execution feedback, not as evidence that reward-signal analysis is pointless.

---

## H4.1 — High w_partial Local Optimum

### Reward Function
R4 (Composite)

### Mechanism
When `w_partial` is high relative to `w_exec`, the partial credit signal dominates. The agent converges to a policy that maximises partial credit (column/table/WHERE component scores) without achieving execution match.

**Example:** With `w_partial=0.60, w_exec=0.30`:
- Correct query: `0.60*0.95 + 0.30*1.0 = 0.87`
- Partially correct query (0.70 partial, 0.0 exec): `0.60*0.70 + 0.30*0.0 = 0.42`

At `w_partial=0.60`, the correct policy still dominates (0.87 > 0.42). But consider:
- Near-correct query (0.90 partial, 0.5 exec — overlap but not exact): `0.60*0.90 + 0.30*0.5 = 0.69`
- Correct query (1.0 partial, 1.0 exec): `0.60*1.0 + 0.30*1.0 = 0.90`

The gap narrows. At extreme `w_partial`, the agent can get "stuck" in a region where incremental improvements in partial credit prevent it from exploring the larger steps needed to achieve full execution match.

### Empirical Analysis
The released weight ablation trains three REINFORCE agents on Tasks 2 and 5 with three weight profiles:
- **Balanced**: `w_exec=0.50, w_partial=0.30`
- **High exec**: `w_exec=0.70, w_partial=0.20`
- **High partial**: `w_exec=0.30, w_partial=0.50`

The key result is the opposite of what a successful weighting fix would show: execution match remains effectively unchanged across all three settings. Increasing `w_exec` suppresses composite reward but does not improve semantic behavior. This is evidence for an execution-insensitive partial-credit optimum, not for a clean trade-off frontier.

### Counter-measure
There is no complete counter-measure for H4.1 in the current repository. The default weights `w_exec=0.50, w_partial=0.30` are best viewed as a pragmatic baseline for this toy environment, not as a validated optimum. The weight ablation demonstrates that reweighting alone does not repair the semantic blind spot. The useful research outcome is that this failure mode is now explicit and reproducible rather than hidden behind aggregate reward improvements.

---

## H4.2 — LIMIT-Based Efficiency Penalty Suppression

### Reward Function
R4 (Composite)

### Mechanism
The efficiency penalty fires when `actual_rows > 5 * expected_rows`. A naive implementation computes `actual_rows` from the query result after execution, which means a query with `LIMIT 100` returns at most 100 rows — potentially below the penalty threshold even if the true result set is millions of rows.

**Attack scenario:**
- Cross-join on 50 customers × 200 orders = 10,000 rows
- Expected rows: ~200 (correct INNER JOIN result)
- Ratio without LIMIT: 10,000 / 200 = 50 >> 5 → penalty fires
- With `LIMIT 100`: returned rows = 100
- Ratio with LIMIT: 100 / 200 = 0.5 < 5 → **no penalty**

If LIMIT is a valid action (it is, in the `LIMIT_VAL` phase), the agent can learn to always add `LIMIT 100` to any cross-join query, suppressing the efficiency penalty while still generating semantically wrong queries.

### Counter-measure
**The critical implementation in `executor.py`:**
```python
def _count_without_limit(self, conn, sql):
    """
    Wrap the query in COUNT(*) after stripping LIMIT clause.
    This gives the true result-set size before LIMIT truncation.
    Counter-measure for H4.2.
    """
    stripped = self._strip_limit(sql)
    count_sql = f"SELECT COUNT(*) FROM ({stripped}) _subq"
    ...
```

The efficiency penalty uses `pred_rows_no_limit` (from the COUNT(*) wrapper), not `pred_rows` (from the LIMIT-truncated result). This is documented in the `exec_info` dict returned by `ExecutionMatchReward`:
```python
"pred_rows": pred_result.n_rows,                    # with LIMIT (for display)
"pred_rows_no_limit": pred_result.row_count_no_limit,  # for efficiency penalty
```

The `_strip_limit()` method in `executor.py` uses a regex to remove trailing `LIMIT N [OFFSET M]` before wrapping in COUNT(*).

---

## Summary: Research Implications

The eight hacking scenarios documented here span four categories of reward gaming:

1. **Normalisation exploits** (H1.1, H1.2): Gaming string comparison via the normalisation pipeline. Specific to exact match rewards. Mitigation: execution match as the primary semantic signal.

2. **Coverage exploits** (H2.1, H3.1): Maximising result set size or selected column count to inflate set-intersection numerators. Mitigation: Jaccard (not recall/precision) for set comparisons.

3. **Fixed-point stagnation** (H2.2, H4.1): The policy converges to a degenerate local optimum where the reward signal provides weak or non-separating gradient toward the correct policy. Mitigation: negative penalty for errors; partial monitoring only.

4. **Penalty evasion** (H3.2, H4.2): The agent learns to avoid triggering specific penalties (operator monoculture, efficiency threshold) through strategies that do not involve generating correct SQL. Mitigation: detection-based monitoring (operator entropy, Spearman trend) and penalty-evasion-proof counting (pre-LIMIT COUNT).

The composite reward R4 addresses scenarios H1.1, H1.2, H2.1, H2.2, and H4.2 only partially through its design. Scenarios H3.2 and H4.1 are observable but not fully mitigated, and the final ablations suggest that some of the nominal mitigations do not move behavior in practice. This is the fundamental tension in the benchmark: dense rewards are necessary for learning, but the current dense signals are still loose enough to admit stable non-semantic optima.

This is consistent with the broader RLHF literature: dense reward signals provide better gradient information but create more surface area for hacking. In this repository, the weight ablation does not answer the reward-design question conclusively; it mainly shows that simple reweighting is insufficient.
