from .exact_match import ExactMatchReward
from .execution_match import ExecutionMatchReward
from .partial_credit import PartialCreditReward
from .composite import CompositeReward

__all__ = [
    "ExactMatchReward",
    "ExecutionMatchReward",
    "PartialCreditReward",
    "CompositeReward",
]


def get_reward_fn(name: str, config: dict, db_path: str):
    """Factory: return a reward callable by name."""
    name = name.lower()
    if name == "exact":
        return ExactMatchReward.from_config(config)
    elif name == "execution":
        return ExecutionMatchReward.from_config(config, db_path)
    elif name == "partial":
        return PartialCreditReward.from_config(config)
    elif name == "composite":
        return CompositeReward.from_config(config, db_path)
    else:
        raise ValueError(f"Unknown reward: {name}. Choose from: exact, execution, partial, composite")
