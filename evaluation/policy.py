"""Versioned evaluation policies.

A policy fixes everything about the environment except the agent: turn and request limits,
pricing, and which shared caches supply evidence and prices. Compare agents only within one
policy. Results from different policies are not comparable. Earlier development policies (v1 to
v3) are retired.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Policy:
    name: str
    max_turns: int
    max_questions_per_turn: int | None
    max_tests_per_turn: int
    visit_cost: int = 300
    # Shared cache folders. Reusing them across policies holds evidence and prices fixed. The
    # names record the policy that created them.
    gatekeeper_cache: str = "gatekeeper-cache-v1"
    price_review_cache: str = "reviewed-cpt-bedside-v3"

    def settings(self):
        """Fields written to each run's settings.json beyond the policy name."""
        fields = asdict(self)
        fields.pop("name")
        return {"policy": fields}


POLICIES = {
    # 10 turns and no question limit. Evidence and price caches are shared with v3.
    "reviewed-cpt-bedside-v4": Policy(
        name="reviewed-cpt-bedside-v4", max_turns=10, max_questions_per_turn=None,
        max_tests_per_turn=4,
    ),
}
DEFAULT_POLICY = "reviewed-cpt-bedside-v4"


def get_policy(name=DEFAULT_POLICY):
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"Unknown policy {name!r}. Choose one of {sorted(POLICIES)}") from None
