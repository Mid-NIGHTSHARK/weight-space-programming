"""Independent, replayable RNG stream derivation for corrected evaluations.

The episode's base seed is expanded with NumPy SeedSequence.spawn into two
child streams.  MinAtar receives the environment child's 32-bit seed and the
categorical policy sampler receives the policy child's 32-bit seed.  The
environment and policy then own separate RandomState instances.  Their states
are therefore never shared or advanced by the other subsystem.
"""

from dataclasses import dataclass
import numpy as np


RNG_PROTOCOL = "numpy-seedsequence-spawn-two-randomstate-v1"


@dataclass(frozen=True)
class EpisodeStreamSeeds:
    base_seed: int
    environment_seed: int
    policy_seed: int


def derive_episode_stream_seeds(base_seed: int) -> EpisodeStreamSeeds:
    """Derive distinct reproducible environment and policy seeds.

    SeedSequence's child spawn keys provide domain-separated streams from the
    same recorded base seed.  uint32 values are used because MinAtar 1.0.15
    seeds its legacy RandomState with an integer in that range.
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("base_seed must be an integer")
    if int(base_seed) < 0:
        raise ValueError("base_seed must be non-negative")

    env_sequence, policy_sequence = np.random.SeedSequence(int(base_seed)).spawn(2)
    environment_seed = int(env_sequence.generate_state(1, dtype=np.uint32)[0])
    policy_seed = int(policy_sequence.generate_state(1, dtype=np.uint32)[0])
    return EpisodeStreamSeeds(
        base_seed=int(base_seed),
        environment_seed=environment_seed,
        policy_seed=policy_seed,
    )


def make_policy_rng(policy_seed: int) -> np.random.RandomState:
    """Create the legacy categorical sampler RNG on its own stream."""
    return np.random.RandomState(int(policy_seed))
