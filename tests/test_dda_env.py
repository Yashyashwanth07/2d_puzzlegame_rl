"""Unit tests for the DDA Gymnasium environment."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import gymnasium as gym

from envs.dda_env import DDAEnv

try:
    from gymnasium.utils.env_checker import check_env
except ImportError:
    check_env = None


# ---- Fixtures ----

@pytest.fixture
def env():
    """Create a DDAEnv with short sessions for fast tests."""
    return DDAEnv(config={'session_length': 3})


@pytest.fixture
def default_env():
    """Create a default DDAEnv."""
    return DDAEnv()


# ---- Creation tests ----

def test_env_creation(default_env):
    """DDAEnv should create without error."""
    assert default_env is not None


def test_observation_space(default_env):
    """Observation from reset should be within observation_space."""
    obs, info = default_env.reset(seed=42)
    assert default_env.observation_space.contains(obs), \
        f"Obs not in observation_space: {obs}"


def test_action_space(default_env):
    """Action space should be Discrete(9)."""
    assert isinstance(default_env.action_space, gym.spaces.Discrete)
    assert default_env.action_space.n == 9


# ---- Reset tests ----

def test_reset_returns_valid(default_env):
    """reset() should return (obs, info) with correct types."""
    obs, info = default_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


# ---- Step tests ----

def test_step_returns_valid(env):
    """step() should return the standard Gymnasium 5-tuple."""
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_terminates(env):
    """Episode should terminate after session_length levels."""
    env.reset(seed=42)
    steps = 0
    for _ in range(20):  # Safety limit
        obs, reward, terminated, truncated, info = env.step(0)
        steps += 1
        if terminated:
            break
    assert terminated, "Episode should have terminated"
    assert steps == 3, f"Expected 3 steps (session_length=3), got {steps}"


def test_reward_bounds(env):
    """Reward should be within reasonable bounds."""
    env.reset(seed=42)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        assert -2.0 <= reward <= 2.0, f"Reward {reward} out of bounds [-2, 2]"
        if terminated:
            break


# ---- Difficulty bounds tests ----

def test_difficulty_stays_in_bounds():
    """After many random actions, difficulty params should stay valid."""
    env = DDAEnv(config={'session_length': 5})
    env.reset(seed=42)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # Check bounds
        assert env.current_grid_size >= env.grid_size_min
        assert env.current_grid_size <= env.grid_size_max
        assert 0 <= env.current_num_keys <= env.max_keys
        assert 0 <= env.current_num_enemies <= env.max_enemies
        assert 0 <= env.current_num_hints <= env.max_hints
        if terminated:
            env.reset(seed=42)


# ---- Observation normalization ----

def test_observation_normalized(env):
    """All observation values should be in [0, 1]."""
    obs, info = env.reset(seed=42)
    assert np.all(obs >= 0.0), f"Obs has values < 0: {obs}"
    assert np.all(obs <= 1.0), f"Obs has values > 1: {obs}"

    # Also check after steps
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.all(obs >= 0.0), f"Obs has values < 0 after step: {obs}"
        assert np.all(obs <= 1.0), f"Obs has values > 1 after step: {obs}"
        if terminated:
            break


# ---- Gymnasium env checker ----

@pytest.mark.skipif(check_env is None, reason="gymnasium env_checker not available")
def test_gymnasium_check_env():
    """Passes gymnasium's built-in environment checker."""
    env = DDAEnv()
    check_env(env)


# ---- Multi-episode test ----

def test_multiple_episodes(env):
    """Can reset and run multiple episodes without error."""
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        assert obs is not None
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())


# ---- Info dict test ----

def test_info_dict_keys(env):
    """Info dict should contain expected keys after step."""
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    expected_keys = {'solve_time', 'solved', 'deaths', 'reward',
                     'grid_size', 'num_keys', 'num_enemies', 'num_hints',
                     'bot_noise_prob'}
    for key in expected_keys:
        assert key in info, f"Missing key '{key}' in info dict"
