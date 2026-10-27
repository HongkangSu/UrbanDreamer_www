"""Pure primitive policy-temperature metadata; no tensors, randomness or LR changes."""
import math
from numbers import Integral, Real

NONE = 'none'
PROFILE = 'warm5_loglinear_to_ep10_t0p01_v1'
PROFILE_MARKER = 'training_temperature_contract'
_ATTRIBUTES = ('training_temperature_profile', 'training_temperature_episode', 'training_temperature')


def _completed(value):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or value < 0 or int(value) != value):
        raise ValueError('temperature completed episodes must be a nonnegative integer')
    return int(value)


def profile_state(profile, completed):
    """Policy metadata for the NEXT episode after completed; none is always absent/T1."""
    completed = _completed(completed)
    if profile == NONE:
        return dict(profile=NONE, temperature=1.)
    if profile != PROFILE:
        raise ValueError('unsupported training temperature profile')
    episode = completed + 1
    temperature = 1. if episode <= 5 else .01 if episode >= 10 else math.exp(math.log(.01) * (episode - 5) / 5)
    return dict(profile=profile, episode=episode, temperature=temperature)


def validate_state(state):
    if not isinstance(state, dict):
        raise ValueError('training temperature metadata must be a primitive dict')
    profile = state.get('profile')
    if profile == NONE:
        expected = profile_state(NONE, 0)
    else:
        episode = state.get('episode')
        if isinstance(episode, bool) or not isinstance(episode, Integral) or episode < 1:
            raise ValueError('training temperature metadata needs a positive episode')
        expected = profile_state(profile, int(episode) - 1)
    temperature = state.get('temperature')
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0 or state != expected):
        raise ValueError('training temperature profile/episode/temperature disagree')
    return expected


def read_actor_temperature(actor):
    profile = getattr(actor, 'training_temperature_profile', NONE)
    if profile == NONE:
        state = dict(profile=NONE, temperature=getattr(actor, 'training_temperature', 1.))
    else:
        state = dict(profile=profile, episode=getattr(actor, 'training_temperature_episode', None),
            temperature=getattr(actor, 'training_temperature', None))
    return validate_state(state)


def apply_actor_temperature(actor, state):
    state = validate_state(state)
    if state['profile'] == NONE:
        for name in _ATTRIBUTES:
            if hasattr(actor, name):
                delattr(actor, name)
        return
    actor.training_temperature_profile = state['profile']
    actor.training_temperature_episode = state['episode']
    actor.training_temperature = state['temperature']


def copy_actor_temperature(source, target):
    apply_actor_temperature(target, read_actor_temperature(source))


def reset_actor_temperature(actor):
    apply_actor_temperature(actor, profile_state(NONE, 0))


def require_actor_temperature(actor, expected):
    if read_actor_temperature(actor) != validate_state(expected):
        raise ValueError('actor temperature changed within the frozen episode policy')


def temperature_checkpoint_state(profile, completed):
    completed = _completed(completed)
    if profile == NONE:
        return None
    return dict(version=1, profile=profile, completed_episodes=completed,
        actor=profile_state(profile, max(completed - 1, 0)), next_episode=profile_state(profile, completed))


def validate_temperature_checkpoint(profile, metrics, completed):
    expected = temperature_checkpoint_state(profile, completed)
    if expected is None:
        if PROFILE_MARKER in metrics:
            raise ValueError('temperature-profile checkpoint cannot load into default none')
        return None
    if metrics.get(PROFILE_MARKER) != expected:
        raise ValueError('training temperature checkpoint contract mismatch; fresh explicit migration required')
    return dict(expected['actor'])
