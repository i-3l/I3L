from iiil.policies.curobo_policy import CuroboPolicy
from iiil.policies.playback_policy import PlaybackPolicy
from iiil.policies.gello_policy import GelloPolicy
from iiil.policies.spacemouse_policy import SpaceMousePolicy
from iiil.policies.il_policy import ImitationLearningPolicy
from iiil.policies.waypoint_policy import RandomPositionPolicy, WaypointPolicy
from iiil.policies.hybrid_policy import NoisyPolicy, SharedAutonomyPolicy
from iiil.policies.sine_policy import SineWavePolicy

policy_lib = {
    "curobo": CuroboPolicy,
    "gello": GelloPolicy,
    "playback": PlaybackPolicy,
    "random_position": RandomPositionPolicy,
    "sa": SharedAutonomyPolicy,
    "spacemouse": SpaceMousePolicy,
    "waypoint": WaypointPolicy,
    "noise": NoisyPolicy,
    "il": ImitationLearningPolicy,
    "dummy": ImitationLearningPolicy,
    "sine": SineWavePolicy,
}