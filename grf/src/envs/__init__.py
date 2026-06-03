from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
from .grf import Academy_3_vs_1_with_Keeper, Academy_Pass_and_Shoot_with_Keeper, Academy_Run_Pass_and_Shoot_with_Keeper, Academy_Corner,Academy_Counterattack_Hard
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "academy_3_vs_1_with_keeper": partial(env_fn, env=Academy_3_vs_1_with_Keeper),
    "academy_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper),
    "academy_run_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Run_Pass_and_Shoot_with_Keeper),
    "academy_corner": partial(env_fn, env=Academy_Corner),
    "academy_counterattack_hard": partial(env_fn, env=Academy_Counterattack_Hard),
}

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

