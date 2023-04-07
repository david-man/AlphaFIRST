from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.base_env.powerplay_movementbased import Powerplay
from ray import tune
import ray

from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from marllib.envs.base_env.mpe import RllibMPE
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer, get_policy_class_mappo
from marllib.marl.common import get_config, get_model_config
import cloudpickle
from ray.rllib.agents import ppo
import gymnasium as gym
from marllib.marl.common import *
from marllib.marl.algos.scripts import POlICY_REGISTRY
from ray.rllib.agents.ppo import PPOTrainer

from marllib.marl.algos.utils.setup_utils import AlgVar












# mappo.render(env, model, share_policy='group', restore_path={'params_path': "exp_results/mappo_gru_powerplay/MAPPOTrainer_powerplay_movementbased_powerplay_e3478_00000_0_2023-03-10_23-33-36/params.json",  # experiment configuration
#                            'model_path': "exp_results/mappo_gru_powerplay/MAPPOTrainer_powerplay_movementbased_powerplay_e3478_00000_0_2023-03-10_23-33-36/checkpoint_000005/"},
#                            local_mode = True,
#                            checkpoint_end=False)



ENV_REGISTRY["powerplay_movementbased"] = Powerplay


env = marl.make_env(environment_name="powerplay_movementbased", map_name = "powerplay")


# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='common')


# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {'core_arch': 'gru', 'encode_layer': '128-256', 'fc_layer': 2, 'hidden_state_size': 256, 'out_dim_fc_0': 128, 'out_dim_fc_1': 64})
mappo.fit(env, model, stop={'timesteps_total': 10000}, share_policy='group', local_mode = False, num_gpus = 1)


# #
# mappo.test(env, model,
#              restore_path={'params_path': "checkpoint_000001/params.json",  # experiment configuration
#                            'model_path': "checkpoint_000001/checkpoint-1"},  # checkpoint path
#              local_mode=True,
#              share_policy="group",
#              checkpoint_end=False)
