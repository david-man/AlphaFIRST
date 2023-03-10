from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.base_env.powerplay import Powerplay


ENV_REGISTRY["powerplay"] = Powerplay


env = marl.make_env(environment_name="powerplay", map_name = "powerplay")


# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='common')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'timesteps_total': 1000}, share_policy='group', local_mode = False, num_gpus = 1)


#mappo.render(env, model, share_policy='group', restore_path='path_to_checkpoint')
