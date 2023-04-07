import ray
from ray import tune
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.marl.common import recursive_dict_update, merge_default_and_customized
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin, ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing
from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.envs.base_env.powerplay_movementbased import Powerplay
import json
import numpy as np
from ray.rllib.policy.rnn_sequencing import add_time_dimension
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def run_test(algo_config, env, model, stop=None):
    ray.init(local_mode=algo_config["local_mode"])

    ########################
    ### environment info ###
    ########################

    env_config = env.get_env_info()
    
    map_name = algo_config['env_args']['map_name']
    agent_name_ls = env.agents
    env_config["agent_name_ls"] = agent_name_ls
    env.close()

    ######################
    ### policy sharing ###
    ######################
    
    policy_mapping_info = env_config["policy_mapping_info"]
    
    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if algo_config["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

        policies = {"av"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "av")

    elif algo_config["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

            policies = {"shared_policy"}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: "shared_policy")

        else:
            policies = {
                "policy_{}".format(i): (None, env_config["space_obs"], env_config["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif algo_config["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_config["space_obs"], env_config["space_act"], {}) for i in
            range(env_config["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(algo_config["share_policy"]))

    # if happo or hatrpo, force individual
    if algo_config["algorithm"] in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_config["space_obs"], env_config["space_act"], {}) for i in
            range(env_config["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    #########################
    ### experiment config ###
    #########################

    common_config = {
        "seed": int(algo_config["seed"]),
        "env": algo_config["env"] + "_" + algo_config["env_args"]["map_name"],
        "num_gpus_per_worker": algo_config["num_gpus_per_worker"],
        "num_gpus": algo_config["num_gpus"],
        "num_workers": algo_config["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": algo_config["framework"],
        "evaluation_interval": algo_config["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop_config = {
        "episode_reward_mean": algo_config["stop_reward"],
        "timesteps_total": algo_config["stop_timesteps"],
        "training_iteration": algo_config["stop_iters"],
    }

    stop_config = merge_default_and_customized(stop_config, stop)

    if algo_config['restore_path']['model_path'] == '':
        restore = None
    else:
        restore = algo_config['restore_path']
        render_config = {
            "evaluation_interval": 1,
            "evaluation_num_episodes": 100,
            "evaluation_num_workers": 1,
            "evaluation_config": {
                "record_env": False,
                "render_env": True,
            }
        }

        common_config = recursive_dict_update(common_config, render_config)

        render_stop_config = {
            "training_iteration": 1,
        }

        stop_config = recursive_dict_update(stop_config, render_stop_config)

    ##################
    ### run script ###
    ##################

    model_class = model
    config_dict = algo_config
    common_config = common_config
    env_dict = env_config
    stop = stop_config
    restore = restore
    ModelCatalog.register_custom_model(
        "Centralized_Critic_Model", model_class)

    _param = AlgVar(config_dict)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = _param["batch_episode"] * env_dict["episode_limit"]
    if "fixed_batch_timesteps" in config_dict:
        train_batch_size = config_dict["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    batch_mode = _param["batch_mode"]
    lr = _param["lr"]
    clip_param = _param["clip_param"]
    vf_clip_param = _param["vf_clip_param"]
    use_gae = _param["use_gae"]
    gae_lambda = _param["lambda"]
    kl_coeff = _param["kl_coeff"]
    num_sgd_iter = _param["num_sgd_iter"]
    vf_loss_coeff = _param["vf_loss_coeff"]
    entropy_coeff = _param["entropy_coeff"]

    config = {
        "batch_mode": batch_mode,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": lr if restore is None else 1e-10,
        "entropy_coeff": entropy_coeff,
        "num_sgd_iter": num_sgd_iter,
        "clip_param": clip_param,
        "use_gae": use_gae,
        "lambda": gae_lambda,
        "vf_loss_coeff": vf_loss_coeff,
        "kl_coeff": kl_coeff,
        "vf_clip_param": vf_clip_param,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }
    config.update(common_config)

    algorithm = config_dict["algorithm"]
    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    if restore is not None:
        with open(restore["params_path"], 'r') as JSON:
            raw_config = json.load(JSON)
            raw_config = raw_config["model"]["custom_model_config"]['model_arch_args']
            #print(raw_config)
            check_config = config["model"]["custom_model_config"]['model_arch_args']
            if check_config != raw_config:
                raise ValueError("is not using the params required by the checkpoint model")
        model_path = restore["model_path"]
    else:
        model_path = None
    #print(config)
    x = MAPPOTrainer.with_updates(default_policy="policy_red_")(config = config, env = Powerplay)
    
    x.restore(checkpoint_path="checkpoint_000001/checkpoint-1")
    
    p = Powerplay({})
    obs_ = p.reset()
    #print(obs_)
    #print(inputs)
    policy = x.get_policy("policy_red_")
    #print(np.array([obs_["red_1"]["action_mask"], obs_["red_2"]["action_mask"]]).shape)
    
    obs_red = {"obs": np.array([obs_["red_1"]["obs"], obs_["red_2"]["obs"]]), "action_mask": np.array([obs_["red_1"]["action_mask"], obs_["red_2"]["action_mask"]])}
    
    
    # print(x.get_policy("policy_red_"))
    # print(x.config)
    #print("STATE: " + str(policy.get_exploration_state()))
    actions = policy.compute_actions(obs_red, policy_id = "policy_red_", state_batches = [torch.tensor([policy.get_initial_state()[0],policy.get_initial_state()[0]])])
    
    #print(actions)
    #trainer = PPOTrainer(config = config, env = Powerplay)
    
    ray.shutdown()
