"""
example on how to wrap env to customize the group policy sharing
"""
from marllib.envs.base_env.smac import *
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

policy_mapping_dict = {
    "all_scenario": {
        "description": "SMAC all scenarios manually into two teams",
        "team_prefix": ("TeamA_", "TeamB_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
        # be careful using one_agent_one_policy when training in maps like 27m_vs_30m, which has relatively large
        # number of agents
    },
}

class Two_Teams_SMAC(MultiAgentEnv):

    def __init__(self, map_name):
        self.env = RLlibSMAC(map_name)
        self.num_agents = self.env.num_agents
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.agents_origin = self.env.agents
        agent_ls = []
        # simple case: separate agent into two groups
        for i in range(self.env.num_agents):
            if i > self.num_agents // 2:
                agent_ls.append("TeamA_{}".format(i))
            else:
                agent_ls.append("TeamB_{}".format(i))
        self.agents = agent_ls

    def reset(self):
        obs_dict_origin = self.env.reset()
        obs_dict = {}
        # swap name
        for agent_origin_name, agent_name in zip(self.agents_origin, self.agents):
            obs_dict[agent_name] = obs_dict_origin[agent_origin_name]

        return obs_dict

    def step(self, actions):

        obs_dict_origin, reward_dict_origin, dones, info = self.env.step(actions)
        obs_dict = {}
        reward_dict = {}
        # swap name
        for agent_origin_name, agent_name in zip(self.agents_origin, self.agents):
            obs_dict[agent_name] = obs_dict_origin[agent_origin_name]
            reward_dict[agent_name] = reward_dict_origin[agent_origin_name]

        return obs_dict, reward_dict, dones, {}

    def get_env_info(self):
        return self.env.get_env_info()


if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["two_teams_smac"] = Two_Teams_SMAC
    # initialize env
    env = marl.make_env(environment_name="two_teams_smac", map_name="3m")
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
    # start learning
    mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
              num_workers=2, share_policy='group', checkpoint_freq=50)
