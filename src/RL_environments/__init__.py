import gymnasium as gym
from RL_environments import agents


gym.register(
    id="Muscle-Reach-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reaching_env_muscle_go2:ReachingMuscleGo2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2MusclePPORunnerCfg",
    }
)

gym.register(
    id="Muscle-Stand-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tilting_env_muscle_go2:StandingMuscleGo2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2MusclePPORunnerCfg",
    }
)

gym.register(
    id="Muscle-Walk-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_muscle_go2:WalkingMuscleGo2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingUnitreeGo2MusclePPORunnerCfg",
    }
)

gym.register(
    id="Muscle-Hop-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hopping_env_muscle:HoppingMuscleGo2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HoppingUnitreeGo2MusclePPORunnerCfg",
    }
)