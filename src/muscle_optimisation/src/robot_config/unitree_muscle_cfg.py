import isaaclab.sim as sim_utils
from muscle_actuator.muscle_actuator_cfg import MuscleActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import torch
muscle_params = {
    "lmin": 0.24,
    "lmax": 1.53,
    "fvman": 1.38,
    "fpmax": 1.76,
    "lce_min": 0.74,
    "lce_max": 0.94,
    "peak_force": 45,
    "dt":1/500,
    "angles": torch.Tensor([[-1.0472, 1.0472], [-1.0472, 1.0472], [-1.0472, 1.0472], [-1.0472, 1.0472], [-1.5708, 3.4907], [-1.5708, 3.4907], [-0.5236, 4.5379], [-0.5236, 4.5379], [-2.7227, -0.8378], [-2.7227, -0.8378], [-2.7227, -0.8378], [-2.7227, -0.8378]]),
    "device": "cuda:0"
}


UNITREE_GO2_MUSCLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": MuscleActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            #effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            muscle_params=muscle_params
        )
    },
)
"""Configuration of Unitree Go2 using DC-Motor actuator model."""

