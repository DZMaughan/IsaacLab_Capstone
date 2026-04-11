"""Configuration for a BD-1 robot."""
from pathlib import Path

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

isaaclab_asset_path = Path(
    Path(isaaclab.__path__[0]).parent.parent, "isaaclab_assets", "isaaclab_assets", "custom"
)

BD_1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=str(Path(isaaclab_asset_path, "bd_1.usd")),
        usd_path="/home/zacmaughan/IsaacLab_Capstone/source/isaaclab_assets/isaaclab_assets/custom/simple_body.usda",
        # scale=(0.1, 0.1, 0.1),
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0),
        #                                        joint_pos={
        #     ".*Hip": 0.0,
        #     ".*Knee": 0.0,
        #     ".*Ankle": 0.0,
        # },
    ),
    actuators={
        "LeftHip": ImplicitActuatorCfg(
            joint_names_expr=["LHip"],
            effort_limit_sim=20.0,
            velocity_limit=5.0,
            velocity_limit_sim=5.0,
            stiffness=0.5,
            damping=0.1,
        ),
        "RightHip": ImplicitActuatorCfg(
            joint_names_expr=["RHip"],
            effort_limit_sim=20.0,
            velocity_limit=5.0,
            velocity_limit_sim=5.0,
            stiffness=0.5,
            damping=0.1,
        ),
        "LeftKnee": ImplicitActuatorCfg(
            joint_names_expr=["LKnee"],
            effort_limit_sim=20.0,
            velocity_limit=5.0,
            velocity_limit_sim=5.0,
            stiffness=0.5,
            damping=0.1,
        ),
        "RightKnee": ImplicitActuatorCfg(
            joint_names_expr=["RKnee"],
            effort_limit_sim=20.0,
            velocity_limit=5.0,
            velocity_limit_sim=5.0,
            stiffness=0.5,
            damping=0.1,
        ),
        "LeftAnkle": ImplicitActuatorCfg(
            joint_names_expr=["LAnkle"],
            effort_limit_sim=20.0,
            velocity_limit=5.0,
            velocity_limit_sim=5.0,
            stiffness=0.5,
            damping=0.1,
        ),
        "RightAnkle": ImplicitActuatorCfg(
            joint_names_expr=["RAnkle"],
            effort_limit_sim=20.0,
            velocity_limit=5.0,
            velocity_limit_sim=5.0,
            stiffness=0.5,
            damping=0.1,
        ),
    },
)