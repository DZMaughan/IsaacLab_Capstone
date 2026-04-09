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
        usd_path="/home/zacmaughan/IsaacLab_Capstone/source/isaaclab_assets/isaaclab_assets/custom/test_flat.usda",
        scale=(0.1, 0.1, 0.1),
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, .3),
        #                                        joint_pos={
        #     ".*Hip": 0.0,
        #     ".*Knee": 0.0,
        #     ".*Ankle": 0.0,
        # },
    ),
    actuators={
        "LeftHip": ImplicitActuatorCfg(
            joint_names_expr=["LeftHip"],
            effort_limit_sim=100.0,
            velocity_limit=5.0,
            stiffness=25.0,
            damping=0.5,
        ),
        "RightHip": ImplicitActuatorCfg(
            joint_names_expr=["RightHip"],
            effort_limit_sim=100.0,
            velocity_limit=5.0,
            stiffness=25.0,
            damping=0.5,
        ),
        "LeftKnee": ImplicitActuatorCfg(
            joint_names_expr=["LeftKnee"],
            effort_limit_sim=100.0,
            velocity_limit=5.0,
            stiffness=25.0,
            damping=0.5,
        ),
        "RightKnee": ImplicitActuatorCfg(
            joint_names_expr=["RightKnee"],
            effort_limit_sim=100.0,
            velocity_limit=5.0,
            stiffness=25.0,
            damping=0.5,
        ),
        "LeftAnkle": ImplicitActuatorCfg(
            joint_names_expr=["LeftAnkle"],
            effort_limit_sim=100.0,
            velocity_limit=5.0,
            stiffness=25.0,
            damping=0.5,
        ),
        "RightAnkle": ImplicitActuatorCfg(
            joint_names_expr=["RightAnkle"],
            effort_limit_sim=100.0,
            velocity_limit=5.0,
            stiffness=25.0,
            damping=0.5,
        ),
    },
)