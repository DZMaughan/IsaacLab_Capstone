from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils.math import quat_from_euler_xyz
import matplotlib.pyplot as plt

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

@configclass
class BD1WalkingEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 20.0


    sim: SimulationCfg = SimulationCfg(dt=1/200, render_interval=decimation)

    wall_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object0",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 5.0, 1), rot=(1.0, 0.0, 0.0, 0.0) 
        ),
    )

    wall_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object1",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -5.0, 1), rot=(1.0, 0.0, 0.0, 0.0) 
        ),
    )

    wall_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0) 
        ),
    )
    
    wall_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0) 
        ),
    )

    block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_.",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0) 
        ),
    )

    env_spacing = 30.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

    goal_reward_scale = 20


class BD1WalkingEnv(DirectMARLEnv):
    cfg: BD1WalkingEnvCfg

    def __init__(self, cfg: BD1WalkingEnvCfg, render_mode: str | None = None, headless: bool | None = None, debug: bool = False, **kwargs):
        self.debug = debug
        super().__init__(cfg, render_mode, **kwargs)

        self.headless = headless

        self.env_spacing = self.cfg.env_spacing

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "robot_vel_reward",
            ]
        }

        def _setup_scene(self):
            self.wall_0 = RigidObject(self.cfg.wall_0)
            self.wall_1 = RigidObject(self.cfg.wall_1)
            self.wall_2 = RigidObject(self.cfg.wall_2)
            self.wall_3 = RigidObject(self.cfg.wall_3)

            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(
                    size=(500.0, 500.0),
                    color=(0.2, 0.2, 0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )
