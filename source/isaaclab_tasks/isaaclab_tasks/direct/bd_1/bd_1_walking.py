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


    def _get_rewards(self) -> torch.Tensor:
        # 1. Linear Velocity Tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        # 2. Angular Velocity Tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # 3. Flat Orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # 4. Joint Acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        # 5. Feet Air Time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1 # Only reward if moving
        )

        # 6. Sinusoid Cadence (The Gait Clock)
        # We reward contact that matches a reference sine wave (alternating feet).
        # Assuming self.gait_phase oscillates between 0 and 2*pi
        phase = self.episode_length_buf * self.step_dt * self.cfg.gait_frequency * 2 * torch.pi
        left_foot_target = torch.sin(phase) > 0
        right_foot_target = torch.sin(phase) < 0
        
        current_contacts = self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_ids, 2] > 1.0
        # Reward for matching the target (Left foot down when target > 0, etc.)
        gait_reward = (current_contacts[:, 0] == left_foot_target).float() + \
                    (current_contacts[:, 1] == right_foot_target).float()

        # 7. Undesired Contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # Check if any body part in the "undesired" list has a contact force > 1.0N
        is_undesired_contact = torch.any(
            torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1) > 1.0, 
            dim=1
        )

        rewards = {
            "track_lin_vel_xy": lin_vel_error_mapped * self.cfg.reward_scales.lin_vel,
            "track_ang_vel_z": yaw_rate_error_mapped * self.cfg.reward_scales.ang_vel,
            "flat_orientation": -flat_orientation * self.cfg.reward_scales.orientation,
            "joint_accel": -joint_accel * self.cfg.reward_scales.joint_accel,
            "feet_air_time": air_time * self.cfg.reward_scales.air_time,
            "gait_cadence": gait_reward * self.cfg.reward_scales.gait,
            "undesired_contacts": -is_undesired_contact.float() * self.cfg.reward_scales.undesired_contact,
        }

        # Sum and Normalize by dt
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
                # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return total_reward