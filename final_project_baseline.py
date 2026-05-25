import argparse
from dataclasses import dataclass, field

import numpy as np

import genesis as gs


@dataclass
class BaselineConfig:
    dt: float = 0.01
    settle_steps: int = 30
    hold_steps: int = 60
    num_path_waypoints: int = 120

    table_size: tuple[float, float, float] = (0.9, 1.2, 0.05)
    table_pos: tuple[float, float, float] = (0.45, 0.0, 0.025)

    cube_size: float = 0.04
    cube_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.55, -0.22, 0.07),
            (0.62, 0.00, 0.07),
            (0.55, 0.22, 0.07),
        ]
    )
    goal_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.20, -0.22, 0.07),
            (0.20, 0.00, 0.07),
            (0.20, 0.22, 0.07),
        ]
    )

    obstacle_size: tuple[float, float, float] = (0.08, 0.28, 0.12)
    obstacle_pos: tuple[float, float, float] = (0.38, 0.0, 0.11)

    grasp_quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    gripper_open: float = 0.04
    gripper_closed: float = 0.0

    pre_grasp_height: float = 0.18
    grasp_height_offset: float = 0.015
    lift_height: float = 0.24
    place_height_offset: float = 0.02
    safe_transfer_height: float = 0.32

    home_qpos: tuple[float, ...] = (
        -1.0124,
        1.5559,
        1.3662,
        -1.6878,
        -1.5799,
        1.7757,
        1.4602,
        0.04,
        0.04,
    )
    success_threshold: float = 0.05


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy().astype(float, copy=False)
    return np.asarray(x, dtype=float)


def build_scene(cfg: BaselineConfig, show_viewer: bool, backend) -> tuple[gs.Scene, dict]:
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.4, -1.2, 1.5),
            camera_lookat=(0.35, 0.0, 0.25),
            camera_fov=35,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=cfg.dt),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane(), name="ground")

    table = scene.add_entity(
        gs.morphs.Box(
            pos=cfg.table_pos,
            size=cfg.table_size,
            fixed=True,
        ),
        surface=gs.surfaces.Plastic(color=(0.75, 0.72, 0.68)),
        name="table",
    )

    obstacle = scene.add_entity(
        gs.morphs.Box(
            pos=cfg.obstacle_pos,
            size=cfg.obstacle_size,
            fixed=True,
        ),
        surface=gs.surfaces.Plastic(color=(0.25, 0.45, 0.90)),
        name="obstacle",
    )

    cubes = []
    goal_markers = []
    for idx, pos in enumerate(cfg.cube_positions):
        cube = scene.add_entity(
            gs.morphs.Box(
                pos=pos,
                size=(cfg.cube_size, cfg.cube_size, cfg.cube_size),
            ),
            material=gs.materials.Rigid(rho=250),
            surface=gs.surfaces.Plastic(color=(0.85, 0.25 + 0.2 * idx, 0.25)),
            name=f"cube_{idx}",
        )
        cubes.append(cube)

    goal_size = (cfg.cube_size * 1.5, cfg.cube_size * 1.5, 0.005)
    for idx, pos in enumerate(cfg.goal_positions):
        marker = scene.add_entity(
            gs.morphs.Box(
                pos=(pos[0], pos[1], cfg.table_pos[2] + cfg.table_size[2] / 2 + goal_size[2] / 2),
                size=goal_size,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2)),
            name=f"goal_marker_{idx}",
        )
        goal_markers.append(marker)

    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        name="franka",
    )

    scene.build()
    setup_franka(robot)
    robot.set_qpos(np.array(cfg.home_qpos))
    for _ in range(cfg.settle_steps):
        scene.step()

    entities = {
        "table": table,
        "obstacle": obstacle,
        "cubes": cubes,
        "goal_markers": goal_markers,
        "robot": robot,
    }
    return scene, entities


def setup_franka(robot):
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    robot.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )


def get_dof_slices(robot):
    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    return motors_dof, fingers_dof


def step_and_hold(scene: gs.Scene, robot, target_qpos: np.ndarray, steps: int):
    for _ in range(steps):
        robot.control_dofs_position(target_qpos)
        scene.step()


def solve_ik(robot, ee_link, position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    qpos = robot.inverse_kinematics(
        link=ee_link,
        pos=np.asarray(position, dtype=float),
        quat=np.asarray(quat, dtype=float),
    )
    return to_numpy(qpos)


def move_to_qpos(scene: gs.Scene, robot, qpos_goal: np.ndarray, cfg: BaselineConfig, use_plan_path: bool):
    if use_plan_path:
        path = robot.plan_path(qpos_goal=qpos_goal, num_waypoints=cfg.num_path_waypoints)
        for waypoint in path:
            robot.control_dofs_position(waypoint)
            scene.step()
    else:
        step_and_hold(scene, robot, qpos_goal, cfg.hold_steps)

    step_and_hold(scene, robot, qpos_goal, cfg.settle_steps)


def move_to_pose(
    scene: gs.Scene,
    robot,
    ee_link,
    position: np.ndarray,
    quat: np.ndarray,
    gripper_open: bool,
    cfg: BaselineConfig,
    use_plan_path: bool = True,
) -> np.ndarray:
    qpos_goal = solve_ik(robot, ee_link, position, quat)
    finger_value = cfg.gripper_open if gripper_open else cfg.gripper_closed
    qpos_goal[-2:] = finger_value
    move_to_qpos(scene, robot, qpos_goal, cfg, use_plan_path=use_plan_path)
    return qpos_goal


def close_gripper(scene: gs.Scene, robot, arm_qpos: np.ndarray, cfg: BaselineConfig):
    qpos = np.asarray(arm_qpos, dtype=float).copy()
    qpos[-2:] = cfg.gripper_closed
    step_and_hold(scene, robot, qpos, cfg.hold_steps)
    return qpos


def open_gripper(scene: gs.Scene, robot, arm_qpos: np.ndarray, cfg: BaselineConfig):
    qpos = np.asarray(arm_qpos, dtype=float).copy()
    qpos[-2:] = cfg.gripper_open
    step_and_hold(scene, robot, qpos, cfg.hold_steps)
    return qpos


def compute_top_grasp_pose(cube_pos: np.ndarray, cfg: BaselineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(cube_pos[:2], dtype=float)

    pre_grasp = np.array([xy[0], xy[1], cfg.pre_grasp_height], dtype=float)
    grasp = np.array([xy[0], xy[1], cube_pos[2] + cfg.grasp_height_offset], dtype=float)
    lift = np.array([xy[0], xy[1], cfg.lift_height], dtype=float)
    return pre_grasp, grasp, lift


def compute_place_poses(goal_pos: np.ndarray, cfg: BaselineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(goal_pos[:2], dtype=float)

    transfer = np.array([xy[0], xy[1], cfg.safe_transfer_height], dtype=float)
    place = np.array([xy[0], xy[1], goal_pos[2] + cfg.place_height_offset], dtype=float)
    retreat = np.array([xy[0], xy[1], cfg.lift_height], dtype=float)
    return transfer, place, retreat


def compute_transfer_waypoints(pick_xy: np.ndarray, place_xy: np.ndarray, cfg: BaselineConfig) -> list[np.ndarray]:
    # TODO(student): Replace this hard-coded waypoint policy with your own obstacle-aware planner.
    via_mid = np.array([cfg.obstacle_pos[0], 0.0, cfg.safe_transfer_height], dtype=float)
    start_safe = np.array([pick_xy[0], pick_xy[1], cfg.safe_transfer_height], dtype=float)
    end_safe = np.array([place_xy[0], place_xy[1], cfg.safe_transfer_height], dtype=float)
    return [start_safe, via_mid, end_safe]


def pick_object(scene: gs.Scene, robot, ee_link, cube, cfg: BaselineConfig, use_plan_path: bool) -> np.ndarray:
    cube_pos = to_numpy(cube.get_pos())
    pre_grasp, grasp, lift = compute_top_grasp_pose(cube_pos, cfg)
    quat = np.asarray(cfg.grasp_quat, dtype=float)

    move_to_pose(scene, robot, ee_link, pre_grasp, quat, gripper_open=True, cfg=cfg, use_plan_path=use_plan_path)
    grasp_qpos = move_to_pose(scene, robot, ee_link, grasp, quat, gripper_open=True, cfg=cfg, use_plan_path=False)
    grasp_qpos = close_gripper(scene, robot, grasp_qpos, cfg)
    lift_qpos = move_to_pose(scene, robot, ee_link, lift, quat, gripper_open=False, cfg=cfg, use_plan_path=False)
    return lift_qpos


def place_object(
    scene: gs.Scene,
    robot,
    ee_link,
    current_qpos: np.ndarray,
    source_pos: np.ndarray,
    goal_pos: np.ndarray,
    cfg: BaselineConfig,
    use_plan_path: bool,
) -> np.ndarray:
    del current_qpos

    quat = np.asarray(cfg.grasp_quat, dtype=float)
    waypoints = compute_transfer_waypoints(np.asarray(source_pos[:2]), np.asarray(goal_pos[:2]), cfg)
    for waypoint in waypoints:
        move_to_pose(scene, robot, ee_link, waypoint, quat, gripper_open=False, cfg=cfg, use_plan_path=use_plan_path)

    transfer, place, retreat = compute_place_poses(goal_pos, cfg)
    move_to_pose(scene, robot, ee_link, transfer, quat, gripper_open=False, cfg=cfg, use_plan_path=False)
    place_qpos = move_to_pose(scene, robot, ee_link, place, quat, gripper_open=False, cfg=cfg, use_plan_path=False)
    place_qpos = open_gripper(scene, robot, place_qpos, cfg)
    retreat_qpos = move_to_pose(scene, robot, ee_link, retreat, quat, gripper_open=True, cfg=cfg, use_plan_path=False)
    return retreat_qpos


def choose_object_order(cubes, goals) -> list[int]:
    del goals
    # TODO(student): Replace this fixed ordering with your own object-selection strategy.
    return list(range(len(cubes)))


def run_rearrangement_task(scene: gs.Scene, robot, cubes, goals: list[np.ndarray], cfg: BaselineConfig, use_plan_path: bool):
    ee_link = robot.get_link("hand")
    order = choose_object_order(cubes, goals)

    for idx in order:
        cube = cubes[idx]
        goal = np.asarray(goals[idx], dtype=float)
        cube_pos_before = to_numpy(cube.get_pos())

        print(f"[Task] Move cube {idx} from {np.round(cube_pos_before, 4)} to {np.round(goal, 4)}")
        lift_qpos = pick_object(scene, robot, ee_link, cube, cfg, use_plan_path=use_plan_path)
        _ = place_object(
            scene,
            robot,
            ee_link,
            lift_qpos,
            source_pos=cube_pos_before,
            goal_pos=goal,
            cfg=cfg,
            use_plan_path=use_plan_path,
        )


def evaluate_task(cubes, goals: list[np.ndarray], cfg: BaselineConfig):
    errors = []
    successes = 0
    for idx, (cube, goal) in enumerate(zip(cubes, goals)):
        cube_pos = to_numpy(cube.get_pos())
        xy_err = float(np.linalg.norm(cube_pos[:2] - np.asarray(goal[:2], dtype=float)))
        z_err = float(abs(cube_pos[2] - goal[2]))
        success = xy_err < cfg.success_threshold and z_err < cfg.success_threshold
        successes += int(success)
        errors.append((xy_err, z_err))
        print(
            f"[Eval] cube {idx}: final={np.round(cube_pos, 4)}, goal={np.round(goal, 4)}, "
            f"xy_err={xy_err:.4f}, z_err={z_err:.4f}, success={success}"
        )

    return {
        "num_success": successes,
        "num_total": len(cubes),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Franka object rearrangement baseline")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show viewer")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU backend")
    parser.add_argument(
        "--no-plan-path",
        action="store_true",
        default=False,
        help="Disable built-in plan_path and use direct pose control only",
    )
    args = parser.parse_args()

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32")

    cfg = BaselineConfig()
    scene, entities = build_scene(cfg, show_viewer=args.vis, backend=backend)

    robot = entities["robot"]
    cubes = entities["cubes"]
    goals = [np.asarray(pos, dtype=float) for pos in cfg.goal_positions]

    run_rearrangement_task(
        scene,
        robot,
        cubes,
        goals,
        cfg,
        use_plan_path=not args.no_plan_path,
    )

    for _ in range(cfg.hold_steps):
        scene.step()

    report = evaluate_task(cubes, goals, cfg)
    total_steps = scene.sim.cur_step if hasattr(scene.sim, "cur_step") else None
    print()
    print("=== Rearrangement Summary ===")
    print(f"Success: {report['num_success']} / {report['num_total']}")
    if total_steps is not None:
        print(f"Total simulation steps: {total_steps}")
        print(f"Elapsed simulated time [s]: {total_steps * cfg.dt:.3f}")


if __name__ == "__main__":
    main()
