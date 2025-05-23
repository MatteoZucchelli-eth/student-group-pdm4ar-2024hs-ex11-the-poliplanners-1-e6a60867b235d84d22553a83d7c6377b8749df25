from dataclasses import dataclass
from mimetypes import init
from typing import Sequence
from decimal import Decimal
import numpy as np

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters

from pdm4ar.exercises.ex11.planner import SpaceshipPlanner
from pdm4ar.exercises_def.ex09 import goal
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

from shapely import LineString
from shapely.wkt import loads


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1


class SpaceshipAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    def __init__(
        self,
        init_state: SpaceshipState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SpaceshipAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets
        self.parameters = MyAgentParams()

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.
        Do **not** modify the signature of this method.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.goal_state = init_sim_obs.goal
        # Extract boundaries
        boundaries_list_linestring = [
            obs.shape for obs in init_sim_obs.dg_scenario.static_obstacles if isinstance(obs.shape, LineString)
        ]
        boundaries_linestring = boundaries_list_linestring[0]
        geometry = loads(boundaries_linestring.wkt)
        boundaries = list(
            geometry.coords
        )  # list of tuples (x, y) for vertices: NOTE: the first and the last point are the same
        self.boundaries = boundaries[:-1]  # remove the last point

        # Create the planner
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.planner = SpaceshipPlanner(
            planets=self.planets,
            satellites=self.satellites,
            sg=self.sg,
            sp=self.sp,
            goal_state=self.goal_state,
            boundaries=self.boundaries,
        )

        self.cmds_plan, self.state_traj, self.final_time = self.planner.compute_trajectory(self.init_state)
        self.already_in = False

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)
        Do **not** modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        expected_state = self.state_traj.at_interp(sim_obs.time)

        if np.linalg.norm(current_state.as_ndarray() - expected_state.as_ndarray()) >= 0.2:
            self.already_in = True
            self.new_planner = SpaceshipPlanner(
                planets=self.planets,
                satellites=self.satellites,
                sg=self.sg,
                sp=self.sp,
                goal_state=self.goal_state,
                init_time=float(sim_obs.time),
                second_time=True,
                old_p=self.final_time,
                boundaries=self.boundaries,
            )
            new_init = SpaceshipState.from_array(current_state.as_ndarray())
            init_control = self.cmds_plan.at_interp(sim_obs.time).as_ndarray()
            self.cmds_plan, self.state_traj, new_final_time = self.new_planner.compute_trajectory(
                new_init,
                float(sim_obs.time),
                init_control=init_control,
                old_X=self.state_traj._values,
                old_U=self.cmds_plan._values,
            )

            self.final_time = new_final_time + float(sim_obs.time)

        cmds = self.cmds_plan.at_interp(sim_obs.time)

        return cmds
