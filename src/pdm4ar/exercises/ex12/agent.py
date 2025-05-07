from dataclasses import dataclass
from typing import Sequence
from cycler import V
from dg_commons import SE2Transform

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from shapely import LineString, Point
from .controller import Controller
from dg_commons import SE2Transform
import numpy as np
import math


@dataclass(frozen=True)
class Pdm4arAgentParams:
    a = 0  # number of points for the trajectory (equal to number of points of our trajectory)


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    sg: VehicleGeometry
    sp: VehicleParameters
    min_turning_radius: float
    start: SE2Transform
    goal: SE2Transform
    path: Sequence[SE2Transform]

    def __init__(self):
        # Create a dictionary to store the trajectories of other agents
        self.trajectory_started = False
        self.dt = 0.1
        self.wall_cars = False
        self.closest_car_name = None
        self.numb_or_points = 4
        self.car_to_follow = None
        self.goal_control_points = None
        self.arrived = False

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal  # type: ignore
        self.sg = init_obs.model_geometry  # type: ignore
        self.sp = init_obs.model_params  # type: ignore

        # Create a dictionary to store the speeds of other agents
        self.scenario: LaneletNetwork = init_obs.dg_scenario.lanelet_network  # type: ignore
        self.goal_points = init_obs.goal.ref_lane.control_points  # type: ignore
        point1 = self.goal_points[0].q.p
        point2 = self.goal_points[1].q.p
        self.orientation = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])  # type: ignore
        self.controller = Controller(self.scenario, self.sp, self.sg, self.dt, self.name, self.orientation)
        self.goal_ID = self.scenario.find_lanelet_by_position([self.goal_points[1].q.p])[0][0]
        self.goal_IDs = self.controller.successor_and_predecessor(self.goal_ID)

        # Find all the goal control points
        for ID in self.goal_IDs:
            if ID is not None:
                if self.goal_control_points is None:
                    self.goal_control_points = self.scenario.find_lanelet_by_id(ID).center_vertices
                else:
                    self.goal_control_points = np.vstack(
                        (self.goal_control_points, self.scenario.find_lanelet_by_id(ID).center_vertices)
                    )
        # Sort the control points based on the orientation of the car
        if np.cos(self.orientation) > 0:
            self.goal_control_points = self.goal_control_points[self.goal_control_points[:, 0].argsort()]
        else:
            self.goal_control_points = self.goal_control_points[self.goal_control_points[:, 0].argsort()[::-1]]
        # Remove duplicate rows
        self.goal_control_points = np.unique(self.goal_control_points, axis=0)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        self.current_state: VehicleState = sim_obs.players[self.name].state  # type: ignore
        self.sim_obs = sim_obs

        # Check if the goal lane is on the right side of the car
        if float(sim_obs.time) == 0.0:
            self.goal_lane_is_right = self._point_is_right(self.goal_points[1].q.p)
            if self._wall_cars():
                self.wall_cars = True

        # Compute the distance from the center line in the goal lane
        self.distance_from_center = self._distance_from_center_line()

        # If i'm arrived reset the PID controller
        if self.distance_from_center < 0.1 and not self.arrived:
            self.arrived = True
            self.controller.reset_PID()

        # If there isn't a car close to me, start the steering control
        if not self._car_side() and not self.trajectory_started:
            self.trajectory_started = True
            self._car_to_follow()
            self.controller.reset_PID()
        elif self._car_side() and not self.trajectory_started:
            self.car_to_follow = None

        # If the trajectory is started, set the control points with goal points
        if self.trajectory_started:
            my_control_points = self.goal_control_points
        # If the trajectory is not started, set the control points with the lanelet points
        else:
            self.my_ID = self.scenario.find_lanelet_by_position(
                [np.array([self.current_state.x, self.current_state.y])]
            )[0][0]
            my_control_points = self.scenario.find_lanelet_by_id(self.my_ID).center_vertices

        # Consider only the closest points as orientation points
        orientation_points = self._compute_orientation_points(my_control_points)

        if self.wall_cars and not self.trajectory_started:
            commands, orientation_points, self.car_to_follow = self._control_wall_cars(my_control_points)

        # If we're not in wall_cars or we're in wall cars and the steering is started, start the steering
        if orientation_points is not None:
            # Follow the computed orientation points
            commands = self.controller.maintain_lane(
                self.current_state, sim_obs, orientation_points, self.arrived, self.car_to_follow, self.wall_cars
            )

        return commands

    def _car_to_follow(self):
        """
        This function returns the name of the car to follow
        """
        distances = {}
        for agent_name in self.sim_obs.players:
            agent = self.sim_obs.players[agent_name]

            try:
                agent_position = [np.array([agent.state.x, agent.state.y])]
                lanelet = self.scenario.find_lanelet_by_position([agent_position])[0][0]
            except IndexError:
                lanelet = self.goal_ID

            if agent_name != self.name and lanelet in self.goal_IDs:
                angle = self.orientation
                if self.goal_lane_is_right:
                    angle -= np.pi / 2
                else:
                    angle += np.pi / 2

                my_x_in_goal = self.current_state.x + self.distance_from_center * np.cos(angle)
                my_y_in_goal = self.current_state.y + self.distance_from_center * np.sin(angle)

                # If the car is in the front of me, consider the max deceleration, otherwise the max acceleration
                if (
                    agent.state.x > my_x_in_goal + 2 * self.sg.length * np.cos(self.orientation)
                    and np.cos(self.orientation) > 0
                ) or (
                    agent.state.x < my_x_in_goal + 2 * self.sg.length * np.cos(self.orientation)
                    and np.cos(self.orientation) < 0
                ):
                    distances[agent_name] = np.linalg.norm(
                        np.array([agent.state.x, agent.state.y]) - np.array([my_x_in_goal, my_y_in_goal])
                    )
        if distances:
            self.car_to_follow = min(distances, key=distances.get)  # type: ignore
        else:
            self.car_to_follow = None

    def _car_side(self) -> bool:
        """
        This function checks if the car is on the side of the ego vehicle
        """
        for agent_name in self.sim_obs.players:
            agent = self.sim_obs.players[agent_name]

            try:
                agent_position = [np.array([agent.state.x, agent.state.y])]
                lanelet = self.scenario.find_lanelet_by_position([agent_position])[0][0]
            except IndexError:
                lanelet = self.goal_ID

            if agent_name != self.name and lanelet in self.goal_IDs:
                angle = self.orientation
                if self.goal_lane_is_right:
                    angle -= np.pi / 2
                else:
                    angle += np.pi / 2

                my_x_in_goal = self.current_state.x + self.distance_from_center * np.cos(angle)

                # If the car is in the front of me, consider the max deceleration, otherwise the max acceleration
                if (
                    my_x_in_goal - self.sg.length * np.cos(self.orientation)
                    <= agent.state.x
                    <= my_x_in_goal + 2 * self.sg.length * np.cos(self.orientation)
                    and np.cos(self.orientation) > 0
                ) or (
                    my_x_in_goal - self.sg.length * np.cos(self.orientation)
                    >= agent.state.x
                    >= my_x_in_goal + 2 * self.sg.length * np.cos(self.orientation)
                    and np.cos(self.orientation) < 0
                ):
                    return True

        return False

    def _point_is_right(self, goal_point):
        """
        This function checks if the goal lane is on the right side of the car
        :param current_x: x coordinate of the current position
        :param current_y: y coordinate of the current position
        :param current_psi: current orientation of the car
        :param goal_x: x coordinate of the goal position
        :param goal_y: y coordinate of the goal position
        :return: True if the goal is on the right side of the car, False otherwise
        """
        goal_x = goal_point[0]
        goal_y = goal_point[1]
        current_x = self.current_state.x
        current_y = self.current_state.y
        current_psi = self.current_state.psi

        # Calculate the heading vector
        heading_x = math.cos(current_psi)
        heading_y = math.sin(current_psi)

        # Vector from current position to the goal
        dx = goal_x - current_x
        dy = goal_y - current_y

        # Compute the cross product
        cross = heading_x * dy - heading_y * dx

        # Return True if the goal is to the right
        return cross < 0

    def _wall_cars(self) -> bool:
        """
        This function checks if there is a lot of cars in the goal lanelet
        :param sim_obs: the current observations of the simulator
        :return: True if there are a lot of cars in the goal lanelet, False otherwise
        """
        car_positions = []
        agents_in_goal = []
        agents = self.sim_obs.players
        for agent_name, agent in agents.items():
            position = [np.array([agent.state.x, agent.state.y])]
            try:
                lanelet = self.scenario.find_lanelet_by_position(position)[0][0]
            except IndexError:
                continue
            if lanelet in self.goal_IDs:
                agents_in_goal.append(agent_name)
                car_positions.append((agent.state.x, agent.state.y))
        distances_between_cars = []
        for i in range(len(car_positions) - 1):
            distances_between_cars.append(
                np.linalg.norm(
                    np.array([car_positions[i][0], car_positions[i][1]])
                    - np.array([car_positions[i + 1][0], car_positions[i + 1][1]])
                )
            )

        if len(agents_in_goal) > 5:  # or np.mean(distances_between_cars) < 2 * self.sg.length:
            return True
        else:
            return False

    def _control_wall_cars(self, control_points):
        """
        This function controls the car when there are a lot of cars in the goal lanelet
        :param orientation_points: the orientation points of the car
        :return: the commands to control the car
        """
        # Compute the distance bewteen the front car and the closest car
        if self.closest_car_name is not None:
            if self.front_car_name is not None:
                distance = np.linalg.norm(
                    np.array(
                        [
                            self.sim_obs.players[self.front_car_name].state.x,
                            self.sim_obs.players[self.front_car_name].state.y,
                        ]
                    )
                    - np.array(
                        [
                            self.sim_obs.players[self.closest_car_name].state.x,
                            self.sim_obs.players[self.closest_car_name].state.y,
                        ]
                    )
                )
            else:
                distance = np.inf

            # If the distance is greater than the length of the car and the car is not moving, start the trajectory
            if distance == np.inf or (
                distance >= self.sg.length * 1.5
                and round(self.current_state.vx - self.sim_obs.players[self.closest_car_name].state.vx, 1) == 0
                and round(self.distance_to_cover, 1) == 0
            ):
                orientation_points = self._compute_orientation_points(self.goal_control_points)
                self.trajectory_started = True
                return VehicleCommands(0, 0), orientation_points, self.front_car_name

        # DA SISTEMAREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE (si può adattare al classico maintain_lane)
        commands, self.closest_car_name, self.front_car_name, self.distance_to_cover = (
            self.controller.maintain_in_wall_cars(
                self.current_state,
                self.sim_obs,
                self.goal_IDs,
                self.goal_lane_is_right,
                control_points,
                self.distance_from_center,
            )
        )
        return commands, None, None

    def _distance_from_center_line(self) -> float:
        """
        This function computes the distance between the car and the center line of the lane
        :return: the distance between the car and the center line of the lane
        """
        distances = [
            np.linalg.norm(np.array([self.current_state.x, self.current_state.y]) - point)
            for point in self.goal_control_points
        ]
        closest_indexes = np.argsort(distances)[:2]
        closest_points = [self.goal_control_points[i] for i in closest_indexes]

        margin = 100
        point1 = min(closest_points, key=lambda x: x[0])
        point2 = max(closest_points, key=lambda x: x[0])
        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        point1 = np.array([point1[0] - margin * np.cos(angle), point1[1] - margin * np.sin(angle)])
        point2 = np.array([point2[0] + margin * np.cos(angle), point2[1] + margin * np.sin(angle)])
        line = LineString([point1, point2])
        point = Point(self.current_state.x, self.current_state.y)

        # Compute the distance between the car and the lane boundary
        distance = point.distance(line)

        return float(distance)

    def _compute_orientation_points(self, control_points):
        """
        This function computes the orientation points of the car
        :param control_points: the control points of the lane
        :return: the orientation points of the car
        """
        if len(control_points) < self.numb_or_points:
            orientation_points = control_points
        else:
            # Find the numb_of_points control_points closest to the car
            distances = [
                np.linalg.norm(np.array([self.current_state.x, self.current_state.y]) - point)
                for point in control_points
            ]
            closest_indexes = np.argsort(distances)[: self.numb_or_points]
            orientation_points = [control_points[i] for i in closest_indexes]
        return orientation_points
