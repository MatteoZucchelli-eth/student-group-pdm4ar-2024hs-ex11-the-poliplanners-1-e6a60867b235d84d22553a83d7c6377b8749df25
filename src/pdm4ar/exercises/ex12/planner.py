from mimetypes import init
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleModel
from dg_commons import SE2Transform
import numpy as np
import math
from dg_commons.sim.models.vehicle import VehicleState
from .structures import *
import matplotlib.pyplot as plt
import copy


class Planner:
    def __init__(self, path: Path, max_acc: float):
        self.path = path
        self.vehicle_states = {}
        self.max_acc = max_acc

        # Convert the path to a list of states
        interpolator = PathPoints(self.path)
        self.trajectory = interpolator.extract_path_points()  # list of SE2Transform

    def compute_vehicle_states(
        self, init_speed: float, goal_speed: float, radius: float, wheelbase: float
    ) -> dict[float, VehicleState]:
        """
        Compute the vehicle states from the path
        :param path: Path object
        :param init_speed: initial speed of the vehicle
        :param goal_speed: goal speed of the vehicle
        :param radius: radius of the turning circle
        :param wheelbase: wheelbase of the vehicle
        :return: dict with the time steps (0.1) as key and the vehicle states (VehicleState) as value
        """
        # Compute the total length of the path
        trajectory_length = 0
        for segment in self.path:
            trajectory_length += segment.length

        # Compute the constant acceleration to reach the goal speed
        acc = min((goal_speed**2 - init_speed**2) / (2 * trajectory_length), self.max_acc)

        # Compute the time to reach the goal speed
        if acc == 0:
            total_time = trajectory_length / init_speed
        else:
            discriminant = init_speed**2 + 2 * acc * trajectory_length
            if discriminant < 0:
                raise ValueError("The discriminant is negative")
            total_time = (-init_speed + math.sqrt(discriminant)) / acc

        # Compute the speed for each point in the trajectory
        speeds = np.linspace(init_speed, goal_speed, len(self.trajectory))

        # Compute the different time steps for each point in the self.trajectory
        time_steps = np.linspace(0, total_time, len(self.trajectory))

        # Create the dict where 0.1 are the keys and the values are the states
        for j in range(int(np.floor(total_time * 10) + 1)):
            t = j / 10
            for i, time in enumerate(time_steps):
                if t == 0:
                    new_x = self.trajectory[i].p[0]
                    new_y = self.trajectory[i].p[1]
                    new_psi = self.trajectory[i].theta
                    new_v = speeds[i]
                    new_delta = 0
                    break
                if time == t:
                    new_x = self.trajectory[i].p[0]
                    new_y = self.trajectory[i].p[1]
                    new_psi = self.trajectory[i].theta
                    new_v = speeds[i]
                    delta_psi = self.trajectory[i].theta - self.trajectory[i - 1].theta
                    dpsi = delta_psi / (time_steps[i] - time_steps[i - 1])
                    new_delta = math.atan((wheelbase * dpsi) / new_v)
                    break
                if time < t <= time_steps[i + 1]:
                    dt = t - time
                    # Compute the distance along the curve axis
                    delta_s = speeds[i] * dt + 0.5 * acc * dt**2

                    # Decompose the distance along the curve axis into x and y components
                    alpha = delta_s / radius
                    chord_length = 2 * radius * math.sin(alpha / 2)

                    # Find the center of the turning circle
                    if self.trajectory[i].p[0] < self.path[1].start_config.p[0]:
                        center = self.path[0].center.p  # type: ignore
                    else:
                        center = self.path[1].center.p  # type: ignore

                    # Compute the new x and y
                    alpha_radius = math.atan2(center[1] - self.trajectory[i].p[1], center[0] - self.trajectory[i].p[0])
                    ang_int = (np.pi - alpha) / 2
                    my_ang = alpha_radius + ang_int
                    new_x = self.trajectory[i].p[0] + chord_length * math.cos(my_ang)
                    new_y = self.trajectory[i].p[1] + chord_length * math.sin(my_ang)

                    # Compute the new psi and v
                    old_delta_s = speeds[i] * (time_steps[i + 1] - time) + 0.5 * acc * (time_steps[i + 1] - time) ** 2
                    new_psi = self.trajectory[i].theta + (delta_s / old_delta_s) * (
                        mod_2_pi(self.trajectory[i + 1].theta) - mod_2_pi(self.trajectory[i].theta)
                    )
                    new_v = speeds[i] + acc * dt

                    # Compute the new delta
                    if np.abs(new_psi - self.trajectory[i].theta) > 5:
                        delta_psi = new_psi - self.trajectory[i].theta - 2 * np.pi
                    else:
                        delta_psi = new_psi - self.trajectory[i].theta
                    dpsi = delta_psi / dt
                    new_delta = math.atan((wheelbase * dpsi) / new_v)
                    break

            self.vehicle_states[t] = VehicleState(x=new_x, y=new_y, psi=mod_2_pi(new_psi), vx=new_v, delta=new_delta)

        # Add the last point
        delta_s = speeds[-1] * 0.1 + 0.5 * acc * 0.1**2
        psi = self.trajectory[-1].theta
        vx = speeds[-1] + acc * 0.1
        self.vehicle_states[round(t + 0.1, 1)] = VehicleState(
            x=self.trajectory[-1].p[0] + np.cos(psi) * delta_s,
            y=self.trajectory[-1].p[1] + np.sin(psi) * delta_s,
            psi=psi,
            vx=vx,
            delta=0,
        )

        # self._plot_vehicle_states()
        self._smooth_delta()

        return self.new_trajectory

    def _smooth_delta(self):
        """
        This function smooths the delta values of the vehicle states
        """
        tf = round(((len(self.vehicle_states) - 1) / 10), 1)
        self.new_trajectory = copy.deepcopy(self.vehicle_states)
        delta = self.vehicle_states[0.1].delta
        xi = self.vehicle_states[0.0].x  # Get the first x value
        xf = self.vehicle_states[round(max(self.vehicle_states.keys()), 1)].x  # Get the last x value

        # Compute sin and approximate for the number of element of trajectory
        deltas = {}
        X = np.linspace(0, xf - xi, 1000)
        for xs in X:
            deltas[xs] = np.pi / 2 * delta * np.sin(2 * np.pi / (xf - xi) * xs)

        # Find the delta corrisponding at each x in the trajecotry
        for ts in range(0, int(tf * 10) + 1):
            ts = round(float(ts / 10), 1)
            x_state = self.vehicle_states[ts].x - xi
            for i in range(len(X) - 1):
                if X[i] <= x_state < X[i + 1]:
                    new_delta = deltas[X[i]] + ((X[i] - X[i + 1]) * (deltas[X[i]] - deltas[X[i + 1]])) / (
                        X[i] - X[i + 1]
                    )
                    self.new_trajectory[ts].delta = new_delta
                    break

    def _plot_vehicle_states(self):
        """
        This function plots the vehicle states
        """
        x_to_plot = [point.x for point in self.vehicle_states.values()]
        y_to_plot = [point.y for point in self.vehicle_states.values()]
        plt.figure()
        plt.plot(x_to_plot, y_to_plot)
        plt.savefig("vehicle_states.png")


class PathPoints:
    def __init__(self, path: Path):
        self.path = path
        self.number_of_points = 25

    def extract_path_points(self) -> list[SE2Transform]:
        """Extracts a fixed number of SE2Transform points on a path"""
        pts_list = []

        for _, seg in enumerate(self.path):
            # if np.allclose(seg.length, 0):
            #     continue
            seg.start_config.theta = mod_2_pi(seg.start_config.theta)
            seg.end_config.theta = mod_2_pi(seg.end_config.theta)
            if seg.type is DubinsSegmentType.STRAIGHT:
                line_pts = self.interpolate_line_points(seg)  # type: ignore
                pts_list.extend(line_pts)
            else:  # Curve
                curve_pts = self.interpolate_curve_points(seg)  # type: ignore
                pts_list.extend(curve_pts)
        pts_list.append(self.path[-1].end_config)
        return pts_list

    def interpolate_line_points(self, line: Line) -> list[SE2Transform]:
        start = line.start_config
        end = line.end_config
        start_to_end = end.p - start.p
        intervals = np.linspace(0, 1.0, self.number_of_points)
        return [SE2Transform(start.p + i * start_to_end, start.theta) for i in intervals]

    def interpolate_curve_points(self, curve: Curve) -> list[SE2Transform]:
        pts_list = []
        angle = curve.arc_angle
        direction = curve.type
        angle = direction.value * angle
        split_angle = angle / (self.number_of_points - 1)
        old_point = curve.start_config
        for _ in range(self.number_of_points):
            pts_list.append(old_point)
            point_next = self.get_next_point_on_curve(curve, point=old_point, delta_angle=split_angle)
            old_point = point_next
        return pts_list

    def get_next_point_on_curve(self, curve: Curve, point: SE2Transform, delta_angle: float) -> SE2Transform:
        point_translated = point.p - curve.center.p
        rot_matrix = self.get_rot_matrix(delta_angle)
        next_point = SE2Transform((rot_matrix @ point_translated) + curve.center.p, point.theta + delta_angle)  # type: ignore
        return next_point

    def get_rot_matrix(self, alpha: float) -> np.ndarray:
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return rot_matrix
