from dg_commons import SE2Transform
import numpy as np
from .structures import *
from dg_commons.sim.models.vehicle import VehicleState
from .planner import Planner
from scipy.optimize import fsolve


class DubinsPath:
    def __init__(
        self,
        wheelbase: float,
        max_acc: float,
        goal_lane_is_right: bool,
        delta_max: float,
        radius_coeff: float,
    ):
        self.max_acc = max_acc
        self.wheelbase = wheelbase
        self.goal_lane_is_right = goal_lane_is_right
        self.lane_width = 0
        self.delta_max = delta_max
        self.tolerance: float = 1e-5
        self.min_car_radius = self.wheelbase / np.tan(self.delta_max)
        self.radius_coeff = radius_coeff

    def set_lane_width(self, lane_width: float):
        self.lane_width = lane_width

    def calculate_min_radius(
        self,
        init_config: VehicleState,
        end_speed: float,
    ) -> float:
        """
        Calculate the Dubins path with only 2 curves depending on the goal lane
        params:
        start_config: the start configuration of the car (x,y,theta)
        radius: the turning radius of the car
        goal_lane_is_right: a boolean indicating if the goal lane is on the right side of the car
        lane_width: the width of the lane
        """
        start_speed = init_config.vx
        v_max = max(start_speed, end_speed)
        w = self.lane_width
        B = self.wheelbase

        # Equation to solve
        def equation(x):
            return w / (4 * x**2) - (self.radius_coeff * np.pi**2 * np.arctan(4 * B * x**2 / w) * v_max) / np.arcsin(x)

        # Numerical solve
        r_initial = 40
        x_initial = np.sqrt(w) / (2 * np.sqrt(r_initial))
        x_solution = fsolve(equation, x_initial)[0]

        r_solution = w / (4 * x_solution**2)
        return r_solution

    def _compute_cosines(self, states: dict[float, VehicleState]) -> list[float]:
        """
        Compute the cosines of the delta values
        """
        deltas = []
        tf = round(((len(states) - 1) / 10), 1)

        for ts in range(0, int(tf * 10)):
            ts = round(float(ts / 10), 1)
            ts_next = round(float(ts + 0.1), 1)
            difference = states[ts_next].delta - states[ts].delta
            deltas.append(abs(difference))

        return deltas

    def calculate_turning_circles(self, current_config: SE2Transform) -> TurningCircle:
        # TODO implement here your solution
        theta = current_config.theta
        x, y = current_config.p
        alpha = theta - np.pi / 2
        right_circle = Curve.create_circle(
            SE2Transform([x + self.radius * np.cos(alpha), y + self.radius * np.sin(alpha)], 0),
            current_config,
            self.radius,
            curve_type=DubinsSegmentType.RIGHT,
        )
        left_circle = Curve.create_circle(
            SE2Transform([x - self.radius * np.cos(alpha), y - self.radius * np.sin(alpha)], 0),
            current_config,
            self.radius,
            curve_type=DubinsSegmentType.LEFT,
        )
        return TurningCircle(left=left_circle, right=right_circle)

    def calculate_dubins_path(
        self,
        init_config: VehicleState,
        end_speed: float,
    ) -> dict:
        """
        Calculate the Dubins path with only 2 curves depending on the goal lane
        params:
        start_config: the start configuration of the car (x,y,theta)
        radius: the turning radius of the car
        goal_lane_is_right: a boolean indicating if the goal lane is on the right side of the car
        lane_width: the width of the lane
        """
        self.radius = self.calculate_min_radius(init_config, end_speed)
        start_config = SE2Transform([init_config.x, init_config.y], init_config.psi)
        start_speed = init_config.vx + 2

        # Compute the end configuration
        half_point_distance = np.sqrt(self.radius**2 - (self.radius - self.lane_width / 2) ** 2)  # okay

        if not self.goal_lane_is_right:
            end_config_x = start_config.p[0] + 2 * (
                half_point_distance * np.cos(start_config.theta) - self.lane_width / 2 * np.sin(start_config.theta)
            )
            # Compute path from the current lane to the left lane
            end_config_y = start_config.p[1] + 2 * (
                half_point_distance * np.sin(start_config.theta) + self.lane_width / 2 * np.cos(start_config.theta)
            )

            end_config = SE2Transform([end_config_x, end_config_y], start_config.theta)
            path = self.LR_path(start_config, end_config)
        else:
            end_config_x = start_config.p[0] + 2 * (
                half_point_distance * np.cos(start_config.theta) + self.lane_width / 2 * np.sin(start_config.theta)
            )
            # Compute path from the current lane to the right lane
            end_config_y = start_config.p[1] + 2 * (
                half_point_distance * np.sin(start_config.theta) - self.lane_width / 2 * np.cos(start_config.theta)
            )

            end_config = SE2Transform([end_config_x, end_config_y], start_config.theta)
            path = self.RL_path(start_config, end_config)

        path = Path(path)

        # Trasform the path into a vehicle state dict
        planner = Planner(path, self.max_acc)
        states = planner.compute_vehicle_states(start_speed, end_speed, self.radius, self.wheelbase)

        return states

    def LR_path(self, start_config: SE2Transform, end_config: SE2Transform):
        start_circle = self.calculate_turning_circles(start_config).left
        end_circle = self.calculate_turning_circles(end_config).right

        # Compute the middle point of the curves
        middle_point_x = (start_config.p[0] + end_config.p[0]) / 2
        middle_point_y = (start_config.p[1] + end_config.p[1]) / 2
        theta = (
            self.tan_computation(
                end_circle.center.p[0] - start_circle.center.p[0], end_circle.center.p[1] - start_circle.center.p[1]
            )
            + np.pi / 2
        )
        middle_point = SE2Transform([middle_point_x, middle_point_y], theta)

        # Set the end config in the first cirlce
        start_circle.end_config = middle_point

        # Set the start config of the second circle
        end_circle.start_config = middle_point

        # Set the angles for the circles
        for circle in [start_circle, end_circle]:
            self.set_circle_angle(circle)

        return [start_circle, end_circle]

    def RL_path(self, start_config: SE2Transform, end_config: SE2Transform):
        start_circle = self.calculate_turning_circles(start_config).right
        end_circle = self.calculate_turning_circles(end_config).left

        # Compute the middle point of the curves
        middle_point_x = (start_config.p[0] + end_config.p[0]) / 2
        middle_point_y = (start_config.p[1] + end_config.p[1]) / 2
        theta = (
            self.tan_computation(
                end_circle.center.p[0] - start_circle.center.p[0], end_circle.center.p[1] - start_circle.center.p[1]
            )
            - np.pi / 2
        )
        middle_point = SE2Transform([middle_point_x, middle_point_y], theta)
        # Set the end config in the first cirlce
        start_circle.end_config = middle_point

        # Set the start config of the second circle
        end_circle.start_config = middle_point

        # Set the angles for the circles
        for circle in [start_circle, end_circle]:
            self.set_circle_angle(circle)

        return [start_circle, end_circle]

    def set_circle_angle(self, circle: Curve):
        radius_1 = circle.start_config.p - circle.center.p
        radius_2 = circle.end_config.p - circle.center.p
        cos_ang = np.dot(radius_1, radius_2) / (np.linalg.norm(radius_1) * np.linalg.norm(radius_2))

        cos_ang = self.correct_approximation(cos_ang, 1.0)
        cos_ang = self.correct_approximation(cos_ang, -1.0)

        arc_angle = np.arccos(cos_ang)
        cross = np.cross(radius_1, radius_2)
        if (cross < 0 and circle.type == DubinsSegmentType.LEFT) or (
            cross > 0 and circle.type == DubinsSegmentType.RIGHT
        ):
            arc_angle = 2 * np.pi - arc_angle
        circle.arc_angle = arc_angle

    def tan_computation(self, delta_x: float, delta_y: float) -> float:
        if delta_x != 0:
            return np.arctan2(delta_y, delta_x)
        elif delta_y > 0:
            return np.pi / 2
        else:
            return np.pi * 1.5

    def correct_approximation(self, value: float, target: float) -> float:
        if np.isclose(value, target, atol=self.tolerance):
            return target
        return value
