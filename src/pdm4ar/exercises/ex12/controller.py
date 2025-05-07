import numpy as np
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from shapely.geometry import LineString, Point
from pdm4ar.exercises.ex05.structures import mod_2_pi


class Controller:
    def __init__(self, scenario, sp: VehicleParameters, sg: VehicleGeometry, dt, name, orientation):
        self.dt = dt
        self.scenario = scenario
        self.sp = sp
        self.sg = sg
        self.name = name
        self.closest_car_name = None
        self.front_car_name = None
        self.car_found = False
        self.orientation = orientation
        self.delta_controller = PIDController(kp=0.4, ki=0.2, kd=0.7)
        self.wall_cars_controller = PIDController(kp=5, ki=0.2, kd=5)
        self.errors = []
        self.time = []
        self.i = 0

    def reset_PID(self):
        self.delta_controller.reset_integral()

    def compute_actual_commands(self, current_state, desired_state) -> VehicleCommands:
        """
        This method is called by the simulator to compute the actual commands to be executed
        :param current_state: the current state of the agent at the current time step
        :param desired_state: the desired state of the agent at the next time step
        :return: the actual commands to be executed
        """
        # Compute the actual acceleration
        acc = (desired_state.vx - current_state.vx) / self.dt

        # Compute the actual ddelta
        ddelta = (desired_state.delta - current_state.delta) / self.dt

        return VehicleCommands(acc=acc, ddelta=ddelta)

    def maintain_lane(
        self,
        current_state: VehicleState,
        sim_obs,
        control_points: list[np.ndarray],
        arrived,
        car_to_follow=None,
        wall_cars=False,
    ) -> VehicleCommands:
        """
        This method is called by the simulator to mantein the lane
        :param current_state: the current state of the agent at the current time step
        :param sim_obs: the current observations of the simulator
        :return: the actual commands to be executed
        """
        # Find the car in front of the agent
        front_car = self._find_front_car(current_state, sim_obs)
        max_speed = 25

        if car_to_follow is not None and car_to_follow in sim_obs.players:
            agent = sim_obs.players[car_to_follow].state
            distance_to_cover = (
                np.sqrt((agent.x - current_state.x) ** 2 + (agent.y - current_state.y) ** 2) - 2 * self.sg.length
            )
            max_speed = self._compute_max_speed(distance_to_cover, agent.vx, current_state.vx)
        elif car_to_follow is not None:
            max_speed = 25
        else:
            # Compute the actual acceleration
            if front_car is None:
                acc = max(min((max_speed - current_state.vx) / self.dt, self.sp.acc_limits[1]), self.sp.acc_limits[0])
            elif isinstance(front_car, VehicleCommands):
                return front_car
            else:
                # Compute the distance to cover
                distance_to_cover = (
                    np.sqrt((front_car.state.x - current_state.x) ** 2 + (front_car.state.y - current_state.y) ** 2)
                    - 2 * self.sg.length
                )
                max_speed = self._compute_max_speed(distance_to_cover, front_car.state.vx, current_state.vx)

        acc = max(min(self.sp.acc_limits[1], (max_speed - current_state.vx) / self.dt), self.sp.acc_limits[0])

        ddelta = self._delta_control(control_points, current_state=current_state, wall_cars=wall_cars)

        return VehicleCommands(acc=acc, ddelta=ddelta)

    def _compute_max_speed(self, distance_to_cover: float, speed_goal: float, current_speed: float) -> float:
        """
        This method computes the maximum speed of the agent considering the distance from the car in front
        of the agent
        :param distance_to_cover: the distance to cover to reach the car in front of the agent
        :param speed_goal: the current_speed of the agent
        :param current_speed: my current speed
        :return: the maximum speed of the agent
        """
        # Consider the maximum speed and dec of the agent
        max_speed = 25
        max_dec = self.sp.acc_limits[0]

        # Compute the distance at the next state considering my current speed and the speed of the car in front
        distance = distance_to_cover + (speed_goal - current_speed) * self.dt

        # Compute the maximum speed considering the distance to cover
        if distance > 0:
            speed_at_next_state = max(
                min(max_speed, np.sqrt(speed_goal**2 - 2 * max_dec * distance)), self.sp.vx_limits[0]
            )
        else:
            speed_at_next_state = speed_goal

        return speed_at_next_state

    def _find_front_car(self, current_state: VehicleState, sim_obs):
        """
        This function finds the car in front of the agent
        :param current_state: the current state of the agent at the current time step
        :param sim_obs: the current observations of the simulator
        :return: the car in front of the agent, None if there is no car in front
        """
        # Find the lanelet of the agent
        my_position = [np.array([current_state.x, current_state.y])]
        try:
            my_lanelet_ID = self.scenario.find_lanelet_by_position(my_position)[0][0]
        except IndexError:
            return VehicleCommands(acc=0, ddelta=0)

        # Find the predecessor and the successor of the lanelet
        my_lanelet_IDs = self.successor_and_predecessor(my_lanelet_ID)

        # Find the car in front of the agent
        for name, agent in sim_obs.players.items():
            if name != self.name:
                agent_position = [np.array([agent.state.x, agent.state.y])]

                # Try to find the lanelet of the agent, if the list is empty continue
                try:
                    agent_lanelet = self.scenario.find_lanelet_by_position(agent_position)[0][0]
                except IndexError:
                    continue
                if (agent_lanelet in my_lanelet_IDs) and (
                    (agent.state.x > current_state.x and np.cos(self.orientation) > 0)
                    or (agent.state.x < current_state.x and np.cos(self.orientation) < 0)
                ):
                    return agent

        # If there is no car in front of the agent, return None
        return None

    def successor_and_predecessor(self, my_lanelet_ID: int) -> tuple:
        """
        This function returns the successor and the predecessor of the goal lanelet
        :param my_lanelet_ID: the ID of the lanelet of the agent
        :return: the successor and the predecessor of the goal lanelet
        """
        my_lanelet = self.scenario.find_lanelet_by_id(my_lanelet_ID)
        try:
            lane_suc_ID = my_lanelet.successor[0]
        except IndexError:
            lane_suc_ID = None
        try:
            lane_pre_ID = my_lanelet.predecessor[0]
        except IndexError:
            lane_pre_ID = None

        return (lane_suc_ID, lane_pre_ID, my_lanelet_ID)

    def maintain_in_wall_cars(
        self,
        current_state: VehicleState,
        sim_obs,
        goal_IDs,
        goal_lane_is_right: bool,
        control_points: list[np.ndarray],
        project_segment: float,
    ) -> tuple:
        """
        This method is called by the simulator to mantein the lane in the wall cars condition
        :param current_state: the current state of the agent at the current time step
        :param sim_obs: the current observations of the simulator
        :return: the actual commands to be executed
        """
        agents_in_goal = []
        agents = sim_obs.players
        for agent_name, agent in agents.items():
            position = [np.array([agent.state.x, agent.state.y])]
            lanelet = self.scenario.find_lanelet_by_position(position)[0][0]
            if lanelet in goal_IDs:
                agents_in_goal.append(agent_name)

        # Project my position in the goal lane
        angle = self.orientation
        if goal_lane_is_right:
            angle -= np.pi / 2
        else:
            angle += np.pi / 2

        my_x_in_goal = current_state.x + project_segment * np.cos(angle)
        my_y_in_goal = current_state.y + project_segment * np.sin(angle)

        if self.closest_car_name is None or not (
            (
                (agents[self.closest_car_name].state.x + self.sg.length / 2 * np.cos(self.orientation) * 1.5)
                > my_x_in_goal
                and np.cos(self.orientation) > 0
            )
            or (
                (agents[self.closest_car_name].state.x + self.sg.length / 2 * np.cos(self.orientation) * 1.5)
                < my_x_in_goal
                and np.cos(self.orientation) < 0
            )
        ):
            # Find the car in front of the agent
            distances = {}
            for agent_name in agents_in_goal:
                agent = agents[agent_name]
                if (
                    (agent.state.x + (self.sg.length / 2 * np.cos(self.orientation))) > my_x_in_goal
                    and np.cos(self.orientation) > 0
                ) or (
                    (agent.state.x - (self.sg.length / 2 * np.cos(self.orientation))) < my_x_in_goal
                    and np.cos(self.orientation) < 0
                ):
                    distances[agent_name] = np.sqrt(
                        (agent.state.x - my_x_in_goal) ** 2 + (agent.state.y - my_y_in_goal) ** 2
                    )
            keys = sorted(distances, key=distances.get)  # type: ignore
            self.closest_car_name = keys[0]
            try:
                self.front_car_name = keys[1]
            except IndexError:
                self.front_car_name = None

        # Compute the acceleration
        closest_agent = agents[self.closest_car_name]
        if (closest_agent.state.x > my_x_in_goal and np.cos(self.orientation) > 0) or (
            closest_agent.state.x < my_x_in_goal and np.cos(self.orientation) < 0
        ):
            distance = np.sqrt(
                (closest_agent.state.x - my_x_in_goal) ** 2 + (closest_agent.state.y - my_y_in_goal) ** 2
            )
        else:
            distance = -np.sqrt(
                (closest_agent.state.x - my_x_in_goal) ** 2 + (closest_agent.state.y - my_y_in_goal) ** 2
            )
        distance_to_cover = distance + self.sg.length / 2
        other_speed = closest_agent.state.vx
        speed_next_state = self._compute_max_speed(distance_to_cover, other_speed, current_state.vx)

        # COMPUTE THE SPEED TO NOT COLLIDE WITH THE CAR IN FRONT
        # Find the car in front of the agent
        front_car = self._find_front_car(current_state, sim_obs)
        max_speed = 25

        # Compute the actual acceleration
        if front_car is None:
            max_speed_front = max_speed
        elif isinstance(front_car, VehicleCommands):
            return front_car, self.closest_car_name, self.front_car_name, distance_to_cover
        else:
            # Compute the distance to cover
            distance_to_cover_front = (
                np.sqrt((front_car.state.x - current_state.x) ** 2 + (front_car.state.y - current_state.y) ** 2)
                - 2 * self.sg.length
            )
            max_speed_front = self._compute_max_speed(distance_to_cover_front, front_car.state.vx, current_state.vx)

        if self.front_car_name is not None:
            cars_distance = np.sqrt(
                (agents[self.front_car_name].state.x - agents[self.closest_car_name].state.x) ** 2
                + (agents[self.front_car_name].state.y - agents[self.closest_car_name].state.y) ** 2
            )
        else:
            cars_distance = 0

        if cars_distance < self.sg.length:
            speed = max_speed_front
        else:
            speed = min(speed_next_state, max_speed_front)

        speed = min(speed_next_state, max_speed_front)

        acc = max(min(self.sp.acc_limits[1], (speed - current_state.vx) / self.dt), self.sp.acc_limits[0])
        ddelta = self._delta_control(control_points, current_state=current_state, wall_cars=True)

        commands = VehicleCommands(acc=acc, ddelta=ddelta)

        return commands, self.closest_car_name, self.front_car_name, distance_to_cover

    def _delta_control(self, control_points: list[np.ndarray], current_state: VehicleState, wall_cars) -> float:
        """
        This function computes the delta to follow the lane
        :param control_points: the control points of the lane
        :param current_state: the current state of the agent
        :return: the delta to follow the lane
        """
        # Create the lane for control points
        x_coords = np.array([point[0] for point in control_points])
        y_coords = np.array([point[1] for point in control_points])
        coefficients = np.polyfit(x_coords, y_coords, 1)
        m, b = coefficients
        margin = 20
        x_min = min(x_coords) - margin
        x_max = max(x_coords) + margin
        y_min = m * x_min + b
        y_max = m * x_max + b
        center_line = LineString([(x_min, y_min), (x_max, y_max)])

        modulo = self.sg.length

        x_primo = current_state.x + modulo * np.cos(current_state.psi)
        y_primo = current_state.y + modulo * np.sin(current_state.psi)
        point = Point(x_primo, y_primo)
        distance = center_line.project(point)
        distance2 = abs(m * x_primo - y_primo + b) / np.sqrt(m**2 + 1)
        projection = center_line.interpolate(distance)
        x_to_follow, y_to_follow = projection.x, projection.y

        angle_to_follow = np.arctan2(y_to_follow - current_state.y, x_to_follow - current_state.x)
        if np.abs(current_state.psi - angle_to_follow) > 4:
            angle_to_follow = mod_2_pi(angle_to_follow)
            angle_error = mod_2_pi(current_state.psi) - angle_to_follow
        else:
            angle_error = current_state.psi - angle_to_follow

        self.errors.append(angle_error)
        self.time.append(self.i)
        self.i += 1

        """plt.figure()
        plt.plot(self.time, self.errors)
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.title("Error")
        plt.grid(True)
        plt.savefig("error.png")

        x_center, y_center = center_line.xy
        plt.figure()
        plt.plot(x_center, y_center, label="Center Line")

        plt.plot(point.x, point.y, "ro", label="Point")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Center Line and Point")
        plt.legend()
        plt.grid(True)
        plt.savefig("center_line_and_point.png")

        print(f"angle to follow: {angle_to_follow}")
        print(f"current psi: {current_state.psi}")"""

        # Choose the controller for delta considering the wall cars condition
        if wall_cars:
            ddelta = self.wall_cars_controller.compute(0, angle_error)
        else:
            ddelta = self.delta_controller.compute(0, angle_error)

        # print(f"ddelta: {ddelta}")

        return ddelta


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.max_error = 0

    def reset_integral(self):
        self.integral = 0

    def compute(self, setpoint, measured_value):
        dt = 0.1
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error

        if abs(error) > self.max_error:
            self.max_error = abs(error)

        return output
