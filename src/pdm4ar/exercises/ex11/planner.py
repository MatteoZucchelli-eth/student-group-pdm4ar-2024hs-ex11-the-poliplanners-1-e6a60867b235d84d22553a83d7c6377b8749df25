from calendar import c
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from math import gamma
from re import A, S
import time
from typing import Union

import warnings

from pdm4ar.exercises.ex05.structures import mod_2_pi
from pdm4ar.exercises.ex11 import spaceship
from pdm4ar.exercises.ex11.satellites import Satellite
from pdm4ar.exercises_def.ex09 import goal


warnings.filterwarnings("ignore", message=".*ECOS.*")

# from pdm4ar.exercises_def.ex11.satellites import Satellite

import sympy as spy
import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)
import matplotlib.pyplot as plt

from pdm4ar.exercises.ex11.discretization import *
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams
from pdm4ar.exercises_def.ex11.goal import DockingTarget, SpaceshipTarget


# @dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 50  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: float = 10.0  # weight for final time
    weight_distance: float = 10.0  # weight for distance
    weight_control: float = 100.0  # weight for control

    tr_radius: float = 5  # initial trust region radius - to be updated during the iterations
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-4  # Stopping criteria constant

    # Time limits
    max_time: float = 60.0  # max time for the algorithm
    min_time: float = 0.0  # min time for the algorithm


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
        goal_state: SpaceshipTarget | DockingTarget,
        boundaries: list[tuple[float, float]],
        init_time: float = 0,
        second_time: bool = False,
        old_p=0,
    ):
        """
        Pass environment information to the planner.
        """
        # Solver Parameters
        self.params = SolverParameters()
        self.eta = self.params.tr_radius
        self.cost_func_bar = 1e10
        self.iteration = 0
        self.boundaries = boundaries

        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp
        self.radius_sg = (self.sg.l_f + self.sg.l_c) * 1.3
        self.goal = goal_state
        self.goal_state = goal_state.target
        self.last_points = self.params.K // 10
        self.second_time = second_time
        self.init_time = init_time
        self.old_p = old_p

        if second_time:
            self.params.stop_crit = 1e-3
            self.params.K = max(int(np.ceil(self.params.K * (1 - self.init_time / old_p))[0]), 3)

        self.last_points = min(self.last_points, self.params.K)

        if isinstance(self.goal, DockingTarget):
            self.A_dock, self.B_dock, self.C_dock, self.A1_dock, self.A2_dock, self.theta_half = (
                self.goal.get_landing_constraint_points()
            )

            if (
                self.mod_2_pi(self.goal_state.as_ndarray()[2]) == np.pi / 2
                or self.mod_2_pi(self.goal_state.as_ndarray()[2]) == 3 * np.pi / 2
            ):
                self.tan_phi_goal = 0
            else:
                self.tan_phi_goal = np.tan(self.goal_state.as_ndarray()[2])

            if (
                self.mod_2_pi(self.goal_state.as_ndarray()[2]) == np.pi
                or self.mod_2_pi(self.goal_state.as_ndarray()[2]) == 0
            ):
                self.tan_alpha_goal = 0
            else:
                self.tan_alpha_goal = np.tan(self.goal_state.as_ndarray()[2] - np.pi / 2)

        else:  # need to populate even if there isn't a dock because we need to  construnct the constraints
            self.tan_phi_goal = 0
            self.tan_alpha_goal = 0

        # Satellite evaluator
        self.sat_evaluator = Satellite()
        self._sat_eval_func()

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self,
        init_state: SpaceshipState,
        init_time: float = 0,
        init_control: NDArray = np.zeros(2),
        old_X=np.zeros((8, 20)),
        old_U=np.zeros((2, 20)),
    ):
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.init_time = init_time
        self.init_control = init_control
        self.old_X = old_X
        self.old_U = old_U

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Set parameters that don't change during the iterations
        self._set_initial_parameters()

        # TODO: Implement SCvx algorithm or comparable
        # Iter for the max number of iterations
        for iteration in range(self.params.max_iterations):
            print(f"Iteration: {iteration}")

            # 1. Convexify the dynamic around the current trajectory and assign the values to the problem parameters
            self._convexification()

            # 2. Solve the problem
            try:
                self.error = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver, max_iters=1000
                )
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            # 3. Check convergence
            if self._check_convergence():
                print("Convergenza raggiunta.")
                break

            # 4. Update trust region
            self._update_trust_region()

            # self._plot_print()

        # self._unnormalize_variables()
        mycmds, mystates = self._sequence_from_array()

        return mycmds, mystates, self.variables["p"].value

    def initial_guess(self):
        """
        Define initial guess for SCvx.
        """
        if not self.second_time:
            U = np.linspace(self.init_control, 0, self.params.K).T

            augmented_goal_state = [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
                0,
                self.sg.m,
            ]

            # Linear interpolation between initial and goal state
            X = np.linspace(self.init_state.as_ndarray(), augmented_goal_state, self.params.K).T

            # Define initial guess for p
            p = np.ones((1)) * (self.params.max_time + self.params.min_time) / 2

        else:
            X = np.zeros((8, self.params.K))
            U = np.zeros((2, self.params.K))
            p = self.old_p * 1 / 5
            for k in range(self.params.K):
                X[:, k] = self.old_X[len(self.old_X) - self.params.K + k].as_ndarray()
                U[:, k] = self.old_U[len(self.old_U) - self.params.K + k].as_ndarray()

        return X, U, p

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
            "nu": cvx.Variable((self.spaceship.n_x, self.params.K - 1)),
            "nu_s": cvx.Variable((len(self.planets) + len(self.satellites), self.params.K)),
            "nu_ic": cvx.Variable(self.spaceship.n_x),
            "nu_tc": cvx.Variable(self.spaceship.n_x - 2),
            "nu_thrust_region": cvx.Variable(),
            "nu_constraints": cvx.Variable((3, self.last_points - 1)),
            "nu_control": cvx.Variable((self.spaceship.n_u, 2)),
            "nu_limits_X": cvx.Variable((7, self.params.K)),
            "nu_limits_U": cvx.Variable((4, self.params.K)),
        }

        self.n_x = self.spaceship.n_x
        self.n_u = self.spaceship.n_u
        self.n_p = self.spaceship.n_p

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "goal": cvx.Parameter(6),
            "A_bar": cvx.Parameter((self.n_x * self.n_x, self.params.K - 1)),
            "B_plus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1)),
            "B_minus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1)),
            "F_bar": cvx.Parameter((self.spaceship.n_x * self.n_p, self.params.K - 1)),
            "r_bar": cvx.Parameter((self.spaceship.n_x, self.params.K - 1)),
            "eta": cvx.Parameter(),
            "X_bar": cvx.Parameter((self.n_x, self.params.K)),
            "U_bar": cvx.Parameter((self.n_u, self.params.K)),
            "p_bar": cvx.Parameter(self.n_p),
            "nu_constraints_bar": cvx.Parameter((3, self.last_points - 1)),
            "nu_trust_region_bar": cvx.Parameter(),
            "init_control": cvx.Parameter(2),
            "continuity_tolerance": cvx.Parameter(2),
        }

        if self.satellites:
            problem_parameters["C_sat"] = cvx.Parameter((len(self.satellites) * 2, self.params.K))
            problem_parameters["G_sat"] = cvx.Parameter((len(self.satellites), self.params.K))
            problem_parameters["r_first_sat"] = cvx.Parameter((len(self.satellites), self.params.K))

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        constraints = []

        # TIME CONSTRAINTS
        constraints.append(self.variables["p"] >= self.params.min_time)
        constraints.append(self.variables["p"] <= self.params.max_time)

        ################################################################################################################
        # BOUDARY CONDITIONS
        x_coords, y_coords = zip(*self.boundaries)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        constraints.append(self.variables["X"][0, :] >= min_x + self.sg.l_f + self.sg.l_c)
        constraints.append(self.variables["X"][0, :] <= max_x - self.sg.l_f + self.sg.l_c)
        constraints.append(self.variables["X"][1, :] >= min_y + self.sg.l_f + self.sg.l_c)
        constraints.append(self.variables["X"][1, :] <= max_y - self.sg.l_f + self.sg.l_c)

        ################################################################################################################
        # BOUDARY CONDITIONS
        # Initial control condition
        constraints.append(
            self.variables["U"][:, 0] + self.variables["nu_control"][:, 0] == self.problem_parameters["init_control"]
        )
        constraints.append(self.variables["U"][:, -1] + self.variables["nu_control"][:, 1] == 0)
        # Terminal condition
        constraints.append(self.variables["X"][:6, -1] - self.problem_parameters["goal"] + self.variables["nu_tc"] == 0)
        # Initial condition
        constraints.append(
            self.variables["X"][:, 0] - self.problem_parameters["init_state"] + self.variables["nu_ic"] == 0
        )

        ################################################################################################################
        # PROBLEM CONSTRAINTS
        constraints.append(
            -self.variables["X"][6, :] + self.sp.delta_limits[0] <= self.variables["nu_limits_X"][0, :]
        )  # delta condition
        constraints.append(
            self.variables["X"][6, :] - self.sp.delta_limits[1] <= self.variables["nu_limits_X"][1, :]
        )  # delta condition
        constraints.append(-self.variables["X"][7, :] + self.sg.m <= self.variables["nu_limits_X"][2, :])  # m condition
        constraints.append(
            -self.variables["U"][0, :] + self.sp.thrust_limits[0] <= self.variables["nu_limits_U"][0, :]
        )  # thrust condition
        constraints.append(
            self.variables["U"][0, :] - self.sp.thrust_limits[1] <= self.variables["nu_limits_U"][1, :]
        )  # thrust condition
        constraints.append(
            -self.variables["U"][1, :] + self.sp.ddelta_limits[0] <= self.variables["nu_limits_U"][2, :]
        )  # ddelta condition
        constraints.append(
            self.variables["U"][1, :] - self.sp.ddelta_limits[1] <= self.variables["nu_limits_U"][3, :]
        )  # ddelta condition
        constraints.append(
            -self.variables["X"][3:5, :] + self.sp.vx_limits[0] <= self.variables["nu_limits_X"][3:5, :]
        )  # vx, vy condition
        constraints.append(
            self.variables["X"][3:5, :] - self.sp.vx_limits[1] <= self.variables["nu_limits_X"][5:7, :]
        )  # vx, vy condition

        ################################################################################################################
        # DYNAMICS CONSTRAINTS
        for k in range(self.params.K - 1):
            constraints.append(
                self.variables["X"][:, k + 1]
                == (
                    cvx.reshape(self.problem_parameters["A_bar"][:, k], (self.n_x, self.n_x), order="F")
                    @ self.variables["X"][:, k]
                    + cvx.reshape(self.problem_parameters["B_plus_bar"][:, k], (self.n_x, self.n_u), order="F")
                    @ self.variables["U"][:, k + 1]
                    + cvx.reshape(self.problem_parameters["B_minus_bar"][:, k], (self.n_x, self.n_u), order="F")
                    @ self.variables["U"][:, k]
                    + cvx.reshape(self.problem_parameters["F_bar"][:, k], (self.n_x, self.n_p), order="F")
                    @ self.variables["p"]
                    + self.problem_parameters["r_bar"][:, k]
                    + self.variables["nu"][:, k]
                )
            )

        ################################################################################################################
        # TRUST REGION CONSTRAINT
        constraints.append(
            cvx.norm(self.variables["X"] - self.problem_parameters["X_bar"], p=1)
            + cvx.norm(self.variables["U"] - self.problem_parameters["U_bar"], p=1)
            + cvx.norm(self.variables["p"] - self.problem_parameters["p_bar"], p=1)
            - self.problem_parameters["eta"]
            <= self.variables["nu_thrust_region"]
        )

        ################################################################################################################
        # OBSTACLE CONSTRAINTS
        # Planets
        for i, planet in enumerate(self.planets):
            radius = self.planets[planet].radius + self.radius_sg
            H = np.eye(2) * (1 / radius)  # shape and dimension of the obstacle

            for k in range(self.params.K):
                dist_bar = self.problem_parameters["X_bar"][:2, k] - self.planets[planet].center
                C = -(H.T @ H @ dist_bar) / cvx.norm(H @ dist_bar, p="fro")
                r_first = 1 - cvx.norm(H @ dist_bar, p="fro") - C @ self.problem_parameters["X_bar"][:2, k]

                constraints.append(C @ self.variables["X"][:2, k] + r_first <= self.variables["nu_s"][i, k])

        # Satellites
        for i, _ in enumerate(self.satellites):
            for k in range(self.params.K):
                constraints.append(
                    self.problem_parameters["C_sat"][(i * 2) : ((i + 1) * 2), k] @ self.variables["X"][:2, k]
                    + self.problem_parameters["G_sat"][i, k] * self.variables["p"]
                    + self.problem_parameters["r_first_sat"][i, k]
                    <= self.variables["nu_s"][i + len(self.planets), k]
                )

        ################################################################################################################
        # LANDING CONSTRAINTS
        landing_points = slice(self.params.K - self.last_points, self.params.K - 1)

        if isinstance(self.goal, DockingTarget):
            # Constraints on the arms
            # Between -pi/2 and pi/2
            if (
                0 <= self.mod_2_pi(self.goal_state.as_ndarray()[2]) < np.pi / 2
                or 3 * np.pi / 2 < self.mod_2_pi(self.goal_state.as_ndarray()[2]) < 2 * np.pi
            ):
                constraints.append(
                    self.variables["X"][1, landing_points]
                    - self.tan_phi_goal * (self.variables["X"][0, landing_points] - self.B_dock[0])
                    - self.B_dock[1]
                    + self.variables["nu_constraints"][0, :]
                    >= 0  # If is dock is 0 there isn't constraint
                )
                constraints.append(
                    self.variables["X"][1, landing_points]
                    - self.tan_phi_goal * (self.variables["X"][0, landing_points] - self.C_dock[0])
                    - self.C_dock[1]
                    + self.variables["nu_constraints"][1, :]
                    <= 0
                )

            # Between pi/2 and 3*pi/2
            elif np.pi / 2 < self.mod_2_pi(self.goal_state.as_ndarray()[2]) < 3 * np.pi / 2:
                constraints.append(
                    self.variables["X"][1, landing_points]
                    - self.tan_phi_goal * (self.variables["X"][0, landing_points] - self.B_dock[0])
                    - self.B_dock[1]
                    + self.variables["nu_constraints"][0, :]
                    <= 0  # If is dock is 0 there isn't constraint
                )
                constraints.append(
                    self.variables["X"][1, landing_points]
                    - self.tan_phi_goal * (self.variables["X"][0, landing_points] - self.C_dock[0])
                    - self.C_dock[1]
                    + self.variables["nu_constraints"][1, :]
                    >= 0
                )

            # pi/2
            elif self.mod_2_pi(self.goal_state.as_ndarray()[2]) == np.pi / 2:
                constraints.append(
                    self.variables["X"][0, landing_points] - self.B_dock[0] + self.variables["nu_constraints"][0, :]
                    <= 0
                )
                constraints.append(
                    self.variables["X"][0, landing_points] - self.C_dock[0] + self.variables["nu_constraints"][1, :]
                    >= 0
                )

            # 3*pi/2
            else:
                constraints.append(
                    self.variables["X"][0, landing_points] - self.B_dock[0] + self.variables["nu_constraints"][0, :]
                    >= 0
                )
                constraints.append(
                    self.variables["X"][0, landing_points] - self.C_dock[0] + self.variables["nu_constraints"][1, :]
                    <= 0
                )

            # Constraint on the bottom
            A2B = self.B_dock - self.A2_dock
            goal_xy = self.goal_state.as_ndarray()[:2]
            A2goal = goal_xy - self.A2_dock
            dot_prod = np.dot(A2goal, A2B)
            lengh_A2B = np.dot(A2B, A2B)
            t = dot_prod / lengh_A2B
            A3 = self.A2_dock + t * A2B

            if 0 < self.mod_2_pi(self.goal_state.as_ndarray()[2]) < np.pi:
                constraints.append(
                    self.variables["X"][1, landing_points]
                    - self.tan_alpha_goal * (self.variables["X"][0, landing_points] - A3[0])
                    - A3[1]
                    + self.variables["nu_constraints"][2, :]
                    >= 0
                )
            elif np.pi < self.mod_2_pi(self.goal_state.as_ndarray()[2]) < 2 * np.pi:
                constraints.append(
                    self.variables["X"][1, landing_points]
                    - self.tan_alpha_goal * (self.variables["X"][0, landing_points] - A3[0])
                    - A3[1]
                    + self.variables["nu_constraints"][2, :]
                    <= 0
                )
            elif self.mod_2_pi(self.goal_state.as_ndarray()[2]) == 0:
                constraints.append(
                    self.variables["X"][1, landing_points] - A3[0] + self.variables["nu_constraints"][2, :] >= 0
                )
            else:
                constraints.append(
                    self.variables["X"][1, landing_points] - A3[0] + self.variables["nu_constraints"][2, :] <= 0
                )

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # TERMINAL COST
        # Define terminal cost penalized
        phi_lambda = self.params.weight_p * self.variables["p"] + self.params.lambda_nu * (
            cvx.norm(self.variables["nu_ic"], p=1)
            + cvx.norm(self.variables["nu_tc"], p=1)
            + cvx.norm(self.variables["nu_thrust_region"], p=1)
            + cvx.norm(self.variables["nu_constraints"], p=1)
            + cvx.norm(self.variables["nu_control"], p=1)
        )

        ################################################################################################################
        # CUMULATIVE COST
        # Distances between 2 points
        delta_x = self.variables["X"][0, 1:] - self.variables["X"][0, :-1]
        delta_y = self.variables["X"][1, 1:] - self.variables["X"][1, :-1]

        # Define gamma lambda
        gamma_lambda = [
            self.params.weight_distance * cvx.norm(cvx.vstack([delta_x[k], delta_y[k]]), p="fro")
            + self.params.weight_control * cvx.norm(self.variables["U"][:, k], p=1)
            + self.params.lambda_nu
            * (
                cvx.norm(self.variables["nu"][:, k], p=1)
                + cvx.norm(self.variables["nu_s"][:, k], p=1)
                + cvx.norm(self.variables["nu_limits_X"][:, k], p=1)
                + cvx.norm(self.variables["nu_limits_U"][:, k], p=1)
            )
            for k in range(self.params.K - 1)
        ] + [
            self.params.weight_control * cvx.norm(self.variables["U"][:, self.params.K - 1], p=1)
            + self.params.lambda_nu * cvx.norm(self.variables["nu_s"][:, self.params.K - 1], p=1)
        ]

        # Compute trapezoidal integration
        delta_t = 1.0 / (self.params.K - 1)
        gamma_sum = delta_t / 2 * np.sum(np.array(gamma_lambda[:-1]) + np.array(gamma_lambda[1:]))

        ################################################################################################################
        # OBJECTIVE FUNCTION
        objective = phi_lambda + gamma_sum

        return cvx.Minimize(objective)

    def _set_initial_parameters(self):
        # POPULATE PARAMETERS THAT DON'T CHANGE DURING THE ITERATION
        self.problem_parameters["init_state"].value = self.init_state.as_ndarray()
        self.problem_parameters["init_control"].value = self.init_control
        goal = self.goal_state.as_ndarray()
        if isinstance(self.goal, DockingTarget):
            goal[:2] = goal[:2] - (self.A2_dock - self.B_dock) / np.linalg.norm(self.A2_dock - self.B_dock) * 0.1
        self.problem_parameters["goal"].value = goal

        # INITIAL NU
        self.problem_parameters["nu_constraints_bar"].value = np.ones((3, self.last_points - 1))
        self.problem_parameters["nu_trust_region_bar"].value = 1

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        # Populate Problem Parameters
        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar
        self.problem_parameters["eta"].value = self.eta
        self.problem_parameters["X_bar"].value = self.X_bar
        self.problem_parameters["U_bar"].value = self.U_bar
        self.problem_parameters["p_bar"].value = self.p_bar

        # Populate Problem Parameters for satellites
        if self.satellites:
            C_sat = np.zeros((len(self.satellites) * 2, self.params.K))
            G_sat = np.zeros((len(self.satellites), self.params.K))
            r_first_sat = np.zeros((len(self.satellites), self.params.K))
            for i, satellite in enumerate(self.satellites):
                name_planet = satellite.split("/")[0]
                radius = self.satellites[satellite].radius + self.radius_sg
                for k in range(self.params.K):
                    input_values = (
                        self.X_bar[:2, k],
                        self.p_bar,
                        self.satellites[satellite].tau + mod_2_pi(self.init_time * self.satellites[satellite].omega),
                        self.satellites[satellite].orbit_r,
                        radius,
                        self.satellites[satellite].omega,
                        self.params.K,
                        k,
                        self.planets[name_planet].center,
                    )
                    C_sat[(i * 2) : ((i + 1) * 2), k] = self.C_sat_eval(*input_values).reshape(-1)
                    G_sat[i, k] = self.G_sat_eval(*input_values)
                    r_first_sat[i, k] = self.r_first_sat_eval(*input_values)

            self.problem_parameters["C_sat"].value = C_sat
            self.problem_parameters["G_sat"].value = G_sat
            self.problem_parameters["r_first_sat"].value = r_first_sat

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        delta_x = np.linalg.norm(self.variables["X"].value - self.X_bar, axis=0)
        delta_p = np.linalg.norm(self.variables["p"].value - self.p_bar)

        return bool(delta_p + np.max(delta_x) <= self.params.stop_crit)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        # Compute rho
        self._compute_rho()

        # Update trust region considering the computed rho
        if self.rho < self.params.rho_0:
            self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
        elif self.params.rho_0 <= self.rho < self.params.rho_1:
            self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
            self.problem_parameters["nu_constraints_bar"].value = self.variables["nu_constraints"].value
            self.problem_parameters["nu_trust_region_bar"].value = self.variables["nu_thrust_region"].value
        elif self.params.rho_1 <= self.rho < self.params.rho_2:
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
            self.problem_parameters["nu_constraints_bar"].value = self.variables["nu_constraints"].value
            self.problem_parameters["nu_trust_region_bar"].value = self.variables["nu_thrust_region"].value
        else:
            self.eta = min(self.params.max_tr_radius, self.params.beta * self.eta)
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
            self.problem_parameters["nu_constraints_bar"].value = self.variables["nu_constraints"].value
            self.problem_parameters["nu_trust_region_bar"].value = self.variables["nu_thrust_region"].value

    def _compute_rho(self):
        """
        Compute rho for trust region update.
        """
        # TERMINAL COST
        # Define terminal cost penalized bar
        phi_lambda_bar = self.params.weight_p * self.problem_parameters["p_bar"].value + self.params.lambda_nu * (
            np.linalg.norm(self.X_bar[:, 0] - self.problem_parameters["init_state"].value, ord=1)
            + np.linalg.norm(
                (self.X_bar[:6, -1] - self.problem_parameters["goal"].value),
                ord=1,
            )
            + np.max(self.problem_parameters["nu_trust_region_bar"].value, 0)
            + np.linalg.norm(self.problem_parameters["nu_constraints_bar"].value, ord=1)
            + np.linalg.norm(self.U_bar[:, 0] - self.init_control, ord=1)
            + np.linalg.norm(self.U_bar[:, -1], ord=1)
        )

        # Define terminal cost penalized opt
        phi_lambda_opt = self.params.weight_p * self.variables["p"].value + self.params.lambda_nu * (
            np.linalg.norm(self.variables["X"][:, 0].value - self.problem_parameters["init_state"].value, ord=1)
            + np.linalg.norm(
                (self.variables["X"][:6, -1].value - self.problem_parameters["goal"].value),
                ord=1,
            )
            + np.max(self.variables["nu_thrust_region"].value, 0)
            + np.linalg.norm(self.variables["nu_constraints"].value, ord=1)
            + np.linalg.norm(self.variables["U"][:, 0].value - self.init_control, ord=1)
            + np.linalg.norm(self.variables["U"][:, -1].value, ord=1)
        )

        ################################################################################################################
        # CUMULATIVE COST BAR
        # Define flow map
        flow_map_bar = self.integrator.integrate_nonlinear_piecewise(self.X_bar, self.U_bar, self.p_bar)

        # Define defects
        delta_bar = self.X_bar[:, 1:] - flow_map_bar[:, 1:]

        # Define gamma lambda bar
        delta_x = self.X_bar[0, 1:] - self.X_bar[0, :-1]
        delta_y = self.X_bar[1, 1:] - self.X_bar[1, :-1]
        distances = np.linalg.norm(np.vstack([delta_x, delta_y]), axis=0)

        obs_constr_bar = np.zeros((len(self.planets) + len(self.satellites), self.params.K))
        for i, planet in enumerate(self.planets):
            H = np.eye(2) * 1 / (self.planets[planet].radius + self.radius_sg)
            dist = self.X_bar[:2, :] - np.array(self.planets[planet].center)[:, np.newaxis]
            obs_constr_bar[i, :] = 1 - np.linalg.norm(H @ dist, axis=0)

        for i, satellite in enumerate(self.satellites):
            name_planet = satellite.split("/")[0]
            radius = self.satellites[satellite].radius + self.radius_sg
            for k in range(self.params.K):
                obs_constr_bar[i + len(self.planets), k] = self.s_sat_eval(
                    self.X_bar[:2, k],
                    self.p_bar,
                    self.satellites[satellite].tau + mod_2_pi(self.init_time * self.satellites[satellite].omega),
                    self.satellites[satellite].orbit_r,
                    radius,
                    self.satellites[satellite].omega,
                    self.params.K,
                    k,
                    self.planets[name_planet].center,
                )

        limit_constraints_bar = np.zeros((11, self.params.K))
        limit_constraints_bar[0, :] = -self.X_bar[6, :] + self.sp.delta_limits[0]
        limit_constraints_bar[1, :] = self.X_bar[6, :] - self.sp.delta_limits[1]
        limit_constraints_bar[2, :] = -self.X_bar[7, :] + self.sg.m
        limit_constraints_bar[3, :] = -self.U_bar[0, :] + self.sp.thrust_limits[0]
        limit_constraints_bar[4, :] = self.U_bar[0, :] - self.sp.thrust_limits[1]
        limit_constraints_bar[5, :] = -self.U_bar[1, :] + self.sp.ddelta_limits[0]
        limit_constraints_bar[6, :] = self.U_bar[1, :] - self.sp.ddelta_limits[1]
        limit_constraints_bar[7, :] = -self.X_bar[3, :] + self.sp.vx_limits[0]
        limit_constraints_bar[8, :] = self.X_bar[3, :] - self.sp.vx_limits[1]
        limit_constraints_bar[9, :] = -self.X_bar[4, :] + self.sp.vx_limits[0]
        limit_constraints_bar[10, :] = self.X_bar[4, :] - self.sp.vx_limits[1]

        gamma_lambda_bar = [
            self.params.weight_distance * distances[k]
            + self.params.weight_control * np.linalg.norm(self.U_bar[:, k], ord=1)
            + self.params.lambda_nu
            * (
                np.linalg.norm(delta_bar[:, k], ord=1)
                + np.linalg.norm(obs_constr_bar[:, k][obs_constr_bar[:, k] > 0], ord=1)
                + np.linalg.norm(limit_constraints_bar[:, k][limit_constraints_bar[:, k] > 0], ord=1)
            )
            for k in range(self.params.K - 1)
        ] + [
            self.params.weight_control * np.linalg.norm(self.U_bar[:, self.params.K - 1], ord=1)
            + self.params.lambda_nu
            * (
                np.linalg.norm(obs_constr_bar[:, self.params.K - 1][obs_constr_bar[:, self.params.K - 1] > 0], ord=1)
                + np.linalg.norm(
                    limit_constraints_bar[:, self.params.K - 1][limit_constraints_bar[:, self.params.K - 1] > 0], ord=1
                )
            )
        ]

        # Compute trapezoidal integration
        delta_t = 1.0 / (self.params.K - 1)
        gamma_bar = delta_t / 2 * np.sum((gamma_lambda_bar[:-1] + gamma_lambda_bar[1:]))

        ################################################################################################################
        # CUMULATIVE COST OPT
        # Define flow map opt
        flow_map_opt = self.integrator.integrate_nonlinear_piecewise(
            self.variables["X"].value, self.variables["U"].value, self.variables["p"].value
        )

        # Define defects opt
        delta_opt = self.variables["X"].value[:, 1:] - flow_map_opt[:, 1:]

        # Define gamma lambda opt
        delta_x = self.variables["X"].value[0, 1:] - self.variables["X"].value[0, :-1]
        delta_y = self.variables["X"].value[1, 1:] - self.variables["X"].value[1, :-1]
        distances = np.linalg.norm(np.vstack([delta_x, delta_y]), axis=0)

        obs_constr_opt = np.zeros((len(self.planets) + len(self.satellites), self.params.K))
        for i, planet in enumerate(self.planets):
            H = np.eye(2) * 1 / (self.planets[planet].radius + self.radius_sg)
            dist = self.variables["X"].value[:2, :] - np.array(self.planets[planet].center)[:, np.newaxis]
            obs_constr_opt[i, :] = 1 - np.linalg.norm(H @ dist, axis=0)

        for i, satellite in enumerate(self.satellites):
            name_planet = satellite.split("/")[0]
            radius = self.satellites[satellite].radius + self.radius_sg
            for k in range(self.params.K):
                obs_constr_opt[i + len(self.planets), k] = self.s_sat_eval(
                    self.variables["X"].value[:2, k],
                    self.variables["p"].value,
                    self.satellites[satellite].tau + mod_2_pi(self.init_time * self.satellites[satellite].omega),
                    self.satellites[satellite].orbit_r,
                    radius,
                    self.satellites[satellite].omega,
                    self.params.K,
                    k,
                    self.planets[name_planet].center,
                )

        limit_constraints_opt = np.zeros((11, self.params.K))
        limit_constraints_opt[0, :] = -self.variables["X"].value[6, :] + self.sp.delta_limits[0]
        limit_constraints_opt[1, :] = self.variables["X"].value[6, :] - self.sp.delta_limits[1]
        limit_constraints_opt[2, :] = -self.variables["X"].value[7, :] + self.sg.m
        limit_constraints_opt[3, :] = -self.variables["U"].value[0, :] + self.sp.thrust_limits[0]
        limit_constraints_opt[4, :] = self.variables["U"].value[0, :] - self.sp.thrust_limits[1]
        limit_constraints_opt[5, :] = -self.variables["U"].value[1, :] + self.sp.ddelta_limits[0]
        limit_constraints_opt[6, :] = self.variables["U"].value[1, :] - self.sp.ddelta_limits[1]
        limit_constraints_opt[7, :] = -self.variables["X"].value[3, :] + self.sp.vx_limits[0]
        limit_constraints_opt[8, :] = self.variables["X"].value[3, :] - self.sp.vx_limits[1]
        limit_constraints_opt[9, :] = -self.variables["X"].value[4, :] + self.sp.vx_limits[0]
        limit_constraints_opt[10, :] = self.variables["X"].value[4, :] - self.sp.vx_limits[1]

        gamma_lambda_opt = [
            self.params.weight_distance * distances[k]
            + self.params.weight_control * np.linalg.norm(self.variables["U"].value[:, k], ord=1)
            + self.params.lambda_nu
            * (
                np.linalg.norm(delta_opt[:, k], ord=1)
                + np.linalg.norm(obs_constr_opt[:, k][obs_constr_opt[:, k] > 0], ord=1)
                + np.linalg.norm(limit_constraints_opt[:, k][limit_constraints_bar[:, k] > 0], ord=1)
            )
            for k in range(self.params.K - 1)
        ] + [
            self.params.weight_control * np.linalg.norm(self.variables["U"].value[:, self.params.K - 1], ord=1)
            + self.params.lambda_nu
            * (
                np.linalg.norm(obs_constr_opt[:, self.params.K - 1][obs_constr_opt[:, self.params.K - 1] > 0], ord=1)
                + np.linalg.norm(
                    limit_constraints_opt[:, self.params.K - 1][limit_constraints_bar[:, self.params.K - 1] > 0], ord=1
                )
            )
        ]

        # Compute trapezoidal integration
        gamma_opt = delta_t / 2 * np.sum((gamma_lambda_opt[:-1] + gamma_lambda_opt[1:]))

        ################################################################################################################
        # COMPUTE RHO
        # Define cost function for bar
        self.cost_func_bar = phi_lambda_bar + gamma_bar

        # Define cost function for opt
        self.cost_func_opt = phi_lambda_opt + gamma_opt

        # Compute rho
        self.rho = (self.cost_func_bar[0] - self.cost_func_opt[0]) / (self.cost_func_bar[0] - self.error)

    def _sat_eval_func(self):
        self.s_sat_eval = self.sat_evaluator.s_sat()
        self.C_sat_eval = self.sat_evaluator.C_sat()
        self.G_sat_eval = self.sat_evaluator.G_sat()
        self.r_first_sat_eval = self.sat_evaluator.r_first_sat()

    def _sequence_from_array(self) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        # Sequence from an array
        # if not isinstance(self.goal, DockingTarget):
        # 1. Create the timestaps
        ts = (np.linspace(0, self.variables["p"].value[0], self.params.K) + self.init_time).tolist()

        # 2. Create the sequences for commands
        F = self.variables["U"].value[0, :]
        ddelta = self.variables["U"].value[1, :]
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]

        # 3. Create the sequences for states
        npstates = self.variables["X"].value.T
        states = [SpaceshipState(*v) for v in npstates]

        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        return mycmds, mystates

    def mod_2_pi(self, x: float) -> float:
        return x - 2 * np.pi * np.floor(x / (2 * np.pi))
