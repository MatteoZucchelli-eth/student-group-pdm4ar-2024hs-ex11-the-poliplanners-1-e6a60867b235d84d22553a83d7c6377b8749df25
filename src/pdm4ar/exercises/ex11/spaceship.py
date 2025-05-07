import sympy as spy

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi delta m", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        0x 1y 2psi 3vx 4vy 5dpsi 6delta 7m
        """
        # Dynamics
        f = spy.zeros(self.n_x, 1)

        # TODO: Define the dynamics f(x, u, p) of the spaceship
        f[0] = self.x[3] * spy.cos(self.x[2]) - self.x[4] * spy.sin(self.x[2])  # xdot = vx*cos(psi) - vy*sin(psi)
        f[1] = self.x[3] * spy.sin(self.x[2]) + self.x[4] * spy.cos(self.x[2])  # ydot = vx*sin(psi) + vy*cos(psi)
        f[2] = self.x[5]  # psidot = dpsi
        f[3] = (
            spy.cos(self.x[6]) * self.u[0] / self.x[7] + self.x[4] * self.x[5]
        )  # vxdot = cos(delta)*thrust/m + vy*dpsi
        f[4] = (
            spy.sin(self.x[6]) * self.u[0] / self.x[7] - self.x[3] * self.x[5]
        )  # vydot = sin(delta)*thrust/m - vx*dpsi
        f[5] = -self.sg.l_r * spy.sin(self.x[6]) * self.u[0] / self.sg.Iz  # dpsidot = -l_r*sin(delta)*thrust/I_z
        f[6] = self.u[1]  # deltadot = ddelta
        f[7] = -self.sp.C_T * self.u[0]  # mdot = -C_T*thrust

        f *= self.p[0]

        # jacobian() comes from SymPy
        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        # lambdify(sym vars, lambda func, 'numpy') is used to convert symbolic SymPy expressions to lambda func that can be evaluated by numpy functions w f(np.array(...)) (NO *)
        # lambda function = anonymous (symbolic) function: no name (no def keyword (exp, sin,...)), body is a single expression
        # example:              x, y = spy.symbols('x y') or X = spy.Matrix(spy.symbols("x y")), x = X[0], y = X[1]
        #                       f = spy.Lambda([x, y], exp(x)*y)
        #                       NOTE: to evaluate f, use f(1, 2) or f(*np.array([1, 2])), where * unpacks
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
