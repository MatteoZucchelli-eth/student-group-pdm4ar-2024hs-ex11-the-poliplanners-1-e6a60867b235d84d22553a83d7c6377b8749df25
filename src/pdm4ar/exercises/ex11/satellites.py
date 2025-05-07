import sympy as spy


class Satellite:
    def __init__(self):
        self.r = spy.Matrix(spy.symbols("x y", real=True))
        self.p = spy.Matrix([spy.symbols("t_f", positive=True, real=True)])
        self.tau_0 = spy.symbols("tau_0", real=True)
        self.r_orb = spy.symbols("r_orb", positive=True, real=True)
        self.r_sat = spy.symbols("r_sat", positive=True, real=True)
        self.H = spy.Matrix([[1, 0], [0, 1]]) / self.r_sat
        self.omega = spy.symbols("omega", real=True)
        self.K = spy.symbols("K", positive=True, real=True)
        self.i = spy.symbols("i", positive=True, real=True)
        self.x_plan = spy.Matrix(spy.symbols("x_plan y_plan", real=True))

    def center_func(self):
        return (
            spy.Matrix(
                [
                    spy.cos(self.tau_0 + self.omega * self.i * self.p[0] / (self.K - 1)),
                    spy.sin(self.tau_0 + self.omega * self.i * self.p[0] / (self.K - 1)),
                ]
            )
            * self.r_orb
            + self.x_plan
        )

    def s_func(self):
        x_sat = self.center_func()
        arg = self.H * (self.r - x_sat)
        return spy.Matrix([1 - arg.norm()])

    def s_sat(self):
        return spy.lambdify(
            (self.r, self.p, self.tau_0, self.r_orb, self.r_sat, self.omega, self.K, self.i, self.x_plan),
            self.s_func(),
            "numpy",
        )

    def C_func(self):
        center = self.center_func()
        return -(self.H * self.H.T * (self.r - center)) / (self.r - center).norm()

    def C_sat(self):
        return spy.lambdify(
            (self.r, self.p, self.tau_0, self.r_orb, self.r_sat, self.omega, self.K, self.i, self.x_plan),
            self.C_func(),
            "numpy",
        )

    def G_func(self):
        s_satel = self.s_func()
        return s_satel.jacobian(self.p)

    def G_sat(self):
        return spy.lambdify(
            (self.r, self.p, self.tau_0, self.r_orb, self.r_sat, self.omega, self.K, self.i, self.x_plan),
            self.G_func(),
            "numpy",
        )

    def r_first_sat(self):
        s = self.s_func()
        C = self.C_func()
        G = self.G_func()
        r_first = s - C.T @ self.r - G * self.p
        return spy.lambdify(
            (self.r, self.p, self.tau_0, self.r_orb, self.r_sat, self.omega, self.K, self.i, self.x_plan),
            r_first,
            "numpy",
        )


class Planet:
    def __init__(self):
        self.r = spy.Matrix(spy.symbols("x y", real=True))
        self.r_plan = spy.symbols("r_sat", positive=True, real=True)
        self.center = spy.Matrix(spy.symbols("x_plan y_plan", real=True))
        self.H = spy.Matrix([[1, 0], [0, 1]]) / self.r_plan
        self.K = spy.symbols("K", positive=True, real=True)

    def s_func(self):
        arg = self.H * (self.r - self.center)
        return spy.Matrix([1 - arg.norm()])

    def s_plan(self):
        return spy.lambdify((self.r, self.r_plan, self.center), self.s_func(), "numpy")

    def C_func(self):
        return -(self.H * self.H.T * (self.r - self.center)) / (self.r - self.center).norm()

    def C_plan(self):
        return spy.lambdify((self.r, self.r_plan, self.center), self.C_func(), "numpy")

    def r_first_plan(self):
        s = self.s_func()
        C = self.C_func()
        r_first = s - C.T @ self.r
        return spy.lambdify(
            (self.r, self.r_plan, self.center),
            r_first,
            "numpy",
        )
