import scipy.optimize as sopt
import firefly as fa
import numpy as np


# Probar la función de Rosenbrock
print("Rosenbrock")
res = sopt.minimize(sopt.rosen, np.linspace(-5.0, 10.0, num=2))
print(f"Python: {res.x}")
# Implementar el FA
kwargs = {
    "func": sopt.rosen,
    "dim": 2,
    "tam_pob": 30,
    "inf": -5.0,
    "sup": 10.0,
    "alpha": 0.9,
    "beta": 0.2,
    "gamma": 1.0,
}
fa_optim = fa.FAOpt(**kwargs)
# fa_optim = fa.FAOpt(sopt.rosen, 16, 40, 0.9, 0.4, 1.0, -5.0, 10.0)
res = fa_optim.optimizar(25)
print("Firefly algorithm: {0}".format(res))

# Implementar la función Sphere, con el origen movido a 1
def sphere(x):
    return sum((x - 1.0) ** 2)


print("Sphere")
# Verificar el resultado con Scipy
res = sopt.minimize(sphere, np.linspace(-5.12, 5.12, num=256))
print(f"Python: {res.x}")
# Implementar el FA
fa_optim = fa.FAOpt(sphere, 256, 40, -5.12, 5.12, 0.9, 0.2, 1.0)
res = fa_optim.optimizar(100)
print("Firefly algorithm: {0}".format(res))

#  Implementar la función de Beale
def beale(x):
    term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
    term2 = (2.25 - x[0] + x[0] * (x[1] ** 2)) ** 2
    term3 = (2.625 - x[0] + x[0] * (x[1] ** 3)) ** 2

    return term1 + term2 + term3


print("Beale")
# Verificar el resultado con Scipy
res = sopt.basinhopping(beale, [-4.5, 4.5])
print(f"Python: {res.x}")
# Implementar el FA
fa_optim = fa.FAOpt(beale, 2, 25, -4.5, 4.5, 0.9, 0.2, 1.0)
res = fa_optim.optimizar(35)
print("Firefly algorithm: {0}".format(res))

#  Implementar la función de Goldstein-Price
def gp(x):
    term1 = (x[0] + x[1] + 1.0) ** 2
    term2 = (
        19.0
        - 14.0 * x[0]
        + 3.0 * x[0] ** 2
        - 14.0 * x[1]
        + 6.0 * x[0] * x[1]
        + 3.0 * x[1] ** 2
    )
    term3 = (2.0 * x[0] - 3.0 * x[1]) ** 2
    term4 = (
        18.0
        - 32.0 * x[0]
        + 12.0 * x[0] ** 2
        + 48.0 * x[1]
        - 36.0 * x[0] * x[1]
        + 27.0 * x[1] ** 2
    )

    return (1.0 + term1 * term2) * (30.0 + term3 * term4)


print("Goldstein-Price")
# Verificar el resultado con Scipy
res = sopt.basinhopping(gp, [-2.0, 2.0])
print(f"Python: {res.x}")
# Implementar el FA
fa_optim = fa.FAOpt(gp, 2, 25, -2.0, 2.0, 0.9, 0.2, 1.0)
res, brillo = fa_optim.optimizar(40, optim=True)
print("Firefly algorithm: {0}\n{1}".format(res, brillo))

