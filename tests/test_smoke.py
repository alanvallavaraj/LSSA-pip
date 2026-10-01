import numpy as np
from lssa import minimize_lssa


class Sphere:
    d = 3
    lo = -5.0
    hi = 5.0

    def eval(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))


def test_lssa_smoke():
    x, f, meta = minimize_lssa(
        Sphere(),
        max_fes=700,
        population=20,
        seed=7,
        return_metadata=True,
    )
    assert x.shape == (3,)
    assert np.isfinite(f)
    assert f >= 0.0
    assert 0 < meta["function_evaluations"] <= 700
