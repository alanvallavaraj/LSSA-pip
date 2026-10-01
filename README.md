# LSSA

**LSSA (Lost Sheep Search Algorithm)** is an adaptive continuous black-box optimisation algorithm.

This package provides the frozen research implementation used for out-of-sample validation. Development candidates such as the B20 branch are kept separate from the stable package release.

## Installation

```bash
pip install lssaopt
```

The PyPI distribution is named `lssaopt`; the Python import remains `lssa`.

## Quick start

```python
import numpy as np
from lssa import minimize_lssa

class Sphere:
    d = 5
    lo = -5.0
    hi = 5.0

    def eval(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

x_best, f_best = minimize_lssa(Sphere(), max_fes=2000, seed=0)
print(f_best, x_best)
```

The problem object must provide:

- `d`: dimensionality
- `lo`: scalar lower bound
- `hi`: scalar upper bound
- `eval(x)`: objective function returning a scalar

## API

```python
minimize_lssa(problem, max_fes=4000, population=30, seed=0, return_metadata=False)
```

LSSA performs minimisation. For reproducibility, set `seed` explicitly.

## Status

Initial PyPI release: `0.1.0`.
