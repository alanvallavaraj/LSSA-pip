
"""
Lost Sheep Search Algorithm (LSSA)
Frozen research implementation for out-of-sample validation.

Important:
- One unpublished algorithm: LSSA.
- One Shepherd with four adaptive modes:
    1) Envelope Shepherd Mode
    2) Spectral Shepherd Mode
    3) Structural-Axis Shepherd Mode
    4) Generic Shepherd Mode
- Mode selection depends only on observed objective behaviour.
- No benchmark name, known optimum, rotation matrix, or function identity is used.
"""

from __future__ import annotations
import numpy as np
from itertools import product
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc
from sklearn.linear_model import Ridge


class BudgetObjective:
    """Strictly counts real objective evaluations and stores only verified best points."""
    def __init__(self, problem, max_fes: int):
        self.problem = problem
        self.max_fes = int(max_fes)
        self.n = 0
        self.best = np.inf
        self.bestx = None

    def __call__(self, x):
        # After exhaustion, no further real objective evaluation occurs.
        # The algorithm always returns self.bestx, which is verified.
        if self.n >= self.max_fes:
            return self.best
        x = np.clip(np.asarray(x, dtype=float), self.problem.lo, self.problem.hi)
        f = float(self.problem.eval(x))
        self.n += 1
        if f < self.best:
            self.best = f
            self.bestx = x.copy()
        return f


def _local_refine(obj, p, x0, alloc, method="L-BFGS-B"):
    alloc = int(max(0, alloc))
    if alloc <= 0 or obj.n >= obj.max_fes:
        return
    limit = min(obj.max_fes, obj.n + alloc)

    class Wrapped:
        def __call__(self, x):
            if obj.n >= limit:
                return obj.best
            return obj(x)

    if method == "L-BFGS-B":
        minimize(
            Wrapped(), np.asarray(x0).copy(),
            method="L-BFGS-B",
            bounds=[(p.lo, p.hi)] * p.d,
            options={
                "maxfun": alloc,
                "maxiter": max(5, alloc // (p.d + 1)),
                "ftol": 1e-14,
                "gtol": 1e-10,
                "maxls": 20,
            },
        )
    elif method == "Nelder-Mead":
        minimize(
            Wrapped(), np.asarray(x0).copy(),
            method="Nelder-Mead",
            options={
                "maxfev": alloc,
                "maxiter": alloc,
                "adaptive": True,
                "xatol": 1e-10,
                "fatol": 1e-12,
            },
        )
    elif method == "Powell":
        minimize(
            Wrapped(), np.asarray(x0).copy(),
            method="Powell",
            bounds=[(p.lo, p.hi)] * p.d,
            options={
                "maxfev": alloc,
                "maxiter": alloc,
                "xtol": 1e-12,
                "ftol": 1e-14,
            },
        )
    else:
        raise ValueError(f"Unsupported local method: {method}")


def _feasible_bounds(x, u, lo, hi):
    lower, upper = [], []
    for xi, ui in zip(x, u):
        if abs(ui) < 1e-14:
            continue
        a = (lo - xi) / ui
        b = (hi - xi) / ui
        lower.append(min(a, b))
        upper.append(max(a, b))
    if not lower:
        return 0.0, 0.0
    return max(lower), min(upper)


def _hessian_estimate(obj, p, x, hfrac=1e-4):
    d = p.d
    h = max(1e-8, hfrac * (p.hi - p.lo))
    f0 = obj(x)
    H = np.zeros((d, d))
    for i in range(d):
        if obj.n + 2 > obj.max_fes:
            break
        ei = np.zeros(d); ei[i] = h
        fp = obj(np.clip(x + ei, p.lo, p.hi))
        fm = obj(np.clip(x - ei, p.lo, p.hi))
        H[i, i] = (fp - 2*f0 + fm) / (h*h)

    for i in range(d):
        for j in range(i+1, d):
            if obj.n + 4 > obj.max_fes:
                break
            ei = np.zeros(d); ej = np.zeros(d)
            ei[i] = h; ej[j] = h
            fpp = obj(np.clip(x + ei + ej, p.lo, p.hi))
            fpm = obj(np.clip(x + ei - ej, p.lo, p.hi))
            fmp = obj(np.clip(x - ei + ej, p.lo, p.hi))
            fmm = obj(np.clip(x - ei - ej, p.lo, p.hi))
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4*h*h)
    return (H + H.T) / 2


# -------------------------------------------------------------------
# Full harmonic Spectral Shepherd Mode
# -------------------------------------------------------------------

def _spectral_energy_grad(X, y, w):
    n = len(y)
    phase = np.exp(-1j * 2*np.pi * (X @ w))
    c = np.dot(y, phase)
    score = (c.real*c.real + c.imag*c.imag) / (n*n)
    dc = (-1j * 2*np.pi) * (X.T @ (y * phase))
    grad = 2*np.real(np.conj(c) * dc) / (n*n)
    return float(score), np.asarray(grad, float)


def _sphere_spectral_peaks(X, y, d, rhos=(.8,1.0,1.2,1.4),
                           starts_per=14, seed=0):
    rng = np.random.default_rng(seed)
    peaks = []
    for rho in rhos:
        for _ in range(starts_per):
            u = rng.normal(size=d)
            u /= max(np.linalg.norm(u), 1e-12)
            w0 = rho * u

            def fun(w):
                sc, gr = _spectral_energy_grad(X, y, w)
                return -sc, -gr

            cons = {
                "type": "eq",
                "fun": lambda w, rho=rho: np.dot(w, w) - rho*rho,
                "jac": lambda w: 2*w,
            }
            try:
                res = minimize(
                    fun, w0, jac=True, method="SLSQP",
                    constraints=[cons],
                    options={"maxiter": 90, "ftol": 1e-10, "disp": False},
                )
                w = np.asarray(res.x, float)
                sc, _ = _spectral_energy_grad(X, y, w)
                if np.isfinite(sc) and np.linalg.norm(w) > 1e-9:
                    peaks.append((sc, w.copy(), float(rho)))
            except Exception:
                pass
    peaks.sort(key=lambda z: z[0], reverse=True)
    return peaks


def _harmonic_design_and_grad(W, X, y, harmonics, ridge=1e-7):
    n, d = X.shape
    cols = [np.ones(n)]
    cache = []
    for i in range(d):
        cache_i = []
        for h in harmonics:
            theta = 2*np.pi*h*(X @ W[i])
            c = np.cos(theta)
            s = np.sin(theta)
            cols.extend([c, s])
            cache_i.append((h, theta, c, s))
        cache.append(cache_i)

    Phi = np.column_stack(cols)
    G = Phi.T @ Phi + ridge*np.eye(Phi.shape[1])
    beta = np.linalg.solve(G, Phi.T @ y)
    pred = Phi @ beta
    resid = y - pred
    mse = float(np.mean(resid*resid))

    grad = np.zeros_like(W)
    col = 1
    for i in range(d):
        S = np.zeros(n)
        for h, theta, c, s in cache[i]:
            bc = beta[col]; bs = beta[col+1]
            S += 2*np.pi*h*(-bc*s + bs*c)
            col += 2
        grad[i] = -2.0/n * (X.T @ (resid*S))
    return mse, grad, beta, pred


def _fit_frequency_matrix(W0, X, fvals, harmonics, maxiter=90):
    d = W0.shape[0]

    def fun(flat):
        W = flat.reshape(d, d)
        mse, gr, _, _ = _harmonic_design_and_grad(W, X, fvals, harmonics)
        norms = np.linalg.norm(W, axis=1) + 1e-12
        U = W / norms[:, None]
        O = U @ U.T - np.eye(d)
        reg = 5e-5*np.sum(O*O) + 5e-5*np.sum((norms-1.0)**2)
        greg = 1e-4*(norms-1.0)[:, None]*(W/norms[:, None])
        return mse + reg, (gr + greg).ravel()

    res = minimize(
        fun, W0.ravel(), jac=True, method="L-BFGS-B",
        bounds=[(-1.5, 1.5)] * (d*d),
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6, "maxls": 25},
    )
    W = res.x.reshape(d, d)
    _, _, beta, pred = _harmonic_design_and_grad(W, X, fvals, harmonics)
    r2 = 1 - np.sum((fvals-pred)**2) / (np.sum((fvals-fvals.mean())**2) + 1e-300)
    return W, beta, pred, float(r2)


def _infer_harmonic_family(W0, X, fvals):
    fits = []
    for b in (2, 3, 4, 5):
        harmonics = tuple(int(b**k) for k in range(5))
        W, _, _, r2 = _fit_frequency_matrix(W0, X, fvals, harmonics, maxiter=45)
        fits.append((r2, b, W))
    fits.sort(key=lambda z: z[0], reverse=True)
    return fits[0]


def _spectral_shepherd_mode(obj, p, seed):
    d = p.d
    span = p.hi - p.lo
    remaining = obj.max_fes - obj.n
    ns = int(min(3072, max(1024, remaining - 420)))

    m = int(np.ceil(np.log2(ns)))
    sob = qmc.Sobol(d=d, scramble=True, seed=seed+5555)
    X = p.lo + span*sob.random_base2(m)[:ns]
    F = np.array([obj(x) for x in X], float)
    y = F - F.mean()

    peaks = _sphere_spectral_peaks(
        X, y, d, rhos=(.8,1.0,1.2,1.4), starts_per=14, seed=seed+22
    )

    selected = []
    for sc, w, rho in peaks:
        u = w / max(np.linalg.norm(w), 1e-12)
        if not selected or all(
            abs(np.dot(u, z[1]/np.linalg.norm(z[1]))) < .82
            for z in selected
        ):
            selected.append((sc, w.copy(), rho))
        if len(selected) >= d:
            break

    if len(selected) < d:
        if obj.n < obj.max_fes:
            _local_refine(obj, p, obj.bestx, obj.max_fes-obj.n, method="Nelder-Mead")
        return {"complete": False}

    W0 = np.vstack([z[1] for z in selected])
    short_r2, bhat, Wshort = _infer_harmonic_family(W0, X, F)

    harmonics = tuple(int(bhat**k) for k in range(7))
    W, beta, pred, model_r2 = _fit_frequency_matrix(
        Wshort, X, F, harmonics, maxiter=120
    )

    grid = np.linspace(0, 1, 6001, endpoint=False)
    phase = []
    col = 1
    for i in range(d):
        vals = np.zeros_like(grid)
        for h in harmonics:
            bc = beta[col]; bs = beta[col+1]
            col += 2
            vals += bc*np.cos(2*np.pi*h*grid) + bs*np.sin(2*np.pi*h*grid)
        phase.append(float(grid[int(np.argmin(vals))]))
    phase = np.asarray(phase)

    ib = int(np.argmin(F))
    xb = X[ib]
    k0 = np.round(W @ xb - phase).astype(int)

    if d <= 12:
        deltas = np.array(list(product([-1,0,1], repeat=d)), dtype=int)
    else:
        rng = np.random.default_rng(seed+808)
        deltas = rng.integers(-1, 2, size=(50000, d))

    targets = phase[None, :] + k0[None, :] + deltas
    XC = targets @ np.linalg.pinv(W).T
    feasible = np.all((XC >= p.lo) & (XC <= p.hi), axis=1)
    XF = XC[feasible]

    if len(XF):
        order = np.argsort(np.linalg.norm(XF-xb, axis=1))[:min(28, len(XF))]
        for j in order:
            if obj.n >= obj.max_fes:
                break
            obj(XF[j])

    if obj.n < obj.max_fes:
        _local_refine(obj, p, obj.bestx, obj.max_fes-obj.n, method="Nelder-Mead")

    return {
        "complete": True,
        "harmonic_base": int(bhat),
        "model_r2": float(model_r2),
        "short_r2": float(short_r2),
    }


# -------------------------------------------------------------------
# Structural-Axis Shepherd Mode
# -------------------------------------------------------------------

def _structural_conservative(obj, p, seed):
    rng = np.random.default_rng(seed+77)
    d = p.d
    span = p.hi-p.lo

    sob = qmc.Sobol(d=d, scramble=True, seed=seed+5555)
    X = p.lo + span*sob.random_base2(8)
    F = np.array([obj(x) for x in X])
    base = X[np.argmin(F)].copy()

    candidates = []
    for j in range(3):
        u = rng.normal(size=d)
        u /= max(np.linalg.norm(u), 1e-12)
        x = np.clip(base + (.10+.07*j)*span*np.sqrt(d)*u, p.lo, p.hi)
        H = _hessian_estimate(obj, p, x, 1e-4)
        vals, V = np.linalg.eigh(H)
        sv = np.sort(vals)
        sep = np.median(np.abs(np.diff(sv))) / (np.std(vals)+1e-12)
        candidates.append((sep, vals, V))

    _, vals, V = max(candidates, key=lambda z:z[0])
    V = V[:, np.argsort(np.abs(vals))[::-1]]

    center = obj.bestx.copy()
    bestf = obj.best

    for _ in range(2):
        moved = False
        for k in range(d):
            if obj.n >= obj.max_fes-100:
                break
            u = V[:, k]
            tl, th = _feasible_bounds(center, u, p.lo, p.hi)
            if th <= tl:
                continue
            ts = np.linspace(tl, th, 45)
            fs = []
            for t in ts:
                if obj.n >= obj.max_fes-100:
                    break
                fs.append(obj(np.clip(center+t*u, p.lo, p.hi)))
            if len(fs) < 5:
                break
            q = int(np.argmin(fs))
            fq = float(fs[q]); tq = float(ts[q])

            if 0 < q < len(fs)-1 and obj.n < obj.max_fes-70:
                limit = min(obj.max_fes, obj.n+20)
                holder = [fq, tq]

                def phi(t):
                    if obj.n >= limit:
                        return holder[0]
                    f = obj(np.clip(center+t*u, p.lo, p.hi))
                    if f < holder[0]:
                        holder[0] = f; holder[1] = t
                    return f

                minimize_scalar(
                    phi, bounds=(ts[q-1], ts[q+1]), method="bounded",
                    options={"maxiter":20, "xatol":1e-12},
                )
                fq, tq = holder

            if fq < bestf:
                center = np.clip(center+tq*u, p.lo, p.hi)
                bestf = fq
                moved = True
        if not moved:
            break

    if obj.n < obj.max_fes:
        _local_refine(obj, p, obj.bestx, obj.max_fes-obj.n)
    return "learned-conservative"


def _structural_native_or_conservative(obj, p, seed):
    rng = np.random.default_rng(seed+77)
    d = p.d
    span = p.hi-p.lo

    sob = qmc.Sobol(d=d, scramble=True, seed=seed+5555)
    X = p.lo + span*sob.random_base2(8)
    F = np.array([obj(x) for x in X])
    base = X[np.argmin(F)].copy()

    Hs, candidates = [], []
    for j in range(3):
        u = rng.normal(size=d)
        u /= max(np.linalg.norm(u), 1e-12)
        x = np.clip(base + (.10+.07*j)*span*np.sqrt(d)*u, p.lo, p.hi)
        H = _hessian_estimate(obj, p, x, 1e-4)
        Hs.append(H)
        vals, V = np.linalg.eigh(H)
        sv = np.sort(vals)
        sep = np.median(np.abs(np.diff(sv))) / (np.std(vals)+1e-12)
        candidates.append((sep, vals, V))

    ratios = []
    for H in Hs:
        off = H - np.diag(np.diag(H))
        ratios.append(np.linalg.norm(off)/(np.linalg.norm(H)+1e-300))
    native = max(ratios) < 1e-5

    if native:
        V = np.eye(d)
        grid = 121
        maxiter = 28
        subtype = "native-dense"
    else:
        _, vals, V = max(candidates, key=lambda z:z[0])
        V = V[:, np.argsort(np.abs(vals))[::-1]]
        grid = 45
        maxiter = 20
        subtype = "learned-conservative"

    center = obj.bestx.copy()
    bestf = obj.best

    for _ in range(2):
        moved = False
        for k in range(d):
            if obj.n >= obj.max_fes-100:
                break
            u = V[:, k]
            tl, th = _feasible_bounds(center, u, p.lo, p.hi)
            ts = np.linspace(tl, th, grid)
            fs = []
            for t in ts:
                if obj.n >= obj.max_fes-100:
                    break
                fs.append(obj(np.clip(center+t*u, p.lo, p.hi)))
            if len(fs) < 5:
                break

            q = int(np.argmin(fs))
            fq = float(fs[q]); tq = float(ts[q])

            if 0 < q < len(fs)-1 and obj.n < obj.max_fes-70:
                limit = min(obj.max_fes, obj.n+maxiter)
                holder = [fq, tq]

                def phi(t):
                    if obj.n >= limit:
                        return holder[0]
                    f = obj(np.clip(center+t*u, p.lo, p.hi))
                    if f < holder[0]:
                        holder[0] = f; holder[1] = t
                    return f

                minimize_scalar(
                    phi, bounds=(ts[q-1], ts[q+1]), method="bounded",
                    options={"maxiter":maxiter, "xatol":1e-12},
                )
                fq, tq = holder

            if fq < bestf:
                center = np.clip(center+tq*u, p.lo, p.hi)
                bestf = fq
                moved = True
        if not moved:
            break

    if obj.n < obj.max_fes:
        _local_refine(obj, p, obj.bestx, obj.max_fes-obj.n)

    return subtype, float(max(ratios))


def _structural_periodic_stabilized(obj, p, seed):
    rng = np.random.default_rng(seed+77)
    d = p.d
    span = p.hi-p.lo

    sob = qmc.Sobol(d=d, scramble=True, seed=seed+5555)
    X = p.lo + span*sob.random_base2(8)
    F = np.array([obj(x) for x in X])
    base = X[np.argmin(F)]

    candidates = []
    for j in range(4):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        x = np.clip(base + (.08+.05*j)*span*np.sqrt(d)*u, p.lo, p.hi)
        H = _hessian_estimate(obj, p, x, 1e-4)
        vals, V = np.linalg.eigh(H)
        sv = np.sort(vals)
        sep = np.median(np.abs(np.diff(sv))) / (np.std(vals)+1e-12)
        candidates.append((sep, vals, V))

    candidates = sorted(candidates, key=lambda z:z[0], reverse=True)
    center = obj.bestx.copy()
    bestf = obj.best

    # Stabilization: test the two strongest independent curvature hypotheses.
    for pass_idx in range(min(2, len(candidates))):
        _, vals, V = candidates[pass_idx]
        V = V[:, np.argsort(np.abs(vals))[::-1]]
        local_center = center.copy()
        local_best = bestf
        passes = 2 if pass_idx == 0 else 1

        for _ in range(passes):
            moved = False
            for k in range(d):
                if obj.n >= obj.max_fes-100:
                    break
                u = V[:, k]
                tl, th = _feasible_bounds(local_center, u, p.lo, p.hi)
                ts = np.linspace(tl, th, 121)
                fs = []
                for t in ts:
                    if obj.n >= obj.max_fes-100:
                        break
                    fs.append(obj(np.clip(local_center+t*u, p.lo, p.hi)))
                if len(fs) < 5:
                    break
                q = int(np.argmin(fs))
                fq = float(fs[q]); tq = float(ts[q])

                if 0 < q < len(fs)-1 and obj.n < obj.max_fes-70:
                    limit = min(obj.max_fes, obj.n+18)
                    holder = [fq, tq]

                    def phi(t):
                        if obj.n >= limit:
                            return holder[0]
                        f = obj(np.clip(local_center+t*u, p.lo, p.hi))
                        if f < holder[0]:
                            holder[0] = f; holder[1] = t
                        return f

                    minimize_scalar(
                        phi, bounds=(ts[q-1], ts[q+1]), method="bounded",
                        options={"maxiter":18, "xatol":1e-10},
                    )
                    fq, tq = holder

                if fq < local_best:
                    local_center = np.clip(local_center+tq*u, p.lo, p.hi)
                    local_best = fq
                    moved = True
            if not moved:
                break

        if local_best < bestf:
            bestf = local_best
            center = local_center.copy()

    if obj.n < obj.max_fes:
        _local_refine(obj, p, obj.bestx, obj.max_fes-obj.n)
    return "periodic-stabilized"


# -------------------------------------------------------------------
# Symmetry-centre behaviour inside Envelope Shepherd Mode
# -------------------------------------------------------------------

def _symmetry_envelope_mode(obj, p, seed, Xdiag, Fdiag, m_pairs=4):
    rng = np.random.default_rng(seed)
    d = p.d
    span = p.hi-p.lo

    E = Xdiag[np.argsort(Fdiag)[:len(Xdiag)//2]]
    c0 = np.median(E, axis=0)

    radius = .12*span
    dirs = []
    for _ in range(m_pairs):
        u = rng.normal(size=d)
        u /= max(np.linalg.norm(u), 1e-12)
        dirs.append(radius*u)

    margin = radius
    clo = np.full(d, p.lo+margin)
    chi = np.full(d, p.hi-margin)
    c0 = np.clip(c0, clo, chi)

    def symmetry_loss(c):
        vals = []
        for v in dirs:
            if obj.n + 2 > obj.max_fes:
                break
            fp = obj(c+v)
            fm = obj(c-v)
            vals.append(((fp-fm)/(1+abs(fp)+abs(fm)))**2)
        return float(np.mean(vals)) if vals else 1e9

    reserve = 180
    maxcalls = max(10, (obj.max_fes-obj.n-reserve)//(2*m_pairs))
    res = minimize(
        symmetry_loss, c0, method="Powell",
        bounds=list(zip(clo, chi)),
        options={
            "maxfev": maxcalls,
            "maxiter": maxcalls,
            "xtol": 1e-7,
            "ftol": 1e-10,
        },
    )

    if obj.n < obj.max_fes:
        obj(np.clip(res.x, p.lo, p.hi))
    if obj.n < obj.max_fes:
        _local_refine(obj, p, obj.bestx, obj.max_fes-obj.n)

    return float(res.fun)


# -------------------------------------------------------------------
# Generic Shepherd Mode
# -------------------------------------------------------------------

def _generic_shepherd_mode(obj, p, seed, X, F):
    rng = np.random.default_rng(seed)
    d = p.d
    muF = .5
    muCR = .8
    archive = []

    while obj.n < int(.72*obj.max_fes):
        n = len(X)
        rank = np.argsort(F)
        pnum = max(2, int(np.ceil(.25*n)))

        i = int(rng.integers(n))
        pbest = int(rng.choice(rank[:pnum]))
        candidates = [j for j in range(n) if j not in (i, pbest)]
        r1 = int(rng.choice(candidates))
        pool = [X[j] for j in range(n) if j not in (i,pbest,r1)] + archive
        r2 = np.asarray(pool[int(rng.integers(len(pool)))])

        Fi = np.clip(muF + .1*rng.standard_cauchy(), .05, 1)
        CR = np.clip(rng.normal(muCR, .1), 0, 1)

        V = np.clip(
            X[i] + Fi*(X[pbest]-X[i]) + Fi*(X[r1]-r2),
            p.lo, p.hi
        )

        mask = rng.random(d) < CR
        mask[int(rng.integers(d))] = True
        T = np.where(mask, V, X[i])
        ft = obj(T)

        if ft <= F[i]:
            archive.append(X[i].copy())
            archive = archive[-n:]
            X[i], F[i] = T, ft
            muF = .98*muF + .02*Fi
            muCR = .98*muCR + .02*CR

    if obj.n < obj.max_fes:
        _local_refine(obj, p, obj.bestx, obj.max_fes-obj.n, method="Powell")


# -------------------------------------------------------------------
# Public LSSA entry point
# -------------------------------------------------------------------

def minimize_lssa(problem, max_fes=4000, population=30, seed=0,
                  return_metadata=False):
    """
    Run LSSA on a minimisation problem.

    Required problem interface:
        problem.d   : int
        problem.lo  : scalar lower bound
        problem.hi  : scalar upper bound
        problem.eval(x) -> scalar objective

    Returns:
        (best_x, best_f) or (best_x, best_f, metadata)
    """
    obj = BudgetObjective(problem, max_fes)
    d = problem.d
    span = problem.hi - problem.lo

    # ---------------- Shared observation stage ----------------
    sob = qmc.Sobol(d=d, scramble=True, seed=seed+1234)
    X64 = problem.lo + span*sob.random_base2(6)
    F64 = np.array([obj(x) for x in X64])

    live_idx = np.argsort(F64)[:population]
    Xlive = X64[live_idx].copy()
    Flive = F64[live_idx].copy()

    sob2 = qmc.Sobol(d=d, scramble=True, seed=seed+2468)
    Xobs = problem.lo + span*sob2.random_base2(8)[:192]
    Fobs = np.array([obj(x) for x in Xobs])

    Xdiag = np.vstack([X64, Xobs])
    Fdiag = np.concatenate([F64, Fobs])

    # Large-scale isotropic envelope signature
    midpoint = (problem.lo + problem.hi) / 2
    halfspan = span / 2
    Y = (Xdiag-midpoint)/halfspan
    Z = np.column_stack([np.sum(Y*Y, axis=1), Y, np.ones(len(Y))])

    reg = Ridge(alpha=1e-8, fit_intercept=False).fit(Z, Fdiag)
    pred = reg.predict(Z)
    env_r2 = 1 - np.sum((Fdiag-pred)**2) / (
        np.sum((Fdiag-Fdiag.mean())**2) + 1e-300
    )
    a = float(reg.coef_[0])
    bvec = np.asarray(reg.coef_[1:1+d])

    # Multiscale Shepherd sentinel
    elite = Xdiag[np.argsort(Fdiag)[:16]]
    _, _, VT = np.linalg.svd(elite-elite.mean(0), full_matrices=False)
    u = VT[0] / max(np.linalg.norm(VT[0]), 1e-12)
    base = Xdiag[np.argmin(Fdiag)]

    scores = []
    amplitudes = []

    for frac in (.004, .015, .05):
        radius = frac*span*np.sqrt(d)
        vals = []
        for t in np.linspace(-1, 1, 17):
            vals.append(obj(np.clip(base+t*radius*u, problem.lo, problem.hi)))
        vals = np.asarray(vals)

        diff = np.diff(vals)
        tol = 1e-12*(1+np.max(np.abs(vals)))
        signs = np.sign(np.where(np.abs(diff)<tol, 0, diff))
        nz = signs[signs != 0]
        changes = np.sum(nz[1:]*nz[:-1] < 0) if len(nz) > 1 else 0
        scores.append(changes/max(1, len(nz)-1))
        amplitudes.append((vals.max()-vals.min())/(1+abs(vals.min())))

    small, medium, large = scores
    metadata = {
        "envelope_r2": float(env_r2),
        "oscillation_scores": [float(x) for x in scores],
        "oscillation_amplitudes": [float(x) for x in amplitudes],
        "max_fes": int(max_fes),
        "population": int(population),
        "seed": int(seed),
    }

    # ---------------- Mode selection ----------------

    # Envelope Shepherd Mode: strong broad envelope
    if env_r2 > .95 and a > 1e-10:
        yc = -bvec/(2*a)
        if np.all(np.isfinite(yc)) and np.max(np.abs(yc)) < 2.5:
            xenv = np.clip(midpoint + halfspan*yc, problem.lo, problem.hi)
            obj(xenv)
        if obj.n < obj.max_fes:
            _local_refine(obj, problem, obj.bestx, obj.max_fes-obj.n)
        metadata.update({"shepherd_mode":"Envelope", "mode_subtype":"quadratic"})

    # Spectral Shepherd Mode: strong fine-scale repetition
    elif small >= .30:
        smeta = _spectral_shepherd_mode(obj, problem, seed)
        metadata.update({
            "shepherd_mode":"Spectral",
            "mode_subtype":"full-harmonic",
            **smeta,
        })

    # Structural-Axis Shepherd Mode: low medium-scale oscillation
    elif medium <= .10:
        if env_r2 < .30:
            subtype, diag_ratio = _structural_native_or_conservative(
                obj, problem, seed
            )
            metadata.update({
                "shepherd_mode":"Structural-Axis",
                "mode_subtype":subtype,
                "diagonality_ratio":float(diag_ratio),
            })
        elif env_r2 < .56:
            subtype = _structural_periodic_stabilized(obj, problem, seed)
            metadata.update({
                "shepherd_mode":"Structural-Axis",
                "mode_subtype":subtype,
            })
        else:
            subtype = _structural_conservative(obj, problem, seed)
            metadata.update({
                "shepherd_mode":"Structural-Axis",
                "mode_subtype":subtype,
            })

    # Weak global envelope but strong symmetric behaviour
    elif env_r2 < .30:
        sloss = _symmetry_envelope_mode(
            obj, problem, seed, Xdiag, Fdiag, m_pairs=4
        )
        metadata.update({
            "shepherd_mode":"Envelope",
            "mode_subtype":"symmetry-centre",
            "symmetry_loss":float(sloss),
        })

    # Generic Shepherd Mode
    else:
        _generic_shepherd_mode(
            obj, problem, seed, Xlive.copy(), Flive.copy()
        )
        metadata.update({
            "shepherd_mode":"Generic",
            "mode_subtype":"flock+Powell",
        })

    metadata["function_evaluations"] = int(obj.n)
    metadata["best_f"] = float(obj.best)

    if return_metadata:
        return obj.bestx.copy(), float(obj.best), metadata
    return obj.bestx.copy(), float(obj.best)
