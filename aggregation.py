"""Pure aggregation/calibration helpers (no network). Tested in test_aggregation.py."""
import math

P_MIN, P_MAX = 0.02, 0.98


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def aggregate_binary(probs: list[float], extremize: float = 1.15) -> float:
    """Trimmed mean of log-odds, then moderate extremization around 50%.

    Averaging independent forecasts pulls toward 50%, so a factor slightly >1 on the
    log-odds corrects that (superforecaster-aggregation literature). Clamped to [2%, 98%].
    """
    if not probs:
        raise ValueError("no predictions")
    xs = sorted(logit(p) for p in probs)
    if len(xs) >= 5:
        xs = xs[1:-1]  # drop min and max
    m = sum(xs) / len(xs)
    return min(P_MAX, max(P_MIN, sigmoid(m * extremize)))


def aggregate_mc(prob_lists: list[list[float]], floor: float = 0.005) -> list[float]:
    """Geometric mean per option (log-space average), floored and renormalized."""
    n = len(prob_lists[0])
    logs = [sum(math.log(max(pl[i], floor)) for pl in prob_lists) / len(prob_lists) for i in range(n)]
    ex = [math.exp(v) for v in logs]
    s = sum(ex)
    return [e / s for e in ex]


def aggregate_percentiles(pct_lists: list[list[float]]) -> list[float]:
    """Mean of values at each declared percentile (quantile averaging), kept monotone."""
    n = len(pct_lists[0])
    out = [sum(pl[i] for pl in pct_lists) / len(pct_lists) for i in range(n)]
    for i in range(1, n):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-9 * (abs(out[i - 1]) + 1)
    return out


def brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2
