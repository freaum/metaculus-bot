from aggregation import *

def test_binary():
    assert abs(aggregate_binary([0.5, 0.5, 0.5]) - 0.5) < 1e-9
    a = aggregate_binary([0.8, 0.8, 0.8], 1.0)
    assert abs(a - 0.8) < 1e-6
    assert aggregate_binary([0.8, 0.8, 0.8]) > 0.8
    assert aggregate_binary([0.999] * 5) == P_MAX and aggregate_binary([0.0001] * 5) == P_MIN
    # outlier trimmed with >=5 preds
    assert aggregate_binary([0.7, 0.7, 0.7, 0.7, 0.01], 1.0) > 0.65

def test_mc():
    r = aggregate_mc([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
    assert abs(sum(r) - 1) < 1e-9 and r[0] > r[1] > r[2]

def test_pct():
    r = aggregate_percentiles([[1, 2, 3], [3, 4, 5]])
    assert r == [2, 3, 4]
