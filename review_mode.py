"""Read-only performance review (Test Bot mode=review). No LLM calls, no forecasts, no comments.

Prints REVIEW lines: leaderboard top entries, per-question contributions of this bot and of the
top bots, and every resolved question this bot forecast with its forecast, resolution and score.
"""
import json
import os
import time

import requests

BASE = "https://www.metaculus.com/api"
H = {"Authorization": "Token " + os.environ.get("METACULUS_TOKEN", "")}


def _get(url, **params):
    r = None
    for _ in range(4):
        r = requests.get(url, headers=H, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(6)
            continue
        break
    time.sleep(1.2)
    return r


def _short(obj, n=300):
    return json.dumps(obj, default=str)[:n]


def _user(e):
    u = e.get("user") or {}
    return u.get("username") or e.get("aggregation_method") or "?", u.get("id"), u.get("is_bot")


def leaderboard(pid):
    r = _get(f"{BASE}/leaderboards/project/{pid}/")
    if r.status_code != 200:
        print("REVIEW", pid, "leaderboard HTTP", r.status_code)
        return []
    data = r.json()
    lbs = data if isinstance(data, list) else [data]
    out = []
    for lb in lbs:
        es = lb.get("entries") or []
        print("REVIEW", pid, "lb", lb.get("score_type"), "entries", len(es), "keys", sorted((es[0] or {}).keys()) if es else [])
        for i, e in enumerate(es):
            name, uid, bot = _user(e)
            row = (i + 1, name, uid, bot, e.get("score"), e.get("coverage"), e.get("contribution_count"), e.get("rank"))
            out.append(row)
            if i < 15 or "freaum" in str(name).lower():
                print("REVIEW", pid, "rank", row)
    return out


def contributions(pid, uid, label):
    r = _get(f"{BASE}/medals/contributions/", project_id=pid, user_id=uid, for_user=uid)
    if r.status_code != 200:
        print("REVIEW", pid, "contrib", label, "HTTP", r.status_code, r.text[:150])
        return {}
    cs = (r.json() or {}).get("contributions") or []
    print("REVIEW", pid, "contrib", label, "n", len(cs), "keys", sorted(cs[0].keys()) if cs else [])
    res = {}
    for c in cs:
        qid = c.get("question_id") or (c.get("question") or {}).get("id")
        res[qid] = c.get("score")
        if label == "me":
            print("REVIEW", pid, "C", qid, c.get("post_id"), c.get("score"), c.get("coverage"),
                  str(c.get("question_resolution"))[:20], str(c.get("question_title"))[:90])
    return res


def resolved_mine(tid):
    from forecasting_tools import ApiFilter, MetaculusClient
    import asyncio

    f = ApiFilter(allowed_tournaments=[tid], allowed_statuses=["resolved"],
                  group_question_mode="unpack_subquestions", is_previously_forecasted_by_user=True)
    try:
        qs = asyncio.run(MetaculusClient().get_questions_matching_filter(f, num_questions=400, error_if_question_target_missed=False))
    except Exception as e:
        print("REVIEW", tid, "resolved ERR", repr(e)[:200])
        return
    print("REVIEW", tid, "resolved_forecasted", len(qs))
    for q in qs:
        qj = (q.api_json or {}).get("question") or {}
        mf = qj.get("my_forecasts") or {}
        latest = mf.get("latest") or {}
        fv = latest.get("forecast_values")
        mine = fv[1] if (fv and len(fv) == 2) else (fv if fv and len(fv) < 12 else latest.get("centers"))
        agg = ((qj.get("aggregations") or {}).get("recency_weighted") or {}).get("latest") or {}
        afv = agg.get("forecast_values")
        comm = afv[1] if (afv and len(afv) == 2) else (afv if afv and len(afv) < 12 else agg.get("centers"))
        print("REVIEW", tid, "Q", q.id_of_post, q.id_of_question, type(q).__name__, "res", q.resolution_string,
              "mine", _short(mine, 120), "comm", _short(comm, 120), "score", _short(mf.get("score_data"), 260),
              "n_hist", len(mf.get("history") or []), "|", (q.question_text or "")[:90])


def coverage(tid):
    """All questions of a tournament (open/closed/resolved): did this bot forecast each one?"""
    from forecasting_tools import ApiFilter, MetaculusClient
    import asyncio

    f = ApiFilter(allowed_tournaments=[tid], allowed_statuses=["open", "closed", "resolved"],
                  group_question_mode="unpack_subquestions")
    try:
        qs = asyncio.run(MetaculusClient().get_questions_matching_filter(f, num_questions=400, error_if_question_target_missed=False))
    except Exception as e:
        print("REVIEW", tid, "coverage ERR", repr(e)[:200])
        return
    n_mine = 0
    for q in sorted(qs, key=lambda x: str(x.open_time)):
        qj = (q.api_json or {}).get("question") or {}
        mf = qj.get("my_forecasts") or {}
        hist = mf.get("history") or []
        first = hist[0].get("start_time") if hist else None
        n_mine += bool(hist)
        print("REVIEW", tid, "COV", q.id_of_post, type(q).__name__[:8], q.state, "res", q.resolution_string,
              "open", str(q.open_time)[:16], "close", str(q.close_time)[:16], "mine", bool(hist), "first", first,
              "nfc", q.num_forecasters, "|", (q.question_text or "")[:70])
    print("REVIEW", tid, "coverage", n_mine, "of", len(qs))


def run():
    for tid in [33121, "minibench"]:
        coverage(tid)
    for pid in ["33121", "33132"]:
        rows = leaderboard(pid)
        me = [r for r in rows if "freaum" in str(r[1]).lower()]
        tops = [r for r in rows if r[2] and "freaum" not in str(r[1]).lower()][:3]
        if me:
            contributions(pid, me[0][2], "me")  # other users' contributions are private (403)
    for tid in [33121, 33132, "minibench"]:
        resolved_mine(tid)
