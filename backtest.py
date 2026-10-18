"""Backtest skeleton (UNTESTED: needs METACULUS_TOKEN + OPENROUTER_API_KEY).
Pulls resolved binary questions from a past tournament and scores raw vs aggregated/extremized
forecasts with Brier. Caveat: LLM knowledge cutoff leakage -> only use questions resolved after
the model's cutoff, and compare variants (EXTREMIZE 1.0 vs 1.15) rather than trusting absolute scores.
Usage: python backtest.py <tournament_id> <n>
"""
import asyncio, sys, json
from forecasting_tools import MetaculusClient, ApiFilter
from aggregation import aggregate_binary, brier

async def main(tid, n):
    client = MetaculusClient()
    qs = await client.get_questions_matching_filter(
        ApiFilter(allowed_tournaments=[tid], allowed_statuses=["resolved"], allowed_types=["binary"]),
        num_questions=n)
    rows = []
    for q in qs:
        res = q.resolution_string
        if res not in ("yes", "no"):
            continue
        rows.append({"url": q.page_url, "outcome": int(res == "yes"), "community": q.community_prediction_at_access_time})
    json.dump(rows, open("backtest_questions.json", "w"), indent=1)
    print(len(rows), "resolved binary questions saved; next: run bot's binary forecast per question (TODO when credits exist)")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2])))
