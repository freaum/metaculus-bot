#!/usr/bin/env bash
# ~5 h loop: run the bot every 20 min, keep a monthly spend total in the "state" branch (spend.json).
end=$((SECONDS + ${LOOP_SECONDS:-18000}))
while [ $SECONDS -lt $end ]; do
  start=$SECONDS
  month=$(date -u +%Y-%m)
  export SPEND_TOTAL=$(python3 - "$month" <<'PY'
import json, sys
m = sys.argv[1]
try:
    d = json.load(open("state/spend.json"))
    print(d.get("total", 0) if d.get("month") == m else 0)
except Exception:
    print(0)
PY
)
  echo "SPEND_TOTAL(month $month)=$SPEND_TOTAL"
  poetry run python main.py 2>&1 | tee run.log || echo "run failed, continuing"
  cost=$(grep -o 'COSTLOG usd=[0-9.]*' run.log | tail -1 | cut -d= -f2)
  if [ -n "$cost" ]; then
    python3 - "$month" "$cost" <<'PY'
import json, sys
m, c = sys.argv[1], float(sys.argv[2])
try:
    d = json.load(open("state/spend.json"))
except Exception:
    d = {}
if d.get("month") != m:
    d = {"month": m, "total": 0.0}
d["total"] = round(float(d.get("total", 0)) + c, 4)
json.dump(d, open("state/spend.json", "w"))
print("spend.json ->", d)
PY
    (cd state && git add spend.json && git -c user.name=keepalive -c user.email=keepalive@users.noreply.github.com commit -qm "spend $month" && git push -q origin HEAD:state) || echo "state push failed"
  fi
  el=$((SECONDS - start))
  if [ $el -lt ${LOOP_GAP:-1200} ]; then sleep $((${LOOP_GAP:-1200} - el)); fi
done
