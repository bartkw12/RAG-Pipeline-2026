import json
d = json.load(open("_golden_candidates.json", "r", encoding="utf-8"))
for q, v in d.items():
    n = v["n_chunks"]
    strat = v["strategy"]
    top = v["top_chunks"]
    tc = top[0]["test_case_id"] if top else ""
    mod = top[0]["module"] if top else ""
    ct = top[0]["chunk_type"] if top else ""
    print(f"{q[:55]:55s}  {strat:20s}  n={n:2d}  top={ct:15s}  mod={mod:6s}  tc={tc}")
