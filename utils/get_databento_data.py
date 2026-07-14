import os
from dotenv import load_dotenv
import databento as db

load_dotenv()

API_KEY = os.environ["DATABENTO_API_KEY"]

# Use an individual futures contract, not a continuous back-adjusted series.
SYMBOL = "ESZ3"
START = "2023-11-01T14:30:00Z"
END = "2023-11-01T21:00:00Z"

client = db.Historical(API_KEY)

request = {
    "dataset": "GLBX.MDP3",
    "symbols": [SYMBOL],
    "stype_in": "raw_symbol",
    "schema": "mbp-1",  # trades plus top-of-book bid/ask
    "start": START,
    "end": END,
}

# Check the charge before consuming free credit.
estimated_cost = client.metadata.get_cost(**request)
print(f"Estimated cost: ${estimated_cost:.4f}")

MAX_COST = 5.00
if estimated_cost > MAX_COST:
    raise RuntimeError(
        f"Request costs ${estimated_cost:.2f}, above the ${MAX_COST:.2f} limit."
    )

data = client.timeseries.get_range(**request)

df = data.to_df()
df.to_csv("ESZ3_mbp1_2023-11-01.csv")

print(df.head())
print(f"Saved {len(df):,} events")