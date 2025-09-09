import datetime
import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv
import os

import pytz

load_dotenv()
login = int(os.getenv("LOGIN"))
password = os.getenv("PASSWORD")
server = os.getenv("SERVER")
symbol = os.getenv("MT5_SYMBOL")
mt_path = os.getenv("MT5_PATH")

print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
 
if not mt5.initialize(mt_path, login=login, password=password, server=server):
    print("initialize() failed, error code =",mt5.last_error())
    quit()
 
selected=mt5.symbol_select("GBPUSD.a",True)
if not selected:
    print("Failed to select GBPUSD.a")
    mt5.shutdown()
    quit()
 
timezone = pytz.timezone("Etc/UTC")
utc_from = datetime.datetime(2025, 1, 10, tzinfo=timezone)
utc_to = datetime.datetime(2025, 1, 11, tzinfo=timezone)
ticks = mt5.copy_ticks_range("GBPUSD.a", utc_from, utc_to, mt5.COPY_TICKS_ALL)
print("Ticks received:",len(ticks))

df = pd.DataFrame(ticks)  # preserves field names from MT5
# Convert timestamp(s) to UTC datetimes (choose one or keep both)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)

# Optional: nicer column order
cols = ["time", "time_msc", "bid", "ask", "last", "volume", "volume_real", "flags"]
df = df[cols]

df.to_csv("tick_result.csv", index=False)

mt5.shutdown()