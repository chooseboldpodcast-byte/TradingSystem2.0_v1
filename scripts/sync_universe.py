#!/usr/bin/env python3
"""
Sync live_universe.txt with download_data.py
Excludes benchmark tickers (^GSPC, etc.)
"""
from download_data import ALL_SYMBOLS

# Exclude benchmarks (anything starting with ^)
tradeable_symbols = [s for s in ALL_SYMBOLS if not s.startswith('^')]

# Sort alphabetically
tradeable_symbols.sort()

# Write to live_universe.txt
with open('live_universe.txt', 'w') as f:
    for symbol in tradeable_symbols:
        f.write(f"{symbol}\n")

print(f"✅ Synced live_universe.txt with {len(tradeable_symbols)} symbols")
print(f"   (Excluded benchmarks like ^GSPC)")
print(f"\nTradeable symbols:")
for symbol in tradeable_symbols[:10]:
    print(f"  - {symbol}")
if len(tradeable_symbols) > 10:
    print(f"  ... and {len(tradeable_symbols) - 10} more")