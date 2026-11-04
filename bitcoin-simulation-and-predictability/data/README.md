# Data

`btc_daily_coinmetrics.csv`: daily Bitcoin data, 18 Jul 2010 – 23 May 2026, from the
[Coin Metrics Community Network Data](https://github.com/coinmetrics/data) (file `csv/btc.csv`).

| Column | Coin Metrics field | Meaning |
|---|---|---|
| `price_usd` | `PriceUSD` | Reference price in US dollars (end of day, UTC) |
| `volume_usd` | `volume_reported_spot_usd_1d` | Reported spot trading volume |
| `active_addresses` | `AdrActCnt` | Addresses active on the blockchain that day |
| `transactions` | `TxCnt` | On-chain transactions |
| `hash_rate` | `HashRate` | Mining hash rate |

**Licence:** this data is © Coin Metrics, Inc. and is licensed under
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
It is redistributed here, unmodified apart from the column selection, for non-commercial academic use.
The MIT licence of this repository applies to the code only.
