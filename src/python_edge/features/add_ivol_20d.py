from __future__ import annotations

import pandas as pd



def add_ivol_20d(df1d: pd.DataFrame, market_proxy: pd.Series | None = None) -> pd.DataFrame:
    out = df1d.copy()
    ret = out["close"].pct_change()
    if market_proxy is None:
        resid = ret
    else:
        aligned = pd.concat(
            [ret.rename("ret"), market_proxy.rename("mkt")],
            axis=1,
            join="left",
        )
        beta = aligned["ret"].rolling(20, min_periods=20).cov(aligned["mkt"]) / aligned["mkt"].rolling(20, min_periods=20).var()
        fitted = beta * aligned["mkt"]
        resid = aligned["ret"] - fitted
    out["ivol_20d"] = resid.rolling(20, min_periods=20).std()
    return out