"""Station early-warning samples from the DB Timetables scrape (piebro/deutsche-bahn-data).

Two steps, shared by the training scripts and the TimeNet connector:

aggregate(files)       raw monthly parquet -> 15-minute bins per station, the 400 busiest
                       stations, their 4 main feeder stations, and a national series
cut_samples(tables)    bins -> (station, t0) samples with a 24 h / 7 d context and a label

A sample is a station that is calm at t0 (its last two hours were at most its usual 75th
percentile of unreliable trains). The label says whether a disruption starts in [t0, t0 + 2 h):
the share of late (>= 6 min) or cancelled arrivals reaches the station's own 90th percentile,
at least 20 %. Inputs only use bins planned before t0; thresholds use the training months only.
Times are the source's local time (Europe/Berlin), as published.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd


LATE_MIN = 6  # DB's punctuality threshold: late from 6 minutes
N_STATIONS = 400
N_FEEDERS = 4  # feeder stations kept per station
REF_MONTH = "2026-03"  # picks stations and feeders
BIN = pd.Timedelta(minutes=15)
CTX = 96  # 24 h of 15-minute bins
WEEK_H = 168  # 7 days, hourly
HORIZON = 8  # 2 h of 15-minute bins
FIRST_HOUR, LAST_HOUR = 6, 21  # t0 candidates from 06:00 to 21:30
MIN_TRAINS = 4  # arrivals needed in both 2 h windows around t0
INDEX_START, INDEX_END = "2025-11-01", "2026-11-01"
SPLITS = {
    "train": ("2025-11-08", "2026-06-30"),
    "val": ("2026-07-02", "2026-07-31"),
    "test": ("2026-08-02", "2026-09-30"),
    "oot": ("2026-10-02", "2026-10-31"),
}
CAP = {"train": 80_000, "val": 10_000, "test": 30_000, "oot": 20_000}
CHANNELS = [
    "station_delay_min",
    "station_late_share",
    "station_cancel_share",
    "station_arrivals",
    "feeder1_delay_min",
    "feeder2_delay_min",
    "feeder3_delay_min",
    "feeder4_delay_min",
    "national_delay_min",
]


# --------------------------------------------------------------------------- aggregate


def aggregate(files: Sequence[Path], ref_file: Path, threads: int | None = None) -> dict[str, pd.DataFrame]:
    """Aggregate the raw monthly files into station, feeder, bin and national tables.

    The SQL below formats in local file paths from the caller and module constants only, so
    the string-built queries are not an injection risk.

    Args:
        files: The raw monthly parquet files.
        ref_file: The monthly file that picks the stations and their feeders.
        threads: DuckDB threads; DuckDB's default when None.

    Returns:
        The ``stations``, ``feeders``, ``bins`` and ``national`` tables.
    """
    con = duckdb.connect()
    if threads:
        con.sql(f"set threads={threads}")
    # ruff: ignore[hardcoded-sql-expression] (formats in a local file path and a constant)
    con.sql(f"""
        create table stations as
        select eva, min(station_name) as name, count(*) as n_month,
               row_number() over (order by count(*) desc, eva) as rank
        from read_parquet('{ref_file}')
        where arrival_planned_time is not null and train_type <> 'Bus'
        group by eva order by n_month desc, eva limit {N_STATIONS}
    """)
    # Feeders: the station a train stopped at just before this one, counted per ride.
    # ruff: ignore[hardcoded-sql-expression] (formats in a local file path)
    con.sql(f"""
        create table feeders as
        with r as (
            select train_line_ride_id as ride, train_line_station_num as k, eva
            from read_parquet('{ref_file}') where train_type <> 'Bus'
        ), p as (
            select a.eva, b.eva as feeder_eva, count(*) as n
            from r a join r b on a.ride = b.ride and b.k = a.k - 1
            where a.eva in (select eva from stations) and b.eva <> a.eva
            group by 1, 2
        )
        select *, row_number() over (partition by eva order by n desc, feeder_eva) as rank
        from p qualify rank <= 4
    """)
    feeder_set = "select eva from stations union select feeder_eva from feeders"
    paths = sorted(str(p) for p in files)
    # One row per stop event. Ids repeat across monthly files at month borders; keep the
    # latest scrape of each (ties broken on every column used, so the result is deterministic).
    # ruff: ignore[hardcoded-sql-expression] (formats in local file paths and a fixed subquery)
    con.sql(f"""
        create table ev as
        select eva,
               time_bucket(interval 15 minute, arrival_planned_time) as bin,
               greatest(coalesce(epoch(arrival_change_time - arrival_planned_time)::bigint, 0), 0) as d_s,
               coalesce(arrival_is_canceled, false) as c
        from (
            select distinct on (id) * from read_parquet({paths!r}, union_by_name = true)
            where arrival_planned_time is not null and train_type <> 'Bus'
              -- replacement services only appear in the source from 2026-09; drop them so
              -- training and test months count the same kind of trains
              and not coalesce(is_replacement_train, false)
              and eva in ({feeder_set})
            order by id, time desc, arrival_change_time desc nulls last, eva, arrival_is_canceled
        )
    """)
    # ruff: ignore[hardcoded-sql-expression] (formats in a constant)
    con.sql(f"""
        create table bins as
        select eva, bin, count(*) as n,
               -- integer seconds, summed exactly, then minutes (delays capped at 3 h)
               sum(case when c then 0 else least(d_s, 10800) end) / 60.0 as delay_sum,
               sum((not c and d_s >= {LATE_MIN * 60})::int) as n_late,
               sum(c::int) as n_cancel
        from ev group by 1, 2
    """)
    con.sql("""
        create table national as
        select bin, sum(n) as n, sum(delay_sum) as delay_sum, sum(n_late) as n_late,
               sum(n_cancel) as n_cancel
        from bins where eva in (select eva from stations) group by 1
    """)
    tables = ["stations", "feeders", "bins", "national"]
    return {t: con.sql(f"select * from {t}").df() for t in tables}  # ruff: ignore[hardcoded-sql-expression]


# --------------------------------------------------------------------------- samples


def unseen_station(eva: str) -> bool:
    """Stable 20 % of stations held out of training entirely."""
    return int(hashlib.sha256(eva.encode()).hexdigest(), 16) % 5 == 0


class _Grid(NamedTuple):
    """Everything cut from the bins once and shared by all splits; stations come first in ``dense``."""

    index: pd.DatetimeIndex
    evas: list[str]
    names: dict[str, str]
    dense: np.ndarray  # (E, T, 4) n, delay_sum, n_late, n_cancel for stations, then feeders
    series: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # _derive(dense)
    hourly_delay: np.ndarray
    nat_delay: np.ndarray
    missing: np.ndarray
    feed_idx: np.ndarray
    feed_names: np.ndarray
    unseen: np.ndarray


def _dense(bins: pd.DataFrame, evas: list[str], index: pd.DatetimeIndex) -> np.ndarray:
    """(E, T, 4) array of n, delay_sum, n_late, n_cancel; zeros where no train was planned."""
    pos = {e: i for i, e in enumerate(evas)}
    b = bins[bins.eva.isin(pos)]
    ti = ((pd.to_datetime(b.bin) - index[0]) // BIN).to_numpy()
    ok = (ti >= 0) & (ti < len(index))
    arr = np.zeros((len(evas), len(index), 4), np.float32)
    ei = b.eva.map(pos).to_numpy()
    arr[ei[ok], ti[ok]] = b[["n", "delay_sum", "n_late", "n_cancel"]].to_numpy(np.float32)[ok]
    return arr


def _derive(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean delay, late share, cancel share, arrivals per bin (0 where no trains)."""
    n, dsum, late, canc = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    ran = n - canc
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_delay = np.where(ran > 0, dsum / ran, 0.0)
        late_share = np.where(ran > 0, late / ran, 0.0)
        cancel_share = np.where(n > 0, canc / n, 0.0)
    return mean_delay, late_share, cancel_share, n


def _window_share(a: np.ndarray, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Unreliable share and train count over [start, start+length) for every station."""
    s = a[:, start : start + length].sum(axis=1)
    n = s[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (s[:, 2] + s[:, 3]) / n, np.nan), n


def _national(national: pd.DataFrame, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """National mean delay per bin, and the scrape gaps (bins under half their usual count)."""
    # A bin is missing when the national count is under half of its usual value for that
    # weekday and time of day.
    national = national.assign(bin=lambda x: pd.to_datetime(x.bin)).sort_values("bin")
    nat = national.set_index("bin").reindex(index, fill_value=0)
    nat_n = nat.n.to_numpy(np.float32)
    ran = (nat.n - nat.n_cancel).to_numpy(np.float32)
    nat_delay = np.where(ran > 0, nat.delay_sum.to_numpy(np.float32) / np.maximum(ran, 1), 0)
    slot = (index.day_of_week * 96 + index.hour * 4 + index.minute // 15).to_numpy()
    expected = pd.Series(nat_n).groupby(slot).transform("median").to_numpy()
    return nat_delay, nat_n < 0.5 * expected


def _feeder_rows(feeders: pd.DataFrame, evas: list[str], all_evas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Feeder rows in ``dense`` and feeder ids per station, padded with -1 and ""."""
    pos = {e: i for i, e in enumerate(all_evas)}
    feed_idx = np.full((len(evas), N_FEEDERS), -1)
    feed_names = np.full((len(evas), N_FEEDERS), "", dtype=object)
    for eva, feeder_eva, rank in zip(feeders["eva"], feeders["feeder_eva"], feeders["rank"], strict=True):
        if eva in pos and feeder_eva in pos:
            feed_idx[evas.index(eva), rank - 1] = pos[feeder_eva]
            feed_names[evas.index(eva), rank - 1] = feeder_eva
    return feed_idx, feed_names


def _grid(tables: dict[str, pd.DataFrame], log: Callable[[str], object]) -> _Grid:
    """Dense bins, derived series, national series and feeder rows for the whole index."""
    stations = tables["stations"].sort_values("rank")
    index = pd.date_range(INDEX_START, INDEX_END, freq=BIN, inclusive="left")
    evas = [str(e) for e in stations.eva]
    all_evas = evas + sorted(set(tables["feeders"].feeder_eva) - set(evas))
    dense = _dense(tables["bins"], all_evas, index)
    nat_delay, missing = _national(tables["national"], index)
    log(f"missing bins: {missing.mean():.3%}")
    feed_idx, feed_names = _feeder_rows(tables["feeders"], evas, all_evas)
    hourly = dense[: len(evas)].reshape(len(evas), -1, 4, 4).sum(axis=2)
    return _Grid(
        index=index,
        evas=evas,
        names=dict(zip(stations.eva, stations.name, strict=True)),
        dense=dense,
        series=_derive(dense),
        hourly_delay=_derive(hourly)[0],
        nat_delay=nat_delay,
        missing=missing,
        feed_idx=feed_idx,
        feed_names=feed_names,
        unseen=np.array([unseen_station(e) for e in evas]),
    )


def _bin_at(index: pd.DatetimeIndex, day: str, days_after: int = 0) -> int:
    """Position of the first bin at or after midnight of ``day`` plus ``days_after`` days."""
    return int(index.searchsorted((pd.Timestamp(day) + pd.Timedelta(days=days_after)).to_datetime64()))


def _candidate_t0(index: pd.DatetimeIndex, missing: np.ndarray) -> np.ndarray:
    """Every 30 min from 06:00 to 21:30, with full context and target data."""
    miss_cum = np.concatenate([[0], np.cumsum(missing)])
    t0_all = np.arange(WEEK_H * 4, len(index) - HORIZON, 2)
    hrs = index.hour.to_numpy()
    t0_all = t0_all[(hrs[t0_all] >= FIRST_HOUR) & (hrs[t0_all] <= LAST_HOUR)]
    return t0_all[(miss_cum[t0_all + HORIZON] - miss_cum[t0_all - CTX]) == 0]


def _thresholds(grid: _Grid, t0_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-station 90th (at least 20 %) and 75th percentile of the 2 h share in the training period."""
    tr_lo, tr_hi = (_bin_at(grid.index, d) for d in SPLITS["train"])
    tr_t0 = t0_all[(t0_all >= tr_lo) & (t0_all < tr_hi)]
    station_bins = grid.dense[: len(grid.evas)]
    shares = np.stack([_window_share(station_bins, t, HORIZON)[0] for t in tr_t0[::3]], axis=1)
    p90 = np.maximum(np.nanpercentile(shares, 90, axis=1), 0.20)
    p75 = np.nanpercentile(shares, 75, axis=1)
    return p90, p75


def _split_rows(grid: _Grid, ts: np.ndarray, p75: np.ndarray, *, skip_unseen: bool) -> np.ndarray:
    """(station, t0, current share, future share) for every station that is calm at a t0 in ``ts``."""
    station_bins = grid.dense[: len(grid.evas)]
    rows = []
    for t in ts:
        cur, cur_n = _window_share(station_bins, t - HORIZON, HORIZON)
        fut, fut_n = _window_share(station_bins, t, HORIZON)
        ok = (cur_n >= MIN_TRAINS) & (fut_n >= MIN_TRAINS) & (cur <= p75)
        if skip_unseen:
            ok &= ~grid.unseen
        rows.extend((s, t, cur[s], fut[s]) for s in np.flatnonzero(ok))
    return np.array(rows)


def _context(grid: _Grid, s_i: np.ndarray, t_i: np.ndarray) -> np.ndarray:
    """(N, channels, 96) inputs over the 24 h before t0, in the order of CHANNELS."""
    mean_delay, late_share, cancel_share, n_arr = grid.series
    ctx = np.zeros((len(s_i), len(CHANNELS), CTX), np.float32)
    win = t_i[:, None] + np.arange(-CTX, 0)[None]
    ctx[:, 0] = mean_delay[s_i[:, None], win]
    ctx[:, 1] = late_share[s_i[:, None], win]
    ctx[:, 2] = cancel_share[s_i[:, None], win]
    ctx[:, 3] = n_arr[s_i[:, None], win]
    for k in range(N_FEEDERS):
        f = grid.feed_idx[s_i, k]
        has = f >= 0
        ctx[has, 4 + k] = mean_delay[f[has][:, None], win[has]]
    ctx[:, 8] = grid.nat_delay[win]
    return ctx


def _future_share(grid: _Grid, s_i: np.ndarray, t_i: np.ndarray) -> np.ndarray:
    """(N, 8) unreliable share in each target bin (0 where no trains)."""
    n_arr = grid.series[3]
    fwin = t_i[:, None] + np.arange(HORIZON)[None]
    n_fut = n_arr[s_i[:, None], fwin]
    unreliable = grid.dense[s_i[:, None], fwin, 2] + grid.dense[s_i[:, None], fwin, 3]
    return np.where(n_fut > 0, unreliable / np.maximum(n_fut, 1), 0).astype(np.float32)


def _split_arrays(grid: _Grid, rows: np.ndarray, p90: np.ndarray) -> dict[str, np.ndarray]:
    """The arrays of one split, one entry per row of ``rows``."""
    s_i, t_i = rows[:, 0].astype(int), rows[:, 1].astype(int)
    y_share = rows[:, 3]
    week = grid.hourly_delay[s_i[:, None], (t_i // 4)[:, None] + np.arange(-WEEK_H, 0)[None]]
    t0 = grid.index[t_i]
    return {
        "ctx": _context(grid, s_i, t_i),
        "week": week.astype(np.float32),
        "fut": _future_share(grid, s_i, t_i),
        "y": (y_share >= p90[s_i]).astype(np.int8),
        "y_share": y_share.astype(np.float32),
        "cur_share": rows[:, 2].astype(np.float32),
        "thr": p90[s_i].astype(np.float32),
        "eva": np.array([str(grid.evas[i]) for i in s_i], dtype=str),
        "name": np.array([str(grid.names[grid.evas[i]]) for i in s_i], dtype=str),
        "feeder_eva": np.array([[str(x) for x in grid.feed_names[i]] for i in s_i], dtype=str).reshape(
            len(rows), N_FEEDERS
        ),
        "t0": np.asarray(t0.strftime("%Y-%m-%dT%H:%M"), dtype=str),
        "hour": t0.hour.to_numpy(),
        "dow": t0.day_of_week.to_numpy(),
        "unseen": grid.unseen[s_i],
        "n_feeders": (grid.feed_idx[s_i] >= 0).sum(axis=1),
    }


def cut_samples(
    tables: dict[str, pd.DataFrame], seed: int = 0, log: Callable[[str], object] = print
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Cut the binned tables into early-warning samples per split.

    Args:
        tables: The tables from :func:`aggregate` (or read back from its parquet files).
        seed: Seed for subsampling the splits that have more samples than their cap.
        log: Receives one progress line per step.

    Returns:
        The arrays per split (``train``, ``val``, ``test``, ``oot``) and the per-station
        thresholds (``evas``, ``p90``, ``p75``).
    """
    rng = np.random.default_rng(seed)
    grid = _grid(tables, log)
    t0_all = _candidate_t0(grid.index, grid.missing)
    p90, p75 = _thresholds(grid, t0_all)
    thresholds = {"evas": np.array(grid.evas), "p90": p90, "p75": p75}
    out = {}
    for split, (lo_d, hi_d) in SPLITS.items():
        lo = _bin_at(grid.index, lo_d)
        hi = _bin_at(grid.index, hi_d, days_after=1)
        ts = t0_all[(t0_all >= lo) & (t0_all < hi)]
        rows = _split_rows(grid, ts, p75, skip_unseen=split == "train")
        if len(rows) > CAP[split]:
            rows = rows[rng.choice(len(rows), CAP[split], replace=False)]
            rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
        arrays = out[split] = _split_arrays(grid, rows, p90)
        log(
            f"{split}: candidates {len(ts)} t0, N={len(rows)}, positive rate {arrays['y'].mean():.3f}, "
            f"unseen stations {arrays['unseen'].mean():.2f}"
        )
    return out, thresholds


# --------------------------------------------------------------------------- incoming trains

INCOMING = [
    "incoming_planned",
    "incoming_delay_mean_min",
    "incoming_delay_max_min",
    "incoming_not_started_share",
    "incoming_delay_gain_mean_min",
]
"""Per 15-minute bin of [t0, t0 + 2 h): the trains due at the station in that bin, as seen at t0."""
GAIN_STOPS = 2  # delay gained since this many stops before the last stop passed
DELAY_CAP_S = 10_800


def incoming(
    files: Sequence[Path], splits: dict[str, dict[str, np.ndarray]], threads: int | None = None
) -> dict[str, np.ndarray]:
    """Describe the trains due at each sample's station in the 2 h after t0, from what is known at t0.

    For every train run planned to arrive at the station in [t0, t0 + 2 h), the latest stop it
    departed from before t0 gives its current delay (departure delay there, at least 0) and how
    much delay it gained over the GAIN_STOPS stops before that. A run with no stop departed before
    t0 has not started yet. A run is one ride id on one day: the stop id without its stop number.
    Cancelled stops are not departures. Nothing at or after t0 enters, except the timetable.

    Args:
        files: The raw monthly parquet files.
        splits: The sample arrays from :func:`cut_samples` (uses ``eva`` and ``t0``).
        threads: DuckDB threads; DuckDB's default when None.

    Returns:
        For each split, an array (N, 5, HORIZON) in the order of INCOMING.
    """
    con = duckdb.connect()
    if threads:
        con.sql(f"set threads={threads}")
    frames = []
    for name, arrays in splits.items():
        frames.append(
            pd.DataFrame(
                {
                    "split": name,
                    "row": np.arange(len(arrays["eva"])),
                    "eva": arrays["eva"].astype(str),
                    "t0": pd.to_datetime(arrays["t0"]),
                }
            )
        )
    samples = pd.concat(frames, ignore_index=True)
    samples["sid"] = np.arange(len(samples))
    con.register("samples_df", samples[["sid", "eva", "t0"]])
    paths = sorted(str(p) for p in files)
    # ruff: ignore[hardcoded-sql-expression] (formats in local file paths)
    con.sql(f"""
        create table st as
        select regexp_replace(id, '-[0-9]+$', '') as run, train_line_station_num as k, eva,
               arrival_planned_time as arr_pl, departure_planned_time as dep_pl,
               departure_change_time as dep_chg, coalesce(departure_is_canceled, false) as dep_c
        from (
            select distinct on (id) * from read_parquet({paths!r}, union_by_name = true)
            where train_type <> 'Bus' and not coalesce(is_replacement_train, false)
            order by id, time desc, arrival_change_time desc nulls last, eva, arrival_is_canceled
        )
    """)
    con.sql("""
        create table sb as
        select sid, eva, b, t0, t0 + b * interval 15 minute as bstart
        from samples_df, range(8) r(b)
    """)
    # Trains due in each bin (the timetable is known in advance).
    con.sql("""
        create table pairs as
        select sb.sid, sb.b, sb.t0, st.run, st.k as kt
        from sb join st on st.eva = sb.eva and time_bucket(interval 15 minute, st.arr_pl) = sb.bstart
    """)
    # Departures that happened: not cancelled, at the changed time when there is one.
    con.sql("""
        create table dep as
        select run, k, coalesce(dep_chg, dep_pl) as t_dep,
               greatest(epoch(coalesce(dep_chg, dep_pl) - dep_pl)::bigint, 0) as d_s,
               epoch(coalesce(dep_chg, dep_pl) - dep_pl)::bigint as signed_s
        from st where dep_pl is not null and not dep_c
          and run in (select distinct run from pairs)
    """)
    # Latest departure before t0 of each run, then the same run GAIN_STOPS stops earlier.
    # ruff: ignore[hardcoded-sql-expression] (formats in constants)
    con.sql(f"""
        create table seen as
        select p.sid, p.b, p.run, p.kt,
               case when q.k < p.kt then least(q.d_s, {DELAY_CAP_S}) end as d_s,
               case when q.k < p.kt and r.k is not null and r.t_dep < p.t0
                    then q.signed_s - r.signed_s end as gain_s
        from pairs p
        asof left join dep q on p.run = q.run and p.t0 > q.t_dep
        left join dep r on r.run = q.run and r.k = q.k - {GAIN_STOPS}
    """)
    agg = con.sql("""
        select sid, b, count(*) as n, count(d_s) as n_seen, coalesce(sum(d_s), 0) as dsum,
               coalesce(max(d_s), 0) as dmax, count(gain_s) as n_gain, coalesce(sum(gain_s), 0) as gsum
        from seen group by sid, b
    """).df()
    out = np.zeros((len(samples), len(INCOMING), HORIZON), np.float32)
    sid, b = agg.sid.to_numpy(), agg.b.to_numpy()
    n, seen = agg.n.to_numpy(np.float64), agg.n_seen.to_numpy(np.float64)
    out[sid, 0, b] = n
    out[sid, 1, b] = np.where(seen > 0, agg.dsum.to_numpy(np.float64) / np.maximum(seen, 1) / 60, 0)
    out[sid, 2, b] = agg.dmax.to_numpy(np.float64) / 60
    out[sid, 3, b] = np.where(n > 0, (n - seen) / np.maximum(n, 1), 0)
    ng = agg.n_gain.to_numpy(np.float64)
    out[sid, 4, b] = np.where(ng > 0, agg.gsum.to_numpy(np.float64) / np.maximum(ng, 1) / 60, 0)
    return {name: out[(samples.split == name).to_numpy()] for name in splits}


# --------------------------------------------------------------------------- prompt

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
QUESTION = "Will a disruption start at this station within the next two hours? Answer yes or no."


def context_text(name: str, dow: int, t0: str, cur_share: float, thr: float) -> str:
    """Write the station context of a sample, as the models and the TimeNet task read it.

    Args:
        name: Station name.
        dow: Day of the week of t0, Monday = 0.
        t0: Time of the sample as ``YYYY-MM-DDTHH:MM``.
        cur_share: Share of late or cancelled arrivals in the two hours before t0.
        thr: The station's disruption threshold.

    Returns:
        The context sentences.
    """
    return (
        f"You monitor train arrivals at {name} station in Germany. "
        f"It is {DOW[int(dow)]}, {t0[11:16]}. "
        f"In the last two hours {cur_share:.0%} of arrivals here were late or cancelled. "
        f"A disruption means at least {thr:.0%} of arrivals late or cancelled "
        f"within the next two hours."
    )
