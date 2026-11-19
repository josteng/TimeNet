from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from timenet.dataset.axis import RegularAxis
from timenet.types import ClassificationTask, ScalarPredictionTask, Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.error404.db_station_early_warning.connector import (
    MONTHS,
    DbStationEarlyWarningConnector,
)
from timenet_connectors.datasets.error404.db_station_early_warning.pipeline import (
    REF_MONTH,
    aggregate,
    cut_samples,
    incoming,
)


_STATIONS = (("8000001", "Alpha"), ("8000002", "Beta"), ("8000003", "Gamma"))
"""Invented stations on one line, served every 15 minutes in the order Alpha, Beta, Gamma."""


def _month_frame(month: str, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    departures = pd.date_range(start, end, freq="15min", inclusive="left")
    # Calm most of the time; a few multi-hour disruptions per month make positive labels.
    burst = np.zeros(len(departures))
    for centre in rng.choice(len(departures), size=6, replace=False):
        burst[centre : centre + 12] = 1.0
    rows = []
    for i, dep in enumerate(departures):
        ride = f"ride-{month}-{i}"
        for k, (eva, name) in enumerate(_STATIONS):
            planned = dep + pd.Timedelta(minutes=5 * k)
            delay = int(rng.integers(0, 3) + burst[i] * rng.integers(5, 20))
            cancelled = bool(burst[i] and rng.random() < 0.1)
            rows.append(
                {
                    "station_name": name,
                    "eva": eva,
                    "train_type": "RE",
                    "train_line_ride_id": ride,
                    "train_line_station_num": k + 1,
                    "arrival_planned_time": planned,
                    "arrival_change_time": planned + pd.Timedelta(minutes=delay),
                    "arrival_is_canceled": cancelled,
                    "departure_planned_time": planned + pd.Timedelta(minutes=1),
                    "departure_change_time": planned + pd.Timedelta(minutes=1 + delay),
                    "departure_is_canceled": cancelled,
                    "is_replacement_train": False,
                    "time": planned,
                    "id": f"{ride}-{k}",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def refs(tmp_path_factory: pytest.TempPathFactory) -> list[HubFileRef]:
    root = tmp_path_factory.mktemp("monthly")
    rng = np.random.default_rng(7)
    out = []
    for month in MONTHS:
        path = root / f"data-{month}.parquet"
        _month_frame(month, rng).to_parquet(path, index=False)
        out.append(HubFileRef(path=path, part=month))
    return out


def _paths(refs: list[HubFileRef]) -> dict[str, Path]:
    return {ref.part: ref.path for ref in refs}


def test_convert_matches_pipeline_samples(refs: list[HubFileRef]) -> None:
    paths = _paths(refs)
    samples, _ = cut_samples(aggregate([paths[m] for m in MONTHS], paths[REF_MONTH]), log=lambda _msg: None)
    dataset = DbStationEarlyWarningConnector().convert(refs)

    n_samples = sum(len(arrays["y"]) for arrays in samples.values())
    assert n_samples > 0
    tasks = list(dataset.tasks)
    assert len(tasks) == 2 * n_samples
    assert sum(isinstance(task, ClassificationTask) for task in tasks) == n_samples
    assert sum(isinstance(task, ScalarPredictionTask) for task in tasks) == n_samples

    splits = {task.split for task in tasks}
    assert splits <= {Split.TRAIN, Split.VALIDATION, Split.TEST}
    cells = {task.metadata["cell"] for task in tasks}
    assert cells <= {"train", "validation", "test", "oot"}


def test_record_axes_place_context_right_before_t0(refs: list[HubFileRef]) -> None:
    paths = _paths(refs)
    samples, _ = cut_samples(aggregate([paths[m] for m in MONTHS], paths[REF_MONTH]), log=lambda _msg: None)
    dataset = DbStationEarlyWarningConnector().convert(refs)
    record = next(iter(dataset.records))
    t0 = datetime.fromisoformat(str(record.metadata["t0"])).replace(tzinfo=ZoneInfo("Europe/Berlin"))
    origin_us = record.start_time.timestamp
    assert origin_us == int((t0 - timedelta(days=7)).timestamp() * 1_000_000)

    by_name = {signal.name: signal for signal in record.signals}
    day = by_name["arrival_delay_15min"]
    week = by_name["arrival_delay_hourly"]
    assert len(day.to_numpy()) == 96
    assert len(week.to_numpy()) == 168
    # The first Record is the first training sample, value for value.
    first = samples["train"]
    assert record.metadata["t0"] == str(first["t0"][0])
    assert np.array_equal(day.to_numpy(), first["ctx"][0, 0])
    assert np.array_equal(by_name["network_arrival_delay_15min"].to_numpy(), first["ctx"][0, 8])
    assert np.array_equal(week.to_numpy(), first["week"][0])
    # The 24 h window starts 6 days after the origin and its last bin ends exactly at t0.
    assert isinstance(day.time_axis, RegularAxis)
    assert isinstance(week.time_axis, RegularAxis)
    assert day.time_axis.offset_us == 6 * 86_400_000_000
    assert day.time_axis.period_us == 900_000_000
    assert day.time_axis.offset_us + 96 * day.time_axis.period_us == 7 * 86_400_000_000
    assert week.time_axis.period_us == 3_600_000_000


def test_labels_only_depend_on_the_future_window(refs: list[HubFileRef]) -> None:
    paths = _paths(refs)
    samples, thresholds = cut_samples(aggregate([paths[m] for m in MONTHS], paths[REF_MONTH]), log=lambda _msg: None)
    for arrays in samples.values():
        # Every sample is calm at t0 and its label is the future share against its own threshold.
        assert np.array_equal(arrays["y"], (arrays["y_share"] >= arrays["thr"]).astype(np.int8))
        assert (arrays["thr"] >= 0.2).all()
    assert set(thresholds["evas"]) == {eva for eva, _ in _STATIONS}


def _due_by_hand(raw: pd.DataFrame, eva: str, t0: pd.Timestamp, b: int) -> tuple[int, float]:
    """Trains due in bin b and their mean delay at the last stop left before t0, from the raw rows."""
    start = t0 + pd.Timedelta(minutes=15 * b)
    here = raw[(raw.eva == eva) & (raw.arrival_planned_time >= start)]
    here = here[here.arrival_planned_time < start + pd.Timedelta(minutes=15)]
    delays = []
    for run, kt in zip(here.run, here.train_line_station_num, strict=True):
        before = raw[(raw.run == run) & (raw.train_line_station_num < kt)]
        left = before[(before.departure_change_time < t0) & ~before.departure_is_canceled]
        if len(left):
            last = left.sort_values("departure_change_time").iloc[-1]
            delays.append(max((last.departure_change_time - last.departure_planned_time).total_seconds(), 0))
    return len(here), (float(np.mean(delays)) / 60 if delays else 0.0)


def test_incoming_trains_use_only_departures_before_t0(refs: list[HubFileRef]) -> None:
    paths = _paths(refs)
    files = [paths[m] for m in MONTHS]
    samples, _ = cut_samples(aggregate(files, paths[REF_MONTH]), log=lambda _msg: None)
    due = incoming(files, samples)["test"]
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw["run"] = raw["id"].str.replace(r"-[0-9]+$", "", regex=True)
    arrays = samples["test"]
    for i in range(0, len(arrays["y"]), max(1, len(arrays["y"]) // 5)):
        t0 = pd.Timestamp(str(arrays["t0"][i]))
        assert isinstance(t0, pd.Timestamp)
        for b in range(8):
            n, mean_delay = _due_by_hand(raw, str(arrays["eva"][i]), t0, b)
            assert due[i, 0, b] == n
            assert due[i, 1, b] == pytest.approx(mean_delay, abs=1e-5)
