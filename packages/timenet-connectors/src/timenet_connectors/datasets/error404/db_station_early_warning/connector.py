"""Early warning of train disruptions at German stations, from the DB Timetables scrape.

Source: ``piebro/deutsche-bahn-data`` on the HuggingFace Hub, one row per train per stop, scraped
from Deutsche Bahn's Timetables API (CC BY 4.0). The connector reads twelve monthly files
(November 2025 to October 2026) at a pinned revision and aggregates them into 15-minute bins for
the 400 busiest stations, the four stations that feed most trains into each, and the network.

Each sample is a station at a moment t0 when it runs normally: in the last two hours at most its
usual 75th percentile of arrivals were late (6 minutes or more) or cancelled. The question is
whether a disruption starts within the next two hours, meaning the share of late or cancelled
arrivals in ``[t0, t0 + 2 h)`` reaches the station's own 90th percentile (at least 20 %).
Percentiles come from the training months only, and every input lies before t0.

One Record per sample, with the time origin seven days before t0:

* ``station``: mean arrival delay, late share, cancelled share and planned arrivals over the 24 h
  before t0 in 15-minute bins, and the mean arrival delay hourly over the 7 days before t0.
* ``feeders``: mean arrival delay at each feeder station over the 24 h before t0 (up to four).
* ``network``: mean arrival delay over all 400 stations over the 24 h before t0.
* ``incoming``: the trains due at the station in [t0, t0 + 2 h), per 15-minute bin of their planned
  arrival, as known at t0: how many are due, their mean and largest delay at the last stop they
  left before t0, the share not started yet, and the delay they gained over their last two stops.
  The axis places each bin at its planned arrival time; every value was known at t0.

Two tasks per Record share the station context as their prompt: a ClassificationTask
(``disruption`` / ``no_disruption``) and a ScalarPredictionTask (the share of late or cancelled
arrivals in the next two hours). Splits are by time, and one station in five is held out of
training entirely: train is November 2025 to June 2026, validation July 2026, test August to
September 2026 (``cell: test``) and October 2026 (``cell: oot``, a later out-of-time check).
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from fractions import Fraction
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import numpy as np

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.json import JsonValue
from timenet.types import ClassificationTask, Label, LabelSet, ScalarPredictionTask, Split, TimeSeriesSpec, ureg
from timenet.types.clock import TimeOrigin
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.error404.db_station_early_warning.pipeline import (
    QUESTION,
    REF_MONTH,
    aggregate,
    context_text,
    cut_samples,
    incoming,
)


MONTHS = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
BERLIN = ZoneInfo("Europe/Berlin")
"""The source publishes local times in Berlin."""
EVERY_15_MIN = Fraction(1, 900)
HOURLY = Fraction(1, 3600)
DAY_US = 86_400_000_000
SPLIT_OF: Mapping[str, tuple[Split, str]] = {
    "train": (Split.TRAIN, "train"),
    "val": (Split.VALIDATION, "validation"),
    "test": (Split.TEST, "test"),
    "oot": (Split.TEST, "oot"),
}
DISRUPTION = LabelSet(
    name="disruption_within_2h",
    labels=(
        Label(
            name="disruption",
            description="Within two hours, the share of late or cancelled arrivals reaches the station's 90th percentile.",
        ),
        Label(name="no_disruption"),
    ),
)
DELAY = TimeSeriesSpec(spec_type="arrival_delay", name="Mean arrival delay", unit_value=ureg.minute)
LATE = TimeSeriesSpec(spec_type="late_share", name="Share of arrivals 6+ minutes late", unit_value=ureg.dimensionless)
CANCELLED = TimeSeriesSpec(
    spec_type="cancelled_share", name="Share of arrivals cancelled", unit_value=ureg.dimensionless
)
ARRIVALS = TimeSeriesSpec(spec_type="planned_arrivals", name="Planned arrivals", unit_value=ureg.dimensionless)
NOT_STARTED = TimeSeriesSpec(
    spec_type="not_started_share", name="Share of due trains not started", unit_value=ureg.dimensionless
)
INCOMING_SIGNALS = (
    ("due_trains_15min", ARRIVALS),
    ("due_trains_mean_delay_15min", DELAY),
    ("due_trains_max_delay_15min", DELAY),
    ("due_trains_not_started_share_15min", NOT_STARTED),
    ("due_trains_delay_gained_15min", DELAY),
)


def _axis(rate: Fraction, offset_days: int) -> RegularAxis:
    return RegularAxis.from_rate_hz(rate, offset_us=offset_days * DAY_US)


def _record(sample: Mapping[str, Any]) -> Record:
    """One Record: 7 days hourly, then the last 24 h in 15-minute bins (offset 6 days)."""
    t0 = datetime.fromisoformat(str(sample["t0"])).replace(tzinfo=BERLIN)
    ctx = sample["ctx"]
    day = _axis(EVERY_15_MIN, 6)
    station = Source(
        name="station",
        signals=(
            Signal.from_values(ctx[0], spec=DELAY, name="arrival_delay_15min", time_axis=day),
            Signal.from_values(ctx[1], spec=LATE, name="late_share_15min", time_axis=day),
            Signal.from_values(ctx[2], spec=CANCELLED, name="cancelled_share_15min", time_axis=day),
            Signal.from_values(ctx[3], spec=ARRIVALS, name="planned_arrivals_15min", time_axis=day),
            Signal.from_values(sample["week"], spec=DELAY, name="arrival_delay_hourly", time_axis=_axis(HOURLY, 0)),
        ),
    )
    feeder_evas = [str(e) for e in sample["feeder_eva"]]
    feeders = Source(
        name="feeders",
        signals=tuple(
            Signal.from_values(ctx[4 + k], spec=DELAY, name=f"feeder{k + 1}_arrival_delay_15min", time_axis=day)
            for k, eva in enumerate(feeder_evas)
            if eva
        ),
    )
    network = Source(
        name="network",
        signals=(Signal.from_values(ctx[8], spec=DELAY, name="network_arrival_delay_15min", time_axis=day),),
    )
    due = _axis(EVERY_15_MIN, 7)
    incoming_source = Source(
        name="incoming",
        signals=tuple(
            Signal.from_values(sample["inc"][k], spec=spec, name=name, time_axis=due)
            for k, (name, spec) in enumerate(INCOMING_SIGNALS)
        ),
        metadata={"known_at": "t0", "axis": "planned arrival bin of the trains described"},
    )
    sources = (station, feeders, network) if feeders.signals else (station, network)
    sources = (*sources, incoming_source)
    return Record(
        start_time=TimeOrigin(t0 - timedelta(days=7)),
        sources=sources,
        metadata={
            "station_eva": str(sample["eva"]),
            "station_name": str(sample["name"]),
            "t0": str(sample["t0"]),
            "feeder_evas": [e for e in feeder_evas if e],
        },
    )


class DbStationEarlyWarningConnector(BaseHuggingFaceConnector):
    """Connector for station early-warning samples built from ``piebro/deutsche-bahn-data``."""

    HF_REPO = "piebro/deutsche-bahn-data"
    SOURCE_REVISION = "00434ac0880994890c80ad1f3ebf0e5ab0985d5a"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {m: f"monthly_processed_data/data-{m}.parquet" for m in MONTHS}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Aggregate the monthly files, cut the samples and add one Record and two Tasks per sample.

        Args:
            raw_refs: The monthly files from :meth:`download`.

        Returns:
            The populated dataset.
        """
        paths = self.ref_paths(raw_refs, *MONTHS)
        tables = aggregate([paths[m] for m in MONTHS], paths[REF_MONTH])
        splits, _ = cut_samples(tables, seed=0, log=lambda _msg: None)
        due = incoming([paths[m] for m in MONTHS], splits)
        for name, arrays in splits.items():
            arrays["inc"] = due[name]
        dataset = TimeFDataset(metadata=self.metadata())
        for name, arrays in splits.items():
            split, cell = SPLIT_OF[name]
            for i in range(len(arrays["y"])):
                sample = {key: values[i] for key, values in arrays.items()}
                record = dataset.add_record(record=_record(sample))
                prompt = (
                    context_text(sample["name"], sample["dow"], sample["t0"], sample["cur_share"], sample["thr"])
                    + " "
                    + QUESTION
                )
                meta: dict[str, JsonValue] = {
                    "cell": cell,
                    "unseen_station": bool(sample["unseen"]),
                    "threshold_share": float(sample["thr"]),
                    "current_share": float(sample["cur_share"]),
                }
                label = "disruption" if int(sample["y"]) == 1 else "no_disruption"
                dataset.add_task(
                    task=ClassificationTask(
                        inputs=(record,),
                        targets=(label,),
                        label_set=DISRUPTION,
                        prompt=prompt,
                        split=split,
                        metadata=meta,
                    )
                )
                dataset.add_task(
                    task=ScalarPredictionTask(
                        inputs=(record,),
                        targets=(float(np.round(sample["y_share"], 6)),),
                        unit=ureg.dimensionless,
                        target_name="late_or_cancelled_share_next_2h",
                        prompt=context_text(
                            sample["name"], sample["dow"], sample["t0"], sample["cur_share"], sample["thr"]
                        )
                        + " What share of arrivals will be late or cancelled in the next two hours?",
                        split=split,
                        metadata=meta,
                    )
                )
        return dataset


CONNECTOR = DbStationEarlyWarningConnector
