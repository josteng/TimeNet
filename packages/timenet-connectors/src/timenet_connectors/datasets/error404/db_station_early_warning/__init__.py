"""Early warning of train disruptions at German stations, from the DB Timetables scrape."""

from timenet_connectors.datasets.error404.db_station_early_warning.connector import (
    CONNECTOR as CONNECTOR,
    DbStationEarlyWarningConnector as DbStationEarlyWarningConnector,
)
