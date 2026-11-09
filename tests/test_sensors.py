"""
Unit tests for all sensors core python functionality not dependent on Google Earth Engine (GEE) api.
As a result, GEE objects are mocked out.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mangrove_mapper.sensors import ALOSPalsar, Landsat8, Sentinel1, Sentinel2

with open(Path(__file__).parent.parent / "mangrove_mapper/configs/config.yml") as f:
    CONFIG = yaml.safe_load(f)


@pytest.fixture
def mock_ee():
    with patch("mangrove_mapper.sensors.base.ee") as mock:
        mock.Geometry.Polygon.return_value = MagicMock()
        yield mock


@pytest.fixture
def landsat8(mock_ee):
    return Landsat8()


@pytest.fixture
def sentinel2(mock_ee):
    return Sentinel2()


@pytest.fixture
def sentinel1(mock_ee):
    return Sentinel1()


@pytest.fixture
def alos_palsar(mock_ee):
    return ALOSPalsar()


# Shared tests across all sensors

@pytest.fixture(params=["landsat8", "sentinel2", "sentinel1", "alos_palsar"])
def any_sensor(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["landsat8", "sentinel2"])
def any_optical_sensor(request):
    return request.getfixturevalue(request.param)


class TestSensorDates:
    def test_defaults_when_no_dates_provided(self, any_sensor):
        assert any_sensor.start_date == CONFIG["default_date_range"]["start_date"]
        assert any_sensor.end_date == CONFIG["default_date_range"]["end_date"]

    def test_defaults_when_only_start_date_provided(self, mock_ee):
        sensor = Landsat8(start_date="2021-01-01")
        assert sensor.start_date == CONFIG["default_date_range"]["start_date"]
        assert sensor.end_date == CONFIG["default_date_range"]["end_date"]

    def test_defaults_when_only_end_date_provided(self, mock_ee):
        sensor = Landsat8(end_date="2021-12-31")
        assert sensor.start_date == CONFIG["default_date_range"]["start_date"]
        assert sensor.end_date == CONFIG["default_date_range"]["end_date"]

    def test_uses_provided_dates(self, mock_ee):
        sensor = Landsat8(start_date="2021-01-01", end_date="2021-12-31")
        assert sensor.start_date == "2021-01-01"
        assert sensor.end_date == "2021-12-31"


class TestSensorIndices:
    def test_raises_for_incomplete_index(self, any_optical_sensor):
        any_optical_sensor.indices_config["optical_indices"]["TEST"] = None
        mock_image = MagicMock()

        with pytest.raises(ValueError, match="TEST"):
            any_optical_sensor.compute_indices(mock_image)

        del any_optical_sensor.indices_config["optical_indices"]["TEST"]

    def test_all_configured_indices_added_as_bands(self, any_sensor):
        mock_image = MagicMock()
        mock_image.addBands.return_value = mock_image
        mock_image.expression.return_value.rename.return_value = mock_image

        any_sensor.compute_indices(mock_image)

        expected_count = sum(
            1 for idx in any_sensor.indices_config[any_sensor.indices_key].values()
            if idx and any_sensor.sensor_key in idx
        )
        assert mock_image.addBands.call_count == expected_count


class TestSensorPrefix:
    @pytest.mark.parametrize("sensor,expected", [
        ("landsat8", "l8"),
        ("sentinel2", "s2"),
        ("sentinel1", "s1"),
        ("alos_palsar", "alos"),
    ])
    def test_prefix_value(self, sensor, expected, request):
        assert request.getfixturevalue(sensor).prefix == expected

    def test_prefixes_are_unique(self, landsat8, sentinel2, sentinel1, alos_palsar):
        prefixes = [s.prefix for s in (
            landsat8, sentinel2, sentinel1, alos_palsar)]
        assert len(set(prefixes)) == len(prefixes)


class TestSensorBands:

    @pytest.mark.parametrize("sensor", [
        "landsat8",
        "sentinel2",
    ])
    def test_composite_selects_correct_bands(self, sensor, request):
        sensor_fixture = request.getfixturevalue(sensor)
        expected_bands = CONFIG["bands"][sensor]

        mock_collection = MagicMock()
        mock_clipped = MagicMock()
        mock_collection.median.return_value.clip.return_value = mock_clipped

        sensor_fixture.composite(mock_collection)

        mock_clipped.select.assert_called_once_with(expected_bands)


@pytest.fixture
def mock_ee_sentinel1(mock_ee):
    with patch("mangrove_mapper.sensors.sentinel1.ee") as mock:
        yield mock


@pytest.fixture
def mock_ee_alos_palsar(mock_ee):
    with patch("mangrove_mapper.sensors.alos_palsar.ee") as mock:
        yield mock


class TestSentinel1Composite:
    @pytest.fixture(autouse=True)
    def mock_refined_lee(self):
        # composite() calls the real refined_lee when the config flag is on;
        # patch it so these tests don't depend on the current config.yml value
        with patch("mangrove_mapper.sensors.sentinel1.refined_lee") as mock:
            yield mock

    def test_filters_ascending_and_descending(self, sentinel1, mock_ee_sentinel1):
        mock_collection = MagicMock()

        sentinel1.composite(mock_collection)

        assert mock_collection.filter.call_count == 2

    def test_median_computed_on_full_collection(self, sentinel1, mock_ee_sentinel1):
        mock_collection = MagicMock()

        sentinel1.composite(mock_collection)

        mock_collection.median.assert_called_once()

    def test_min_computed_on_orbit_passes(self, sentinel1, mock_ee_sentinel1):
        mock_collection = MagicMock()
        mock_orbit = MagicMock()
        mock_collection.filter.return_value = mock_orbit

        sentinel1.composite(mock_collection)

        assert mock_orbit.min.call_count == 2

    def test_speckle_filter_applied_to_full_composite(
            self, sentinel1, mock_ee_sentinel1, mock_refined_lee):
        sentinel1.config['preprocessing']['sentinel1']['speckle_filter'] = True

        sentinel1.composite(MagicMock())

        # The filter runs once on the assembled composite, and the clip
        # happens on the filtered result (filter before clip)
        mock_refined_lee.assert_called_once()
        mock_refined_lee.return_value.clip.assert_called_once_with(
            sentinel1.aoi_polygon)

    def test_speckle_filter_skipped_when_disabled(
            self, sentinel1, mock_ee_sentinel1, mock_refined_lee):
        sentinel1.config['preprocessing']['sentinel1']['speckle_filter'] = False

        sentinel1.composite(MagicMock())

        mock_refined_lee.assert_not_called()


class TestSentinel1Process:
    def test_process_selects_configured_bands(self, sentinel1, mock_ee):
        expected_bands = CONFIG["bands"]["sentinel1"]
        mock_image = MagicMock()
        mock_image.addBands.return_value = mock_image
        sentinel1.load = MagicMock(return_value=MagicMock())
        sentinel1.preprocess = MagicMock(return_value=MagicMock())
        sentinel1.composite = MagicMock(return_value=MagicMock())
        sentinel1.compute_indices = MagicMock(return_value=mock_image)

        sentinel1.process()

        mock_image.select.assert_called_with(expected_bands)


class TestALOSPalsarComposite:
    def test_first_image_called_on_collection(self, alos_palsar, mock_ee_alos_palsar):
        mock_collection = MagicMock()

        alos_palsar.composite(mock_collection)

        mock_collection.first.assert_called_once()


class TestALOSPalsarProcess:
    def test_process_selects_configured_bands(self, alos_palsar, mock_ee):
        expected_bands = CONFIG["bands"]["alos_palsar"]
        mock_image = MagicMock()
        mock_image.addBands.return_value = mock_image
        alos_palsar.load = MagicMock(return_value=MagicMock())
        alos_palsar.preprocess = MagicMock(return_value=MagicMock())
        alos_palsar.composite = MagicMock(return_value=MagicMock())
        alos_palsar.compute_indices = MagicMock(return_value=mock_image)

        alos_palsar.process()

        mock_image.select.assert_called_with(expected_bands)
