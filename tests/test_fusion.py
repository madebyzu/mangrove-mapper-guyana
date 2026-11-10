"""
Unit tests for the fusion pipeline (band prefixing, sensor stacking, target labelling)
and the MangroveMapper export orchestration, not dependent on the Google Earth Engine (GEE) api.
As a result, GEE objects are mocked out.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from mangrove_mapper.main import MangroveMapper
from mangrove_mapper.utils.fusion import _prefix_bands, concat_sensors, load_target

with open(Path(__file__).parent.parent / "mangrove_mapper/configs/config.yml") as f:
    CONFIG = yaml.safe_load(f)


@pytest.fixture
def mock_fusion_ee():
    with patch("mangrove_mapper.utils.fusion.ee") as mock:
        yield mock


class TestPrefixBands:
    def test_renames_image_with_mapped_names(self, mock_fusion_ee):
        image = MagicMock()
        mapped_names = image.bandNames.return_value.map.return_value

        _prefix_bands(image, "s2")

        image.rename.assert_called_once_with(mapped_names)

    def test_names_are_built_server_side(self, mock_fusion_ee):
        image = MagicMock()

        _prefix_bands(image, "s2")

        map_fn = image.bandNames.return_value.map.call_args[0][0]
        result = map_fn("B2")

        # Both prefix and band name must be wrapped in ee.String so the
        # concatenation happens server-side (a client-side f-string would
        # stringify the placeholder object instead of the band name)
        assert mock_fusion_ee.String.call_args_list == [call("s2"), call("B2")]
        expected = mock_fusion_ee.String.return_value.cat.return_value.cat.return_value
        assert result == expected


class TestConcatSensors:
    def test_processes_and_prefixes_every_sensor(self, mock_fusion_ee):
        sensors = []
        for prefix in ["l8", "s1", "s2", "alos"]:
            sensor = MagicMock()
            sensor.prefix = prefix
            sensors.append(sensor)

        concat_sensors(sensors)

        for sensor in sensors:
            sensor.process.assert_called_once()
            sensor.process.return_value.rename.assert_called_once()
        mock_fusion_ee.Image.cat.assert_called_once()
        (stacked_images,) = mock_fusion_ee.Image.cat.call_args[0]
        assert len(stacked_images) == len(sensors)


class TestLoadTarget:
    TARGET = CONFIG["target"]

    def test_builds_binary_label_with_water_masked(self, mock_fusion_ee):
        aoi = MagicMock()
        dataset = (
            mock_fusion_ee.ImageCollection.return_value
            .filterBounds.return_value.mosaic.return_value.clip.return_value
        )
        gsw = mock_fusion_ee.Image.return_value
        occurrence = gsw.select.return_value.clip.return_value.unmask.return_value

        label = load_target(
            self.TARGET["dataset_id"], aoi,
            self.TARGET["mangrove_class"], self.TARGET["permanent_water_threshold"],
        )

        mock_fusion_ee.ImageCollection.assert_called_once_with(
            self.TARGET["dataset_id"])
        mock_fusion_ee.ImageCollection.return_value.filterBounds.assert_called_once_with(
            aoi)
        mock_fusion_ee.ImageCollection.return_value.filterBounds.return_value.mosaic.return_value \
            .clip.assert_called_once_with(aoi)

        # The water mask comes from JRC GSW 'occurrence', not from a class in
        # the target dataset. Land is unmasked to 0% so it survives the cut.
        mock_fusion_ee.Image.assert_called_once_with(
            "JRC/GSW1_4/GlobalSurfaceWater")
        gsw.select.assert_called_once_with("occurrence")
        gsw.select.return_value.clip.assert_called_once_with(aoi)
        gsw.select.return_value.clip.return_value.unmask.assert_called_once_with(
            0)

        # lt(), not gte(): pixels BELOW the threshold are kept, so a threshold
        # of 100 masks only always-water and leaves intertidal mudflat in the
        # non-mangrove domain
        occurrence.lt.assert_called_once_with(
            self.TARGET["permanent_water_threshold"])

        dataset.eq.assert_called_once_with(self.TARGET["mangrove_class"])
        dataset.eq.return_value.rename.assert_called_once_with("label")

        renamed = dataset.eq.return_value.rename.return_value
        renamed.updateMask.assert_called_once_with(occurrence.lt.return_value)
        assert label == renamed.updateMask.return_value

    def test_label_is_never_resampled_or_cast(self, mock_fusion_ee):
        label = load_target(
            self.TARGET["dataset_id"], MagicMock(),
            self.TARGET["mangrove_class"], self.TARGET["permanent_water_threshold"],
        )

        # The label is categorical: resampling or float-casting it would
        # corrupt the classes stratifiedSample depends on
        label.resample.assert_not_called()
        label.toFloat.assert_not_called()


@pytest.fixture
def mock_main_ee():
    with patch("mangrove_mapper.main.ee") as mock:
        yield mock


@pytest.fixture
def mapper(mock_main_ee):
    return MangroveMapper(project_name="test-project")


@pytest.fixture
def mock_pipeline():
    with patch("mangrove_mapper.main.concat_sensors") as concat, \
            patch("mangrove_mapper.main.load_target") as target, \
            patch("mangrove_mapper.main.download_from_drive") as download, \
            patch("mangrove_mapper.main.poll_task") as poll, \
            patch("mangrove_mapper.main.Landsat8"), \
            patch("mangrove_mapper.main.Sentinel1"), \
            patch("mangrove_mapper.main.Sentinel2"), \
            patch("mangrove_mapper.main.ALOSPalsar"):
        yield {"concat": concat, "target": target,
               "download": download, "poll": poll}


class TestMangroveMapperInit:
    def test_defaults_date_range(self, mapper):
        # get_sensor_data forwards these to every sensor constructor;
        # None is fine (sensors default from config) but they must exist
        assert mapper.start_date == CONFIG['default_date_range']['start_date']
        assert mapper.end_date == CONFIG['default_date_range']['end_date']

    def test_defaults_aoi_from_config(self, mapper, mock_main_ee):
        mock_main_ee.Geometry.Polygon.assert_called_once_with(
            CONFIG["default_aoi"])


class TestGetSensorData:
    def test_label_added_after_float_cast(self, mapper, mock_pipeline):
        mapper.get_sensor_data("test_export")

        stacked = mock_pipeline["concat"].return_value
        # Features are cast to float first, then the label is appended,
        # so the label band stays integer for stratifiedSample
        stacked.toFloat.assert_called_once()
        stacked.toFloat.return_value.addBands.assert_called_once_with(
            mock_pipeline["target"].return_value)
        mock_pipeline["target"].return_value.toFloat.assert_not_called()

    def test_two_step_sample_uses_export_config(self, mapper, mock_pipeline):
        mapper.get_sensor_data("test_export")

        export = CONFIG["export"]
        # Step 1: points are stratified on the cheap label image, not the
        # heavy sensor stack
        points = mock_pipeline["target"].return_value.stratifiedSample
        points.assert_called_once_with(
            numPoints=export["num_sample_points"],
            classBand=export["target_name"],
            region=mapper.aoi_polygon,
            scale=export["scale"],
            seed=export["seed"],
            tileScale=export["tile_scale"],
            geometries=export["geometries"],
        )
        # Step 2: the stack is evaluated only at those points
        final_image = (
            mock_pipeline["concat"].return_value
            .toFloat.return_value.addBands.return_value
        )
        final_image.sampleRegions.assert_called_once_with(
            collection=points.return_value,
            scale=export["scale"],
            properties=[export["target_name"]],
            tileScale=export["tile_scale"],
            geometries=export["geometries"],
        )

    def test_target_dataset_defaults_to_config(self, mapper, mock_pipeline):
        mapper.get_sensor_data("test_export")

        assert mock_pipeline["target"].call_args[0][0] == \
            CONFIG["target"]["dataset_id"]

    def test_target_dataset_override_used(self, mapper, mock_pipeline):
        mapper.get_sensor_data(
            "test_export", target_dataset="ESA/WorldCover/v200")

        assert mock_pipeline["target"].call_args[0][0] == "ESA/WorldCover/v200"

    def test_polls_without_downloading_when_disabled(self, mapper, mock_pipeline):
        task = mapper.get_sensor_data("test_export", retrieve_from_drive=False)

        # retrieve_from_drive=False watches the export to completion but leaves
        # the file in Drive: poll the task, skip the local download
        mock_pipeline["poll"].assert_called_once_with(task)
        mock_pipeline["download"].assert_not_called()

    def test_downloads_without_separate_poll_when_enabled(self, mapper, mock_pipeline):
        mapper.get_sensor_data("test_export", retrieve_from_drive=True)

        # download_from_drive polls internally
        mock_pipeline["download"].assert_called_once()
        mock_pipeline["poll"].assert_not_called()

    def test_export_task_started_and_downloaded(
            self, mapper, mock_pipeline, mock_main_ee):
        mapper.get_sensor_data(
            "test_export", file_format="CSV", folder="test_folder")

        to_drive = mock_main_ee.batch.Export.table.toDrive
        kwargs = to_drive.call_args.kwargs
        assert kwargs["description"] == "test_export"
        assert kwargs["fileFormat"] == "CSV"
        assert kwargs["folder"] == "test_folder"
        to_drive.return_value.start.assert_called_once()
        mock_pipeline["download"].assert_called_once_with(
            "test_export", "CSV", task=to_drive.return_value)


@pytest.fixture
def mock_interpreter():
    # geemap/ipywidgets are notebook extras; stand in for the interpreter module
    module = MagicMock(LABEL_COL="planet_label", ZOOM=16)
    with patch.dict("sys.modules", {"mangrove_mapper.validation.interpreter": module}):
        yield module


class TestRunInterpretationWidget:
    def test_basemaps_default_to_config(self, mapper, mock_interpreter):
        mapper.run_interpretation_widget("region_2", "plots.csv")

        args = mock_interpreter.build_interactive_map.call_args
        assert args[0][0] == CONFIG["validation"]["collection_path"]

    def test_basemaps_override(self, mapper, mock_interpreter):
        mapper.run_interpretation_widget("region_2", "plots.csv",
                                         collection_path="projects/me/assets/basemaps")

        args = mock_interpreter.build_interactive_map.call_args
        assert args[0][0] == "projects/me/assets/basemaps"


@pytest.fixture
def mock_region_ids():
    with patch("mangrove_mapper.main.ee_region_ids") as mock:
        mock.return_value = (MagicMock(), MagicMock())
        yield mock


class TestGetSensorDataRegionBalanced:
    def test_requires_n_total(self, mapper, mock_pipeline, mock_region_ids):
        with pytest.raises(ValueError, match="n_total"):
            mapper.get_sensor_data("test_export", balance_by_region=True)

    @pytest.mark.parametrize("n_total", [0, -5, 20_000.0])
    def test_rejects_invalid_n_total(self, mapper, mock_pipeline, mock_region_ids, n_total):
        with pytest.raises(ValueError, match="n_total"):
            mapper.get_sensor_data(
                "test_export", balance_by_region=True, n_total=n_total)

    def test_combined_class_is_region_times_two_plus_label(
            self, mapper, mock_pipeline, mock_region_ids):
        mapper.get_sensor_data(
            "test_export", balance_by_region=True, n_total=20_000)

        rid = mock_region_ids.return_value[1]
        target = mock_pipeline["target"].return_value
        rid.multiply.assert_called_once_with(2)
        rid.multiply.return_value.add.assert_called_once_with(target)
        rid.multiply.return_value.add.return_value.rename.assert_called_once_with(
            "cls")

    def test_samples_every_region_class_cell_equally(
            self, mapper, mock_pipeline, mock_region_ids):
        mapper.get_sensor_data(
            "test_export", balance_by_region=True, n_total=20_000)

        export = CONFIG["export"]
        cls = (mock_region_ids.return_value[1].multiply.return_value.add.return_value
               .rename.return_value.clip.return_value)
        # 6 regions x 2 classes = 12 cells, ceil(20,000 / 12) = 1,667 each
        cls.stratifiedSample.assert_called_once_with(
            numPoints=0,
            classBand="cls",
            classValues=list(range(12)),
            classPoints=[1667] * 12,
            region=mapper.aoi_polygon,
            scale=export["scale"],
            seed=export["seed"],
            tileScale=export["tile_scale"],
            geometries=export["geometries"],
        )
        # the label-only stratification is not used
        mock_pipeline["target"].return_value.stratifiedSample.assert_not_called()

    def test_label_read_from_stack_not_point_properties(
            self, mapper, mock_pipeline, mock_region_ids):
        mapper.get_sensor_data(
            "test_export", balance_by_region=True, n_total=20_000)

        final_image = (mock_pipeline["concat"].return_value
                       .toFloat.return_value.addBands.return_value)
        # the points only carry 'cls'; the label comes from the stack's label band
        assert final_image.sampleRegions.call_args.kwargs["properties"] == []

    def test_default_export_does_not_build_regions(
            self, mapper, mock_pipeline, mock_region_ids):
        mapper.get_sensor_data("test_export")

        mock_region_ids.assert_not_called()
