import math
from pathlib import Path

import ee
import yaml
from pyproj import Transformer

from .sensors import ALOSPalsar, Landsat8, Sentinel1, Sentinel2
from .utils.download import download_from_drive, download_prefix, poll_task
from .utils.fusion import concat_sensors, load_target


class MangroveMapper:
    """
    The main class responsible for retrieving data, training models, and generating results.
    Authentication in GEE takes place here as well as instantiation of config values.

    Parameters
    ----------
    project_name : str
        Project name defined by user in their Google Cloud account
    aoi : list[int]
        Area of interest in a 2D list of coordinates
    """

    def __init__(self, project_name=None, aoi=None):
        try:
            ee.Initialize(project=project_name)
        except ee.EEException:
            print("Not logged in. Please authenticate below")
            ee.Authenticate()
            ee.Initialize(project=project_name)

        with open(Path(__file__).parent / "configs/config.yml") as f:
            self.config = yaml.safe_load(f)

        if aoi is None:
            aoi = self.config['default_aoi']

        self.aoi_polygon = ee.Geometry.Polygon(aoi)
        self.project_name = project_name
        self.start_date = self.config['default_date_range']['start_date']
        self.end_date = self.config['default_date_range']['end_date']

    def get_info():
        """
        TODO: potentially
        """

    def get_sensor_data(
            self, file_name, file_format='CSV',
            folder='MangroveMapper_ML_Exports', retrieve_from_drive=True,
            target_dataset=None, mangrove_class=None, presence_only=False,
            s2_level='SR', s1_orbits=Sentinel1.ORBITS):
        """
        Retrieves multi-sensor data to Google Drive with the option to download to
        local drive.

        Parameters
        ----------
        file_name : str
            Name of output file
        file_format : str
            File type (CSV, GeoTIFF, TFRecord, GeoJSON, SHP etc.)
        folder : str
            Folder name where export will live in Google Drive
        retrieve_from_drive : bool
            Option to download to local hard drive automatically
        target_dataset : str, optional
            GEE dataset id for the target labels. Defaults to the
            dataset_id in config.yml if not provided.
        mangrove_class : int
                Pixel value denoting mangrove (95 for ESA WorldCover 'Map', 1 for
                Global Mangrove Watch 'b1').
        presence_only : bool
            Set True for presence-only targets that mark mangrove and leave
            everything else masked (e.g. Global Mangrove Watch). Otherwise, leave
            False for WorldCover target. Default False.
        s2_level : str
            Sentinel-2 processing level, 'SR' (default) or 'TOA'.
        s1_orbits : tuple of str
            Sentinel-1 orbit passes to build backscatter bands. Defaults to both.

        Returns
        -------
        ee.batch.Task
            The export task object
        """
        target_dataset = target_dataset or self.config['target']['dataset_id']
        mangrove_class = mangrove_class or self.config['target']['mangrove_class']
        sensors = [
            Landsat8(self.start_date, self.end_date,
                     self.aoi_polygon, self.project_name),
            Sentinel1(self.start_date, self.end_date,
                      self.aoi_polygon, self.project_name, orbits=s1_orbits),
            Sentinel2(self.start_date, self.end_date,
                      self.aoi_polygon, self.project_name, level=s2_level),
            ALOSPalsar(self.start_date, self.end_date,
                       self.aoi_polygon, self.project_name)
        ]
        stacked_image = concat_sensors(sensors).toFloat()
        target = load_target(target_dataset, self.aoi_polygon,
                             mangrove_class,
                             self.config['target']['permanent_water_threshold'],
                             presence_only=presence_only)
        stacked_image = stacked_image.addBands(target)
        export_settings = self.config['export']

        points = target.stratifiedSample(
            numPoints=export_settings['num_sample_points'],
            classBand=export_settings['target_name'],
            region=self.aoi_polygon,
            scale=export_settings['scale'],
            seed=export_settings['seed'],
            tileScale=export_settings['tile_scale'],
            geometries=export_settings['geometries'])

        training_samples = stacked_image.sampleRegions(
            collection=points,
            scale=export_settings['scale'],
            properties=[export_settings['target_name']],
            tileScale=export_settings['tile_scale'],
            geometries=export_settings['geometries']
        )
        task = ee.batch.Export.table.toDrive(
            collection=training_samples,
            description=file_name,
            folder=folder,
            fileFormat=file_format
        )
        task.start()
        print("Stratified sample export started!")
        print(f"Task ID: {task.id}")

        if retrieve_from_drive:
            download_from_drive(file_name, file_format, task=task)
        else:
            poll_task(task)
        return task

    def pinned_transform(self, crs='EPSG:32621', scale=10, pad=500):
        """
        Fixed affine transform that pins every tiled export to one grid.

        Passing only `scale` lets Earth Engine choose its own grid origin per
        export, so two exports can disagree by a fraction of a pixel and fail to
        mosaic without resampling. Pinning `crsTransform` to the value returned here
        fixes one origin for every export -- both years, all tiles, and any derived
        product -- so mosaicking becomes a straight copy.

        Parameters
        ----------
        crs : str
            Projected CRS the grid is defined in. Default 'EPSG:32621' (UTM 21N,
            which covers the whole Guyana coast).
        scale : int
            Pixel size in metres. Default 10.
        pad : int
            Metres of slack around the AOI so the strip is not clipped. Default 500.

        Returns
        -------
        list of float
            [scale, 0, x0, 0, -scale, y0], the crsTransform for Export.image.toDrive.

        See Also
        --------
        docs/references.md#ai-usage : AI assistance log
        """
        coords = self.config['default_aoi']
        to_crs = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        xs, ys = to_crs.transform([c[0] for c in coords], [
                                  c[1] for c in coords])
        x0 = math.floor((min(xs) - pad) / scale) * scale
        y0 = math.ceil((max(ys) + pad) / scale) * scale
        return [scale, 0, float(x0), 0, -scale, float(y0)]

    def export_image_tiled(self, image, description, scale=None, crs_transform=None,
                           crs='EPSG:32621', aoi=None, tile=2048,
                           folder='MangroveMapper_ML_Exports',
                           retrieve_from_drive=True, output_dir=None):
        """
        Export an image over the AOI as skip-empty GeoTIFF tiles to Drive.

        The contiguous image (wall-to-wall) counterpart of `get_sensor_data` (which exports a
        sparse samples). Pass `crs_transform` (e.g. from `pinned_transform`) to lock
        every tile onto one grid so the tiles mosaic without resampling; pass
        `scale` for a layer only ever reprojected on read. Empty tiles are skipped,
        so a thin AOI in a large bounding box writes only the tiles that hold data.

        Parameters
        ----------
        image : ee.Image
            Image to export, clipped to the AOI.
        description : str
            Export name; every tile's file name starts with this.
        scale : int, optional
            Pixel size in metres. Pass this or `crs_transform`, not both.
        crs_transform : list of float, optional
            Pinned affine transform (see `pinned_transform`). Pass this or `scale`.
        crs : str
            CRS of the export. Default 'EPSG:32621' (UTM 21N).
        aoi : list or ee.Geometry, optional
            Region to export over, overriding instance variable for this call only
            (a coordinate list is wrapped in an ee.Geometry.Polygon).
            Defaults to `self.aoi_polygon`.
        tile : int
            Tile edge in pixels, a multiple of 256. Default 2048.
        folder : str
            Google Drive folder the export is written to.
        retrieve_from_drive : bool
            If True, poll to completion and download the tiles; otherwise poll and
            leave them in Drive. Default True.
        output_dir : pathlib.Path or str, optional
            Local directory to download into. Defaults to <repo>/data.

        Returns
        -------
        list of pathlib.Path or ee.batch.Task
            Downloaded tile paths when `retrieve_from_drive`, else the export task.

        Raises
        ------
        ValueError
            If both or neither of `scale` and `crs_transform` are given.

        See Also
        --------
        docs/references.md#ai-usage : AI assistance log
        """
        if (scale is None) == (crs_transform is None):
            raise ValueError("pass exactly one of scale or crs_transform")

        # default to the instance AOI; a passed coordinate list is wrapped, an
        # ee.Geometry is used as-is (can't default to self.aoi_polygon in the signature)
        if aoi is None:
            region = self.aoi_polygon
        else:
            region = aoi if isinstance(
                aoi, ee.Geometry) else ee.Geometry.Polygon(aoi)

        grid = {'scale': scale} if scale is not None else {
            'crsTransform': crs_transform}
        task = ee.batch.Export.image.toDrive(
            image=image, description=description, folder=folder,
            region=region, crs=crs, **grid,
            fileDimensions=[tile, tile], skipEmptyTiles=True, maxPixels=1e13,
            fileFormat='GeoTIFF')
        task.start()
        print(f"Tiled image export started: {description} (task {task.id})")

        if not retrieve_from_drive:
            return task
        poll_task(task)
        return (download_prefix(description) if output_dir is None
                else download_prefix(description, output_dir=output_dir))

    def run_interpretation_widget(self, aoi, accuracy_plot_path, label_column=None, zoom=None,
                                  collection_path=None):
        """
        Runs the interpretation app in a notebook, labelling accuracy plots
        against Planet imagery on a Google Earth Engine (GEE) interactive map.

        Reads the plots from `accuracy_plot_path` and writes each interpretation
        straight back to that file, saving after every click. The model's
        prediction is deliberately not shown, so the reference label stays
        independent of the map being assessed.

        Parameters
        ----------
        aoi : {'full_coast', 'region_2'}
            Predefined area to interpret over. 'region_2' is the Pomeroon river
            estuary; 'full_coast' is the whole Guyana coastline.
        accuracy_plot_path : str or pathlib.Path
            Accuracy-assessment CSV. Modified in place.
        label_column : str, optional
            Column the interpretation is written to. Defaults to 'planet_label'.
        zoom : int, optional
            Zoom level used when centring on a plot. Defaults to 16
        collection_path : str, optional
            GEE ImageCollection holding the Planet basemaps to label against.
            Defaults to validation.collection_path in config.yml. Planet's
            licence does not allow the basemaps to be shared, so point this
            at a collection in your own GEE project.

        Returns
        -------
        geemap.Map
            The interactive map with navigation and labelling buttons.

        Raises
        ------
        ValueError
            If `aoi` is not one of the predefined values.

        Notes
        -----
        The buttons only work in VS Code notebook or JupyterLab. It will not work in Jupyter in a browser.
        If interactivity still doesn't work consider installing `@jupyter-widgets/jupyterlab-manager`
        per suggestion:
        https://stackoverflow.com/questions/36351109/ipython-notebook-ipywidgets-does-not-show/52614035#52614035
        """
        # imported here rather than at module scope
        # geemap/ipywidgets is quite heavy so it's only used in interpreter.py
        from .validation.interpreter import LABEL_COL, ZOOM, build_interactive_map, label_plots_widgets

        val_config = self.config['validation']

        match aoi:
            case 'full_coast':
                aoi = self.config['default_aoi']
            case 'region_2':
                aoi = val_config['pomeroon_river_aoi']
            case _:
                raise ValueError(
                    f"area of interest must be one of ['full_coast', 'region_2'], "
                    f"got '{aoi}'")

        label_column = label_column or LABEL_COL
        zoom = zoom or ZOOM

        collection_path = collection_path or val_config['collection_path']
        m = build_interactive_map(
            collection_path, aoi, accuracy_plot_path=accuracy_plot_path)
        label_plots_widgets(
            m, accuracy_plot_path=accuracy_plot_path, label_col=label_column, zoom=zoom)

        return m
