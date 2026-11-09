import ee

from .base import SensorProcessor


class Sentinel2(SensorProcessor):
    """
    Sentinel-2 processing pipeline
    ('Copernicus Sentinel data 2026' for Sentinel data)

    Two processing levels are available, selected with `level`:

    - 'SR' (default) Level-2A surface reflectarnce, atmospherically corrected. The
      archive begins 2017-03-28 and non-European coverage is sparse until late 2018.
    - 'TOA' Level-1C top-of-atmosphere, available from 2015-06. Not
      atmospherically corrected.

    Preprocessing such as cloud/shadow masking will be needed to produce a clean satellite image
    from which a classifier can learn.

    The names, pixel size, wavelength and description of each of the Sentinel 2 bands can be found
    in the Google Earth Engine Documentation guide (see link at /docs/references.md#documentation).

    See Also
    --------
    docs/references.md#documentation : Harmonized Sentinel-2 MSI: MultiSpectral Instrument Level-2A
    docs/references.md#documentation : Harmonized Sentinel-2 MSI: MultiSpectral Instrument Level-1C
    """
    SCALE_FACTOR = 10000  # definitions found in documentation
    # experimented in sandbox/notebooks to determine this value as best for the aoi
    MAX_CLOUD_PROBABILITY = 50

    COLLECTIONS = {
        'SR': 'COPERNICUS/S2_SR_HARMONIZED',   # Level-2A, atmospherically corrected
        'TOA': 'COPERNICUS/S2_HARMONIZED',     # Level-1C, top-of-atmosphere
    }

    def __init__(self, start_date=None, end_date=None, aoi_polygon=None,
                 project=None, level='SR'):
        """
        Parameters
        ----------
        level : str
            'SR' (default) for Level-2A surface reflectance, or 'TOA' for Level-1C.
            Defaults to SR so existing callers are unaffected.

        Raises
        ------
        ValueError
            If `level` is not 'SR' or 'TOA'.
        """
        super().__init__(start_date, end_date, aoi_polygon, project)
        if level not in self.COLLECTIONS:
            raise ValueError(
                f"level must be one of {sorted(self.COLLECTIONS)}, got {level}")
        self.level = level

    @property
    def collection_id(self):
        return self.COLLECTIONS[self.level]

    @property
    def sensor_key(self):
        return 'sentinel2'

    @property
    def indices_key(self):
        return 'optical_indices'

    @property
    def prefix(self):
        return 's2'

    def composite(self, image_collection):
        """
        Reduces the image collection to a single annual median composite.
        Clips the image to the AOI and selects only the bands specified in config.yml.

        Parameters
        ----------
        image_collection : ee.ImageCollection
            A collection of images filtered by date and AOI.

        Returns
        -------
        ee.Image
            Annual median composite clipped to the AOI.

        See Also
        --------
        docs/references.md#documentation : Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A
        """
        bands = self.config['bands']['sentinel2']
        return image_collection.median().clip(self.aoi_polygon).select(bands)

    def load(self):
        """
        Instantiates and joins the Sentinel-2 SR and Sentinel2 Cloud Probability
        collections for the date range and aoi specified.

        Returns
        -------
        ee.ImageCollection
            A combined collection of Sentinel-2 SR and Sentinel Cloud Probability images filtered by date and AOI

        """
        s2_sr = (
            ee.ImageCollection(self.collection_id)
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi_polygon)
        )

        s2_cloud = (
            ee.ImageCollection(
                "COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi_polygon)
        )

        s2_sr_with_cloud_mask = ee.Join.saveFirst('cloud_mask').apply(
            primary=s2_sr,
            secondary=s2_cloud,
            condition=ee.Filter.equals(
                leftField='system:index', rightField='system:index'
            ),
        )
        s2_sr_with_cloud_mask = ee.ImageCollection(s2_sr_with_cloud_mask)

        return s2_sr_with_cloud_mask

    def _mask_clouds(self, image):
        """
        Creates a cloud mask based on a max threshold of a pixel's cloud probability.
        All pixels above the threshold would be classified as clouds.

        Refer to 'Cloud Probability' in docs/references.md for more information.

        Parameters
        ----------
        image : ee.Image
            A single satellite image.

        Returns
        -------
        ee.Image
            A single image with cloud mask based on cloud probability
        """
        clouds = ee.Image(image.get('cloud_mask')).select('probability')
        is_not_cloud = clouds.lt(self.MAX_CLOUD_PROBABILITY)
        return image.updateMask(is_not_cloud)

    def _mask_edges(self, s2_image):
        """
        10m bands sometimes contain invalid data scene edges.
        These edge artifacts are masked by intersecting the image with 20m and 60m bands (B8A and B9)

        Parameters
        ----------
        s2_image : ee.Image
            A single Sentinel-2 image

        Returns
        -------
        ee.Image
            Sentinel-2 image with the edge mask applied
        """
        return s2_image.updateMask(
            s2_image.select('B8A').mask().updateMask(
                s2_image.select('B9').mask())
        )

    def preprocess(self, image_collection):
        """
        Applies edge and cloud masks, and then scales the bands of the images.

        Parameters
        ----------
        image_collection : ee.ImageCollection
            Combined Sentinel-2 SR and Cloud Probability datasets

        Returns
        -------
        ee.ImageCollection
            Cloudless processed Sentinel-2 data

        See Also
        --------
        docs/references.md#documentation : Sentinel-2: Cloud Probability
        """
        s2_cloud_masked = image_collection.map(
            self._mask_edges).map(self._mask_clouds)
        s2_processed = s2_cloud_masked.map(
            lambda image: image.divide(self.SCALE_FACTOR))

        return s2_processed
