import ee

from .base import SensorProcessor


class Landsat8(SensorProcessor):
    """
    Landsat8 processing pipeline
    (Landsat-8 image courtesy of the U.S. Geological Survey)

    The dataset in this class has been atmospherically corrected and its bands have been scaled.
    Preprocessing such as cloud/shadow masking will be needed to produce a clean satellite image
    from which a classifier can learn.

    The names, pixel size, wavelength and description of each of the Landsat 8 bands can be found
    in the USGS Landsat 8 Documentation guide (see link at /docs/references.md#documentation).

    See Also
    --------
    docs/references.md#documentation : USGS Landsat 8 Level 2, Collection 2, Tier 1.
    """
    # Constant definitions found in documentation
    SCALE_FACTOR = 2.75e-5
    OFFSET = -0.2

    @property
    def sensor_key(self):
        return 'landsat8'

    @property
    def indices_key(self):
        return 'optical_indices'

    @property
    def prefix(self):
        return 'l8'

    def _apply_scale_factors(self, image):
        """
        Applies scale factors to surface reflectance and temperature bands respectively.

        Parameters
        ----------
        image : ee.Image
            A single satellite image.

        Returns
        -------
        image : ee.Image
            A single satellite image with scale-corrected bands.

        """
        optical_bands = image.select('SR_B.').multiply(
            self.SCALE_FACTOR).add(self.OFFSET)

        return image.addBands(optical_bands, None, True)

    def load(self):
        """
        Instantiates the Landsat8 collection for the date range and area of interest (aoi) specified.

        Returns
        -------
        landsat : ee.ImageCollection
            A collection of images filtered by date and AOI.
        """
        landsat = (
            ee.ImageCollection(
                "LANDSAT/LC08/C02/T1_L2")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi_polygon)
        )
        return landsat

    def _mask_clouds(self, image):
        """
        Applies cloud and cloud shadow mask to a given image

        Relies on bit shifting to isolate the respective cloud and shadow bits,
        setting those bits zero to form a mask and then apply the mask to the image.

        Refer to QA_PIXEL bit index accessible in docs/references.md

        Parameters
        ----------
        image : ee.Image
            A single satellite image.

        Returns
        -------
        image : ee.Image
            A single cloudless satellite image.

        See also
        --------
        docs/references.md#documentation : Landsat Collection 2 quality assessment bands
        """
        # Bit 3 is Cloud Shadow, Bit 4 is Cloud
        qa = image.select('QA_PIXEL')

        # Both flags should be set to zero, indicating clear conditions
        cloud_shadow_mask = qa.bitwiseAnd(1 << 3).eq(0)
        cloud_mask = qa.bitwiseAnd(1 << 4).eq(0)
        mask = cloud_shadow_mask.And(cloud_mask)

        return image.updateMask(mask)

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
        image : ee.Image
            Annual median composite clipped to the AOI.
        """
        bands = self.config['bands']['landsat8']
        return image_collection.median().clip(self.aoi_polygon).select(bands)

    def preprocess(self, image_collection):
        """
        Combines scale factoring and cloud masking in a single preprocessing step

        Parameters
        ----------
        image_collection : ee.ImageCollection
            A collection of images.

        Returns
        -------
        image_collection : ee.ImageCollection
            A collection of cloudless, band-calibrated images.

        """
        return image_collection.map(self._apply_scale_factors).map(self._mask_clouds)
