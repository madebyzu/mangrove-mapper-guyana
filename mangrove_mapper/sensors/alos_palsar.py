import ee

from ..utils.refined_lee import refined_lee
from ..utils.to_db import to_db
from .base import SensorProcessor


class ALOSPalsar(SensorProcessor):
    """
    ALOS-PALSAR Advanced Land Observing Satellite (ALOS) Phased Array type L-band Synthetic Aperture Radar (PALSAR)
    processing pipeline
    ('The data used has been provided by the Japan Aerospace Exploration Agency')

    This dataset produces a global SAR image mosaic with no gap-filling from previous years. It has also been
    ortho-rectified and slope corrected. The units for the polarization bands (HH/HV) are digital numbers.

    See Also
    --------
    docs/references.md#documentation : Global PALSAR-2/PALSAR Yearly Mosaic, version 2.5.0
    docs/references.md#journal-articles--papers : Shimada et al., 2014
    """
    @property
    def sensor_key(self):
        return 'alos_palsar'

    @property
    def indices_key(self):
        return 'sar_indices'

    @property
    def prefix(self):
        return 'alos'

    def process(self):
        """
        Runs the full sensor processing pipeline.

        Overrides the implementation in SensorProcessor because backscatter coefficients need to
        be converted from linear scale to decibels after indices computation.

        Returns
        -------
        image_with_indices : ee.Image
            Processed composite image with spectral bands and computed indices.
        """
        image_collection = self.load()
        processed = self.preprocess(image_collection)
        composite = self.composite(processed)
        image_with_indices = self.compute_indices(composite)
        db_bands = to_db(image_with_indices.select(['HV', 'HH']))
        final_image = image_with_indices.addBands(db_bands, overwrite=True)
        return final_image.select(self.config['bands']['alos_palsar'])

    def composite(self, image_collection):
        """
        Takes the first image from the collection, which is the annual mosaic, and clips it to the AOI.

        Parameters
        ----------
        image_collection : ee.ImageCollection
            A collection of images filtered by date and AOI.

        Returns
        -------
        ee.Image
            Annual composite clipped to the AOI with polarization bands
        """

        alos_palsar_composite = image_collection.first().clip(self.aoi_polygon)
        return alos_palsar_composite

    def load(self):
        """
        Instantiates an ALOS-PALSAR image collection in digital number units
        for the date range, aoi specified, and co-polarization/cross-polarization bands

        Returns
        -------
        ee.ImageCollection
            A collection of ALOS-PALSAR images filtered by date and AOI with polarization bands
        """
        alos_palsar = (
            ee.ImageCollection(
                "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi_polygon)
            .select(['HH', 'HV'])
        )
        return alos_palsar

    def preprocess(self, image_collection):
        """
        Converts to linear scale and applies refined lee speckle filtering.

        Parameters
        ----------
        image_collection : ee.ImageCollection
            ALOS-PALSAR collection in digital number units

        Returns
        -------
        ee.ImageCollection
            Filtered ALOS-PALSAR collection in linear scale
        """
        images_linear = image_collection.map(self._to_linear)
        if self.config['preprocessing']['alos_palsar']['speckle_filter']:
            return images_linear.map(refined_lee)
        return images_linear

    def _to_linear(self, image):
        """
        Converts bands in raw digital numbers to linear scale

        Deduced from the DN to dB formula and linear to dB formula
        (see docs/references.md#documentation)

        Parameters
        ----------
        image: ee.Image
            SAR image with bands as digital numbers

        Returns
        -------
        ee.Image
            Image with bands converted to linear scale

        See Also
        --------
        docs/references.md#documentation : Global PALSAR-2/PALSAR Yearly Mosaic, version 2.5.0
        """
        return image.pow(2).multiply(10**(-8.3))
