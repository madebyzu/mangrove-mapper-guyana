import ee

from ..utils.refined_lee import refined_lee
from ..utils.to_db import to_db
from .base import SensorProcessor


class Sentinel1(SensorProcessor):
    """
    Sentinel-1 processing pipeline
    ('Copernicus Sentinel data 2026' for Sentinel data)

    This dataset includes Sentinel-1 SAR: C-band Synthetic Aperture Radar (SAR) Ground Range (GRD) in linear scale.

    Preprocessing such as speckle filtering and edge masking will be needed to produce a smoothed clean satellite image
    from which a classifier can learn.

    The names, pixel size, min/max and description of each of the Sentinel 1 bands can be found
    in the Google Earth Engine Documentation guide (see link at /docs/references.md#documentation).

    See Also
    --------
    docs/references.md#documentation : Sentinel-1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected,
    log scaling
    """

    ORBITS = ('ASCENDING', 'DESCENDING')
    ORBIT_SUFFIX = {'ASCENDING': 'asc', 'DESCENDING': 'desc'}

    def __init__(self, start_date=None, end_date=None, aoi_polygon=None,
                 project=None, orbits=ORBITS):
        """
        Parameters
        ----------
        orbits : tuple of str
            Orbit passes to build separate backscatter bands. Defaults to both.
            Sentinel-1 coverage does not contain descending orbitals for certain years
            causing failures at export stage.
            Pass e.g. ``('ASCENDING',)`` to build ascending bands only.

        Raises
        ------
        ValueError
            If `orbits` is empty or names an unknown pass.
        """
        super().__init__(start_date, end_date, aoi_polygon, project)
        orbits = tuple(orbits)
        unknown = [o for o in orbits if o not in self.ORBITS]
        if unknown or not orbits:
            raise ValueError(
                f"orbits must be a non-empty subset of {self.ORBITS}, got {orbits}")
        self.orbits = orbits

    @property
    def orbit_bands(self):
        return [f'{pol_channel}_{self.ORBIT_SUFFIX[o]}'
                for o in self.orbits for pol_channel in ('VV', 'VH')]

    def available_orbits(self):
        """
        Orbit passes that actually have scenes over the AOI and date range.

        Useful check before an export: `composite` cannot build bands for an orbit with no
        scenes, and the resulting Earth Engine error ("Band pattern 'VV' was applied
        to an Image with no bands") does not say which orbit is missing.

        Returns
        -------
        dict
            Orbit pass name results in scene count.
        """
        collection = self.load()
        return {o: collection.filter(
            ee.Filter.eq('orbitProperties_pass', o)).size().getInfo()
            for o in self.ORBITS}

    @property
    def sensor_key(self):
        return 'sentinel1'

    @property
    def indices_key(self):
        return 'sar_indices'

    @property
    def prefix(self):
        return 's1'

    def process(self):
        """
        Runs the full sensor processing pipeline.

        Overrides the implementation in SensorProcessor since VV and VH bands are replaced
        by their ascending/descending orbit equivalent, though VV and VH are needed to
        compute median backscatter indices.

        Returns
        -------
        image_with_indices : ee.Image
            Processed composite image with spectral bands and computed indices.
        """
        image_collection = self.load()
        processed = self.preprocess(image_collection)
        composite = self.composite(processed)
        image_with_indices = self.compute_indices(composite)
        # mRRDI is a normalized ratio between [-1, 1] and log10(<= 0) is undefined.
        backscatter_bands = self.orbit_bands
        db_bands = to_db(image_with_indices.select(backscatter_bands))
        final_image = image_with_indices.addBands(db_bands, overwrite=True)
        # Keep the configured band order, minus any orbit bands not being built.
        all_orbit_bands = {f'{p}_{s}' for s in self.ORBIT_SUFFIX.values()
                           for p in ('VV', 'VH')}
        bands_requested = [b for b in self.config['bands']['sentinel1']
                           if b not in all_orbit_bands or b in backscatter_bands]
        return final_image.select(bands_requested)

    def composite(self, image_collection):
        """
        Reduces the image collection to a composite image with median VV/VH bands and
        min-reduced ascending/descending orbit bands. Refined Lee speckle filtering is
        applied to the full composite (all polarization bands) before clipping to the AOI.

        Parameters
        ----------
        image_collection : ee.ImageCollection
            A collection of images filtered by date and AOI.

        Returns
        -------
        ee.Image
            Annual median composite clipped to the AOI with ascending/descending
            polarization bands

        See Also
        --------
        docs/references.md#documentation : Sentinel-1 Algorithms
        docs/references.md#journal-articles--papers : Chan-Bagot et al., 2024
        """
        by_orbit = {o: image_collection.filter(
            ee.Filter.eq('orbitProperties_pass', o)) for o in self.orbits}

        s1_composite = image_collection.median().select(['VV', 'VH'])
        for orbit, collection in by_orbit.items():
            suffix = self.ORBIT_SUFFIX[orbit]
            # Min reducing was selected per Chan-Bagot et al., 2024
            reduced = collection.min().select(
                ['VV', 'VH'], [f'VV_{suffix}', f'VH_{suffix}'])
            s1_composite = s1_composite.addBands(reduced)

        # speckle filtering after compositing to avoid memory limit issues
        # filtering before clipping to ensure edges have neighbors to sample from
        if self.config['preprocessing']['sentinel1']['speckle_filter']:
            s1_composite = refined_lee(s1_composite)

        return s1_composite.clip(self.aoi_polygon)

    def load(self):
        """
        Instantiates a Sentinel1 image collection in linear scale
        for the date range, aoi specified, and co-polarization/cross-polarization bands & incidence angle

        Returns
        -------
        ee.ImageCollection
            A collection of linear Sentinel-1 images filtered by date and AOI with polarization bands
            and incidence angle

        See Also
        --------
        docs/references.md#documentation : Sentinel-1 Algorithms
        """
        s1_linear = (
            ee.ImageCollection(
                "COPERNICUS/S1_GRD_FLOAT")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi_polygon)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .select(['VV', 'VH', 'angle'])
        )
        return s1_linear

    def preprocess(self, image_collection):
        """
        Applies edge masks

        Parameters
        ----------
        image_collection : ee.ImageCollection
            Sentinel-1 collection in linear scale

        Returns
        -------
        ee.ImageCollection
            Filtered Sentinel-1 collection
        """
        s1_masked = image_collection.map(self._mask_edge)
        return s1_masked

    def _mask_edge(self, s1_image):
        """
        Masks out near-zero backscatter values found at scene edges.
        Edge pixels in Sentinel-1 GRD have artificially low backscatter in linear scale,
        so pixels below the threshold are removed.

        Parameters
        ----------
        s1_image : ee.Image
            A single Sentinel-1 image in linear scale

        Returns
        -------
        ee.Image
            Sentinel-1 image with the edge mask applied

        See Also
        --------
        docs/references.md#documentation : Sentinel-1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected,
        log scaling
        """
        MAX_DB = self.config['preprocessing']['sentinel1']['edge_mask_threshold_db']
        MAX_LINEAR = 10 ** (MAX_DB / 10)  # converting from dB to linear

        edge = s1_image.lt(MAX_LINEAR)
        masked_image = s1_image.mask().And(edge.Not())
        return s1_image.updateMask(masked_image)
