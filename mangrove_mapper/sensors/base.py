from abc import ABC, abstractmethod
from pathlib import Path

import ee
import yaml


class SensorProcessor(ABC):
    """
    Abstract base class that standardizes the processing pipeline across all sensors.

    Subclasses must implement:
        - load(): fetch the image collection from GEE
        - preprocess(): apply sensor-specific corrections (e.g. scaling, cloud masking)
        - composite(): reduce the collection to an annual single image
        - compute_indices(): compute spectral or backscatter indices from the composite

    Subclasses inherit:
        - process(): executes the full pipeline in order, ready for model ingestion

    See Also
    --------
    docs/pipeline/phase1_flowchart.png : Pipeline diagram
    docs/references.md#ai-usage : AI assistance log
    """

    def __init__(self, start_date=None, end_date=None, aoi_polygon=None, project=None):
        """
        Constructor for base sensor class. Given each sensor has a nearly identical interface,
        this constructor provides a template from which sensors can inherit.

        Parameters
        ----------
        start_date : str
            Start of the date range. Ex: "2020-01-01". Defaults to config.yml value if not provided.
        end_date : str
            End of the date range. Ex: "2020-12-31". Defaults to config.yml value if not provided.
        aoi_polygon : list of list of float
            2D list of coordinates defining the area of interest (drawn in the GEE Script Editor).
            Defaults to the AOI defined in config.yml if not provided.
        project : str
            Google Cloud project initialization
        """

        ee.Initialize(project=project)

        with open(Path(__file__).parent.parent / "configs/indices.yml") as f:
            self.indices_config = yaml.safe_load(f)

        with open(Path(__file__).parent.parent / "configs/config.yml") as f:
            self.config = yaml.safe_load(f)

        if start_date is None or end_date is None:
            self.start_date = self.config["default_date_range"]["start_date"]
            self.end_date = self.config["default_date_range"]["end_date"]
            print(
                f'start_date or end_date is missing, defaulting to {self.start_date} - {self.end_date}')
        else:
            self.start_date = start_date
            self.end_date = end_date

        self.aoi_polygon = aoi_polygon or ee.Geometry.Polygon(
            [self.config["default_aoi"]])

    def process(self):
        """
        Runs the full sensor processing pipeline

        Returns
        -------
        image_with_indices : ee.Image
            Processed composite image with spectral bands and computed indices.
        """
        image_collection = self.load()
        processed_image_collection = self.preprocess(image_collection)
        image_composite = self.composite(processed_image_collection)
        image_with_indices = self.compute_indices(image_composite)
        return image_with_indices

    def compute_indices(self, image):
        """
        Applying all spectral index equations based on the definitions in the config.

        Parameters
        ----------
        image : ee.Image
            A single satellite image.

        Returns
        -------
        image : ee.Image
            A single satellite image with spectral indices as additional bands.
        """
        indices = self.indices_config[self.indices_key]
        for idx_name, index in indices.items():
            if not index:
                raise ValueError(
                    f'{idx_name} is not filled in. Please fill in or remove index')
            if self.sensor_key not in index:
                continue  # skip indices not defined for this sensor
            value = image.expression(
                index['expression'],
                {key: image.select(band)
                 for key, band in index[self.sensor_key].items()}
            ).rename(idx_name)
            image = image.addBands(value)
        return image

    @property
    @abstractmethod
    def sensor_key(self):
        """
        Returns
        -------
        string
            Sensor key matching what's in the indices.yml ['landsat8', 'sentinel2', 'sentinel1', 'alos_palsar']
        """
        ...

    @property
    @abstractmethod
    def indices_key(self):
        """
        Returns
        -------
        string
            Indices key matching what's in the indices.yml ['optical_indices', 'sar_indices']
        """
        ...

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def preprocess(self, image):
        ...

    @abstractmethod
    def composite(self, image):
        ...
