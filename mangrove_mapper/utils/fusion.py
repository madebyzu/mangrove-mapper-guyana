import ee


def load_target(target_dataset_id, aoi_polygon, mangrove_class, permanent_water_threshold=100,
                target_name='label', presence_only=False):
    """
    Loads the target dataset and prepares binary classes.

    Parameters
    ----------
    target_dataset_id : str
        Name of target dataset in GEE catalogue, e.g. 'ESA/WorldCover/v200' or 'GMW_MNG_2020'.
    aoi_polygon : ee.Geometry.Polygon
        Area of interest (aoi)
    mangrove_class : int
        Pixel value denoting mangrove in the target band (95 for ESA WorldCover
        'Map', 1 for Global Mangrove Watch 'b1').
    permanent_water_threshold : int
        JRC GSW 'occurrence' % band at/above which a pixel is masked as permanent
        water (default 100 = mask only always-water pixels, keeping intertidal
        mudflats in the non-mangrove domain).
    target_name : str
        Target name
    presence_only : bool
        Set True for presence-only targets that mark mangrove and leave
        everything else masked (e.g. Global Mangrove Watch), Default False.

    Returns
    -------
    ee.Image
        binary target (mangrove or not)

    See Also
    --------
    docs/references.md#documentation : JRC Global Surface Water Mapping Layers, v1.4.
    """
    # mosaic (not first) so a tiled collection like GMW combines into one image;
    dataset = ee.ImageCollection(target_dataset_id).filterBounds(
        aoi_polygon).mosaic().clip(aoi_polygon)

    # unmask sets land to 0% occurence
    occurrence = (ee.Image(
        'JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(aoi_polygon).unmask(0))
    # masks at permanent water (100% occurence)
    water_mask = occurrence.lt(permanent_water_threshold)

    label = dataset.eq(mangrove_class)
    if presence_only:
        # presence-only targets leave non-mangrove masked; fill it with 0
        label = label.unmask(0)
    label = label.rename(target_name).updateMask(water_mask)
    return label


def product_label(product_dataset_id, aoi_polygon, mangrove_class,
                  presence_only=False, target_name='mangrove'):
    """
    A contiguous mangrove extent for a published product (e.g. ESA WorldCover).

    Mosaics a (possibly tiled) collection into one image, clips to the aoi, and tests the mangrove class.

    Parameters
    ----------
    product_dataset_id : str
        Name of the product in the GEE catalogue, e.g. 'ESA/WorldCover/v200'.
    aoi_polygon : ee.Geometry.Polygon
        Area of interest (aoi).
    mangrove_class : int
        Pixel value denoting mangrove (95 for ESA WorldCover 'Map', 1 for
        Global Mangrove Watch 'b1').
    presence_only : bool
        Set True for presence-only products that mark mangrove and leave everything
        else masked (e.g. Global Mangrove Watch); non-mangrove is filled with 0.
        Default False.
    target_name : str
        Output band name. Default 'mangrove'.

    Returns
    -------
    ee.Image
        Single band, 1 where the product maps mangrove and 0 elsewhere.

    See Also
    --------
    docs/references.md#documentation : ESA WorldCover 10m v200.
    """
    # .mosaic() not .first() so a tiled collection combines into one image
    dataset = ee.ImageCollection(product_dataset_id).filterBounds(
        aoi_polygon).mosaic().clip(aoi_polygon)

    label = dataset.eq(mangrove_class)
    if presence_only:
        # presence-only products leave non-mangrove masked; fill it with 0
        label = label.unmask(0)
    return label.rename(target_name).toUint8()  # for memory efficiency


def concat_sensors(sensors):
    """
    Concatenates multiple sensors with their processed bands

    Parameters
    ----------
    sensors : list[SensorProcessors]
        List of sensors (Landsat8, Sentinel1, etc.)

    Returns
    -------
    ee.Image
        Image with bands concatenated from multiple sensors
    """
    images = [_prefix_bands(sensor.process(), sensor.prefix)
              for sensor in sensors]
    return ee.Image.cat(images)


def _prefix_bands(image, prefix):
    """
    Appends image bands with sensor prefix

    Parameters
    ----------
    image : ee.Image
        Image whose bands will be renamed
    prefix : str
        Prefix to be appended

    Returns
        Image with renamed band

    Example:
        mRFDI --> alos_mRFDI
        NDVI --> s2_NDVI
    """
    prefixed_band = image.bandNames().map(
        lambda name: ee.String(prefix).cat('_').cat(ee.String(name)))
    return image.rename(prefixed_band)
