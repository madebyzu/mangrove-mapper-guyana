
def to_db(image):
    """
    Converts SAR image, linear scale to decibels (dB).

    Parameters
    ----------
    image: ee.Image
        SAR image with bands in linear scale

    Returns
    -------
    ee.Image
        Image with bands converted to dB

    See Also
    --------
    docs/references.md#websites : Convert Sentinel-1 images data from dB to linear
    """
    return image.log10().multiply(10)
