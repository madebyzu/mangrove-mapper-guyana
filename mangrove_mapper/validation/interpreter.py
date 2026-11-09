"""
Notebook-only interpretation UI for the accuracy assessment.

Kept apart from validator.py so the analysis half stays importable without a
Jupyter stack -- geemap/ipywidgets pull in ~2000 modules.
"""
import os

import ee
import geemap
import ipywidgets as widgets
import pandas as pd
from ipyleaflet import CircleMarker
from IPython.display import display

LABEL_COL = 'planet_label'        # column the interpreter fills in
ZOOM = 16                         # zoom used when centring on a plot


def build_interactive_map(collection_path, aoi_polygon, accuracy_plot_path):
    """
    Creates the Google Earth Engine (GEE) interactive map that displays the
    Planet imagery of the area of interest, masking clouds and adding the plot
    points drawn from the accuracy-assessment sheet.

    Parameters
    ----------
    collection_path : str
        Path to the GEE image collection where the Planet imagery is stored.
    aoi_polygon : list[list[float]]
        Matrix of coordinate vertices outlining the area of interest.
    accuracy_plot_path : str or pathlib.Path
        Accuracy-assessment sheet the sample plots are read from.

    Returns
    -------
    geemap.Map
        GEE map over the area of interest with the sample plots overlaid.

    Notes
    -----
    Plots are added carrying PLOTID alone. The model's prediction is withheld
    deliberately, so that the interpreter's call stays independent of the map
    being assessed.
    """
    def mask_clouds(img):
        udm = img.select("udm1")
        clear = udm.bitwiseAnd(3).eq(0)  # bit 0 = blackfill, bit 1 = cloud
        return img.updateMask(clear)

    def add_plot_points(m):
        df = pd.read_csv(accuracy_plot_path)
        # blind: PLOTID only
        pts = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Point([r.LON, r.LAT]), {'PLOTID': int(r.PLOTID)})
             for r in df.itertuples()
             ])
        m.add_layer(pts.style(color='yellow', pointSize=4,
                              fillColor='00000000'), {}, 'sample points')

    aoi_geom = ee.Geometry.Polygon([aoi_polygon])

    col = ee.ImageCollection(collection_path).filterBounds(aoi_geom)
    info = col.first().getInfo()
    col_masked = col.map(mask_clouds)
    band_ids = [b["id"] for b in info["bands"]]
    # B2, B3, B4 (red, blue, green)
    vis = {"bands": band_ids[:3], "min": 0, "max": 255}
    m = geemap.Map()
    m.add_layer(col.mosaic().clip(aoi_geom), vis, 'planet')
    m.add_layer(col_masked.mosaic().clip(aoi_geom), vis, 'cloud_mask')
    outline = ee.FeatureCollection(aoi_geom).style(
        color='red', fillColor='00000000', width=2)  # transparent fill, red outline

    add_plot_points(m)
    m.add_layer(outline, {}, 'aoi outline')
    m.center_object(aoi_geom, 10)

    return m


def label_plots_widgets(m, accuracy_plot_path, label_col=LABEL_COL, zoom=ZOOM):
    """
    Adds navigation and labelling buttons to a Google Earth Engine (GEE)
    interactive map, so each sample plot can be interpreted in a notebook.

    Centers the map on one plot at a time and writes the interpreter's call
    into `label_col`, saving after every click. The model's prediction is never shown, keeping
    the reference label independent of the map being assessed.

    Parameters
    ----------
    m : geemap.Map
        GEE interactive map with the sample points, from build_interactive_map.
    accuracy_plot_path : str or pathlib.Path
        Accuracy-assessment sheet holding PLOTID/LON/LAT. Modified in place:
        `label_col` is rewritten to disk after every click.
    label_col : str, optional
        Column the interpretation is written to. Default 'planet_label'.
    zoom : int, optional
        Zoom level used when centering on a plot. Default 16;

    Notes
    -----
    The buttons need a live ipywidgets connection, so run this in JupyterLab or VS Code.

    See Also
    --------
    docs/references.md#ai-usage : AI assistance log
    """
    df = pd.read_csv(accuracy_plot_path)
    l_col = df.columns.get_loc(label_col)

    def next_unlabeled(start):
        """First unlabelled row position at or after `start`, else None."""
        for pos in range(max(start, 0), len(df)):
            if pd.isna(df.iat[pos, l_col]):
                return pos
        return None

    # dicts rather than plain names so the nested callbacks can mutate them
    # without needing `nonlocal`
    start = next_unlabeled(0)
    pos = {'v': 0 if start is None else start}
    last = {'msg': ''}

    status = widgets.HTML()
    out = widgets.Output()

    # client-side marker so the plot under judgement is unambiguous; moving it
    # is instant, unlike re-adding an ee layer on every click
    crosshair = CircleMarker(location=(0, 0), radius=9, color='#ff2d2d',
                             weight=3, fill_opacity=0.0)
    old = getattr(m, '_interp_crosshair', None)
    if old is not None and old in m.layers:
        m.remove_layer(old)
    m.add_layer(crosshair)
    m._interp_crosshair = crosshair

    def save():
        tmp = f'{accuracy_plot_path}.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, accuracy_plot_path)

    def show():
        p = pos['v']
        r = df.iloc[p]
        crosshair.location = (float(r.LAT), float(r.LON))
        m.set_center(float(r.LON), float(r.LAT), zoom)
        raw = df.iat[p, l_col]
        lab = 'unlabelled' if pd.isna(
            raw) else ('MANGROVE' if int(raw) == 1 else 'non')
        done = int(df[label_col].notna().sum())
        status.value = (
            f"<b>PLOT {int(r.PLOTID)}</b> &nbsp;(row {p + 1}/{len(df)})&nbsp; | &nbsp;"
            f"this plot: <b>{lab}</b> &nbsp; | &nbsp; {done}/{len(df)} done"
            f"<br><span style='color:#888'>{last['msg']}</span>")

    def set_label(val):
        p = pos['v']
        df.iat[p, l_col] = val
        save()
        last['msg'] = f'saved PLOT {int(df.iloc[p].PLOTID)} = {val}'
        # next one FORWARD, not the first unlabelled in the file, or Prev/Skip
        # would throw the interpreter back to the start of the sheet
        nxt = next_unlabeled(p + 1)
        pos['v'] = min(p + 1, len(df) - 1) if nxt is None else nxt
        show()

    def step(d):
        pos['v'] = min(max(pos['v'] + d, 0), len(df) - 1)
        last['msg'] = ''
        show()

    def jump_to_next_todo():
        nxt = next_unlabeled(0)
        last['msg'] = 'all plots labelled' if nxt is None else ''
        if nxt is not None:
            pos['v'] = nxt
        show()

    def handler(fn):
        """Surface callback errors in `out` instead of swallowing them."""
        def _cb(_btn):
            with out:
                out.clear_output()
                try:
                    fn()
                except Exception:
                    import traceback
                    traceback.print_exc()
        return _cb

    prev = widgets.Button(description='< Prev')
    non = widgets.Button(description='Non (0)', button_style='warning')
    mang = widgets.Button(description='Mangrove (1)', button_style='success')
    skip = widgets.Button(description='Skip >')
    todo = widgets.Button(description='Next unlabelled')
    prev.on_click(handler(lambda: step(-1)))
    non.on_click(handler(lambda: set_label(0)))
    mang.on_click(handler(lambda: set_label(1)))
    skip.on_click(handler(lambda: step(1)))
    todo.on_click(handler(jump_to_next_todo))

    show()
    display(widgets.HBox([prev, non, mang, skip, todo]), status, out, m)
