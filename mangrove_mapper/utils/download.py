# Download an exported file from Google Drive directly to a local folder.

# docs/references.md#websites : Barth, (2024)
# docs/references.md#ai-usage : AI assistance log

import io
import time
from pathlib import Path

import ee
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _drive_service():
    """
    Build a Google Drive v3 client.

    Reuses the credentials already established by ee.Authenticate(), so no
    separate Drive login is needed.

    Returns
    -------
    googleapiclient.discovery.Resource
        Authenticated Drive v3 service.
    """
    return build('drive', 'v3', credentials=ee.data.get_persistent_credentials())


def _download_media(drive_service, file_id, dest, chunksize=64 * 1024 * 1024):
    """
    Stream one Drive file to a local path.

    Parameters
    ----------
    drive_service : googleapiclient.discovery.Resource
        Authenticated Drive v3 service, e.g. from `_drive_service`.
    file_id : str
        Drive id of the file to download.
    dest : pathlib.Path or str
        Local output or destination path. Parent directories are created if missing.
    chunksize : int
        Bytes fetched per request. Default 64 MiB.

    Returns
    -------
    pathlib.Path
        The path the file was written to.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(dest, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunksize)
        done = False
        while not done:
            # GETs the next chunk of bytes and writes to disk, growing the file
            # _ returns status object, could be used as status.progress() => %completed
            _, done = downloader.next_chunk()
    return dest


def download_from_drive(file_name, file_format, output_dir=REPO_ROOT / 'data', task=None):
    """
    Download an exported file from Google Drive to the local hard drive.

    Parameters
    ----------
    file_name : str
        Name of the exported file, matching the description passed to the
        export task (without extension).
    file_format : str
        CSV, KML, TFRecord, SHP, GeoJSON or KMZ
    output_dir : pathlib.Path
        Desired output directory path. Created if it does not exist.
    task : ee.batch.Task, optional
        The started export task. If provided, polls until the export completes
        before downloading; otherwise the file is expected to already be in Drive.

    Raises
    ------
    FileNotFoundError
        If no matching file is found in Drive (e.g. the export has not completed).
    RuntimeError
        If a provided task finishes in a non-completed state.
    """
    if task:
        poll_task(task)

    drive_service = _drive_service()
    # Find the exported file by name
    results = drive_service.files().list(
        q=f"name='{file_name}.{file_format.lower()}' and trashed=false",
        fields="files(id, name)"
    ).execute()

    files = results.get('files', [])
    if not files:
        raise FileNotFoundError(
            f"{file_format} not found in Drive — has the export task completed?")

    output_path = Path(output_dir) / f'{file_name}.{file_format.lower()}'
    _download_media(drive_service, files[0]['id'], output_path)
    print(f"Saved to {output_path}")


def poll_task(task):
    """
    Poll until the batch task finishes, raising if it does not complete.

    Parameters
    ----------
    task : ee.batch.Task
        The started export task to wait on.

    Raises
    ------
    RuntimeError
        If the task finishes in a non-completed state (e.g. FAILED, CANCELLED).
    """
    print("Waiting for export to complete...")
    print("See https://code.earthengine.google.com/tasks for more information about your task")
    while task.active():
        print(f"  {task.status()['state']}...")
        time.sleep(30)

    state = task.status()['state']
    if state != 'COMPLETED':
        raise RuntimeError(
            f"Export failed: {task.status().get('error_message')}")

    print("Export complete, downloading from Drive!")


def download_prefix(prefix, output_dir=REPO_ROOT / 'data'):
    """
    Download every Drive file whose name starts with `prefix` into `output_dir`.

    A tiled image export is written by Drive as one file per surviving tile
    (`<prefix>-<row>-<col>.tif`), so a single exact name will not find them all.
    Files already present locally at the same byte size are skipped, so an
    interrupted download can be resumed by re-running.

    Parameters
    ----------
    prefix : str
        Export description that every tile's file name starts with.
    output_dir : pathlib.Path or str
        Local directory to download into. Created if it does not exist.

    Returns
    -------
    list of pathlib.Path
        Paths of the tiles now on disk, sorted by name.

    Raises
    ------
    FileNotFoundError
        If no matching file is found in Drive (e.g. the export has not completed).
    """
    drive_service = _drive_service()
    output_dir = Path(output_dir)

    # `contains` is server-side and loose, so keep only true prefix matches;
    # page through in case the export wrote more than one page of tiles
    files, page = [], None
    # searching and collecting relevant files
    while True:
        results = drive_service.files().list(
            q=f"name contains '{prefix}' and trashed=false",
            fields="nextPageToken, files(id, name, size)",
            pageSize=200, pageToken=page).execute()
        files += [f for f in results.get('files', [])
                  if f['name'].startswith(prefix)]
        page = results.get('nextPageToken')
        if not page:
            break

    if not files:
        raise FileNotFoundError(
            f"no Drive files starting with '{prefix}' — has the export completed?")

    # downloading collected files, if they don't already exist locally
    for f in sorted(files, key=lambda f: f['name']):
        dest = output_dir / f['name']
        # the size comparison helps to catch files that were partial downloaded or different tiles with the same name
        if dest.exists() and dest.stat().st_size == int(f.get('size', 0)):
            print(f"  have {f['name']}")
            continue
        _download_media(drive_service, f['id'], dest)
        print(f"  got {f['name']}")

    return sorted(output_dir.glob(f"{prefix}*"))
