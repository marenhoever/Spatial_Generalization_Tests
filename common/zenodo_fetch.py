"""Shared helper: fetch external data files from Zenodo if not already present locally.

Used by any model's notebooks that need to pull data bundled on Zenodo rather than
included in the git repo directly (see the Data section of ../README.md for which
record holds what, and why). Standard-library only, no extra pip dependency
needed in any model's requirements.txt/Project.toml.

Typical use in a notebook's config cell:

    import sys
    sys.path.insert(0, "../../common")  # adjust the number of ".." to this notebook's depth
    from zenodo_fetch import ensure_zenodo_files

    ZENODO_RECORD = None  # set to the published record id, e.g. "1234567"
    DATA_DIR = "../../data/<Model>/<subfolder>"
    ensure_zenodo_files(ZENODO_RECORD, {
        "some_file.nc": "<md5 checksum>",
        ...
    }, DATA_DIR)

Zenodo stores a record's files in one flat namespace -- there are no directories
inside a record. A file that lives in a subdirectory locally, or whose basename is
not unique within its record, therefore needs a different name on Zenodo than the
path it is stored under here. Give those files a ``(zenodo_filename, md5)`` pair
instead of a bare checksum:

    ensure_zenodo_files(ZENODO_RECORD, {
        "predictions/axis_flip/incl_flipped_wind/error_2mT_era5_corrected.npy":
            ("revlon__error_2mT_era5_corrected.npy", "<md5>"),
    }, DATA_DIR)
"""

import hashlib
import os
import urllib.parse
import json
import urllib.request
import zipfile
from pathlib import Path

# Zenodo's sandbox is a separate site with separate record ids. Setting
# ZENODO_BASE_URL=https://sandbox.zenodo.org lets a dry run be tested end to end
# before anything is published for real. Unset, this is the real Zenodo.
BASE_URL = os.environ.get("ZENODO_BASE_URL", "https://zenodo.org").rstrip("/")


def _md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_zenodo_files(record_id, files, dest_dir, verify_existing=False):
    """Ensure each file in `files` exists in `dest_dir`, downloading from the given
    Zenodo record if missing, and verifying checksums.

    Parameters
    ----------
    record_id : str or None
        The published Zenodo record id (the number in zenodo.org/records/<id>).
        If None and a file is missing, raises FileNotFoundError with a clear message
        instead of silently failing or trying to download from a placeholder URL.
    files : dict
        Maps each file's path *relative to dest_dir* to either:
          * its expected md5 checksum (the file is called the same thing on Zenodo
            as its basename here), or
          * a ``(zenodo_filename, md5)`` pair, when the name on Zenodo differs --
            see the module docstring for when that is needed.
        Pass None as the checksum to skip verification for a given file.
    dest_dir : str or Path
        Local directory the files should end up in (created if missing).
    verify_existing : bool
        If True, re-checksum files that are already present instead of trusting
        them. Off by default: several of these files are tens of GB, and hashing
        them on every notebook run costs minutes for no benefit once they have
        been verified on download.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for relpath, spec in files.items():
        if isinstance(spec, (tuple, list)):
            zenodo_name, expected_md5 = spec
        else:
            zenodo_name, expected_md5 = Path(relpath).name, spec

        fpath = dest_dir / relpath

        if fpath.exists():
            if verify_existing and expected_md5 is not None:
                actual_md5 = _md5(fpath)
                if actual_md5 != expected_md5:
                    raise ValueError(
                        f"{fpath}: checksum mismatch for the file already on disk "
                        f"(got {actual_md5}, expected {expected_md5}). Delete it to "
                        "re-download."
                    )
                print(f"  {relpath}: OK (existing file, checksum verified)")
            continue

        if record_id is None:
            raise FileNotFoundError(
                f"{fpath} not found locally, and no Zenodo record id was given to "
                "auto-download it. Either place the file there manually, or set the "
                "record id once the data has been published to Zenodo."
            )

        # Files kept in subdirectories here need those directories created first --
        # urlretrieve will not create them and fails with FileNotFoundError.
        fpath.parent.mkdir(parents=True, exist_ok=True)

        # read at call time, not import time, so setting it in a notebook cell
        # works regardless of where the import happens to sit
        base = os.environ.get("ZENODO_BASE_URL", BASE_URL).rstrip("/")
        url = (
            f"{base}/records/{record_id}/files/"
            f"{urllib.parse.quote(zenodo_name)}?download=1"
        )
        print(f"Downloading {zenodo_name} from {url} ...")

        # Download to a .part file and rename only once it is complete and verified,
        # so an interrupted download can never be mistaken for a finished file by
        # the `fpath.exists()` check on the next run.
        tmp_path = fpath.with_name(fpath.name + ".part")
        try:
            urllib.request.urlretrieve(url, tmp_path)

            if expected_md5 is not None:
                actual_md5 = _md5(tmp_path)
                if actual_md5 != expected_md5:
                    raise ValueError(
                        f"{zenodo_name}: checksum mismatch after download "
                        f"(got {actual_md5}, expected {expected_md5}) -- discarded, "
                        "please retry"
                    )
                tmp_path.rename(fpath)
                print(f"  {relpath}: OK (checksum verified)")
            else:
                tmp_path.rename(fpath)
                print(f"  {relpath}: OK (no checksum given to verify)")
        finally:
            tmp_path.unlink(missing_ok=True)


def ensure_forecast_files(record_id, variant, dest_dir, checksums_path, days=None,
                          keep_zips=False):
    """Ensure the per-day GraphCast forecasts for one test case are present.

    Each test case is 366 daily NetCDF files. They are not archived one file per
    Zenodo file, because a Zenodo record holds at most 100 files no matter how
    small they are -- that cap is separate from the 50GB storage quota and is not
    lifted by a quota increase. They are archived as one zip per calendar month
    of the forecast's initialisation date instead: 13 per record, which clears
    the cap and still means fetching March does not mean fetching the year.

    So this function works a month at a time. For each month that has a day you
    asked for and do not already have, it downloads that month's zip, verifies
    the archive against its md5, extracts the members it needs, verifies each of
    those against its own md5, and deletes the zip. A month whose days are all
    already on disk is never downloaded.

    The names, checksums and month groupings all live in
    ``GraphCast/forecast_checksums.json``, which ships with this repository. The
    same file drove the upload that produced the Zenodo records, so what was
    archived and what this fetches cannot drift apart.

    Parameters
    ----------
    record_id : str or None
        Zenodo record holding this test case's forecasts. One record per test
        case: each is ~24.9GB, and a record is capped at 50GB.
    variant : str
        Canonical test-case name -- 'baseline', 'revlat', 'revlon' or 'rotlon'.
        The on-disk directories still carry the original spellings
        (`not_flipped/`, `equatorial_flip/`, ...) and the JSON maps between
        them. Each day entry's "local" is where the file goes on disk; its
        "arc" is the name it has inside the zip, which for the production
        records is the canonical one. Entries without "arc" come from archives
        built before the two were allowed to differ, and are read under
        "local".
    dest_dir : str or Path
        Same meaning as in ``ensure_zenodo_files``: the local directory the
        paths in the JSON are relative to (``data/GraphCast``).
    checksums_path : str or Path
        The JSON file described above.
    days : iterable of int, optional
        Restrict to particular days (0-365). Defaults to every day in the file.
        Note there are 366 daily forecasts but the error arrays cover days
        0-364 -- day 365 is archived for completeness and unused downstream.
        A month is fetched whole even if you only want one of its days: the zip
        is the unit Zenodo stores.
    keep_zips : bool
        Leave the downloaded archives in ``<dest_dir>/_forecast_zips`` instead of
        deleting them once extracted. Off by default -- keeping all of them costs
        another 99.5GB for no benefit, since the extracted files are what is read.
    """
    with open(checksums_path) as fh:
        table = json.load(fh)

    if variant not in table:
        raise KeyError(
            f"unknown test case {variant!r} -- {checksums_path} has "
            f"{sorted(table)}"
        )
    case = table[variant]
    if "zips" not in case:
        raise ValueError(
            f"{checksums_path} is in an older one-file-per-day format that "
            "predates the monthly archives on Zenodo. Restore the copy of this "
            "file that ships with the repository."
        )

    dest_dir = Path(dest_dir)
    all_days = case["days"]
    wanted = sorted(all_days, key=int) if days is None else [str(d) for d in days]
    for day in wanted:
        if day not in all_days:
            raise KeyError(f"{variant}: no day {day} in {checksums_path}")

    # Group the days we still need by the month-zip that holds them, so each
    # archive is fetched at most once however many of its days were asked for.
    needed = {}
    for day in wanted:
        e = all_days[day]
        if (dest_dir / e["local"]).exists():
            continue
        needed.setdefault(e["zip"], []).append(day)

    if not needed:
        print(f"{variant}: all {len(wanted)} requested forecasts already on disk")
        return

    unbuilt = [m for m in sorted(needed) if not case["zips"][m]["md5"]]
    if unbuilt:
        raise ValueError(
            f"{variant}: no checksum recorded for the {', '.join(unbuilt)} "
            f"archive(s) in {checksums_path}, so they cannot be verified. "
            "Restore the copy of this file that ships with the repository."
        )

    todo_bytes = sum(case["zips"][m]["size"] for m in needed)
    print(f"{variant}: {sum(len(v) for v in needed.values())} of {len(wanted)} "
          f"forecasts missing, in {len(needed)} monthly archive(s), "
          f"{todo_bytes/2**30:.1f}GB to download")

    staging = dest_dir / "_forecast_zips"
    for month in sorted(needed):
        spec = case["zips"][month]
        zpath = staging / spec["zenodo_name"]

        ensure_zenodo_files(record_id, {spec["zenodo_name"]: spec["md5"]}, staging)

        try:
            with zipfile.ZipFile(zpath) as zf:
                for day in needed[month]:
                    e = all_days[day]
                    out = dest_dir / e["local"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    # The name inside the archive is not always the on-disk path:
                    # the production records store each member under its canonical
                    # test-case name, while the older archives stored the on-disk
                    # spelling. "arc" records the former; its absence means the
                    # archive predates the two being allowed to differ.
                    arc = e.get("arc", e["local"])
                    # extract via a .part file for the same reason downloads do:
                    # an interrupted extraction must not look like a finished file
                    tmp = out.with_name(out.name + ".part")
                    try:
                        h = hashlib.md5()
                        with zf.open(arc) as src, open(tmp, "wb") as dst:
                            for chunk in iter(lambda: src.read(1 << 20), b""):
                                h.update(chunk)
                                dst.write(chunk)
                        if h.hexdigest() != e["md5"]:
                            raise ValueError(
                                f"{arc}: checksum mismatch after extracting "
                                f"from {spec['zenodo_name']} (got {h.hexdigest()}, "
                                f"expected {e['md5']})")
                        tmp.replace(out)
                    finally:
                        tmp.unlink(missing_ok=True)
            print(f"  {spec['zenodo_name']}: {len(needed[month])} forecast(s) "
                  "extracted and verified")
        finally:
            if not keep_zips:
                zpath.unlink(missing_ok=True)

    if not keep_zips:
        try:
            staging.rmdir()
        except OSError:
            pass
