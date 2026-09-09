# argumentation_functions.py
import xarray
import numpy as np
import jax.numpy as jnp
import jax
from typing import Union

# Canonical test-case names used throughout this release:
#
#   'rotlon'  rotate longitude by 180 deg   (roll lon by half the grid; time +12h)
#   'revlat'  reverse latitude              (mirror lat; negate v-wind; time +183d)
#   'revlon'  reverse longitude             (mirror lon; negate u-wind)
#
# The original notebooks used 'equatorial'/'axis'/'180' for the same three
# transforms. Those are still accepted as aliases so any older call site keeps
# working and behaves identically -- only the spelling changed, never the maths.
#
# NOTE: this NeuralGCM copy is NOT identical to the GraphCast one of the same
# name -- it shifts the `time` coordinate where GraphCast shifts `datetime`,
# and its 'rotlon' shift is +12h where GraphCast's is -12h. Do not swap one for
# the other.
LEGACY_ROTATION_NAMES = {"equatorial": "revlat", "axis": "revlon", "180": "rotlon"}


def _canonical_rotation(rotation: str) -> str:
    """Map a legacy rotation name onto its canonical equivalent."""
    return LEGACY_ROTATION_NAMES.get(rotation, rotation)

def flip(data: Union[xarray.Dataset, np.ndarray], rotation: str) -> Union[xarray.Dataset, np.ndarray]:
    """
    Flips an xarray Dataset or a numpy array spatially and/or temporally depending on the rotation type.

    Parameters
    ----------
    data : xarray.Dataset or numpy.ndarray
        Input data. If ndarray, it should be of shape [channels, lon, lat].
    rotation : str
        One of: 'rotlon', 'revlat', 'revlon'
        (legacy aliases '180', 'equatorial', 'axis' are also accepted)

    Returns
    -------
    xarray.Dataset or numpy.ndarray
        Flipped data of the same type as input.
    """
    rotation = _canonical_rotation(rotation)
    is_xarray = isinstance(data, xarray.Dataset)
    new_data = data.copy()

    def _flip_lat_xr(ds):
        flipped = ds.sel(latitude=slice(None, None, -1))
        flipped = flipped.assign_coords(latitude=-flipped.latitude.values)
        return flipped

    def _flip_lon_xr(ds):
        flipped = ds.sel(longitude=slice(None, None, -1))
        flipped = flipped.assign_coords(longitude=flipped.longitude.values[::-1])
        return flipped

    def _shift_lon_xr(ds):
        n = ds.sizes["longitude"]
        return ds.roll(longitude=n // 2, roll_coords=False)

    def _flip_lat_np(arr):
        return arr[:, :, ::-1]

    def _flip_lon_np(arr):
        return arr[:, ::-1, :]

    def _shift_lon_np(arr):
        n = arr.shape[1]
        return np.roll(arr, shift=n // 2, axis=1)

    if is_xarray:
        if rotation == "revlat":
            new_data = _flip_lat_xr(new_data)
            for key in new_data.data_vars:
                if "v_component_of_wind" in key and "10m" not in key:
                    new_data[key] *= -1
                if "10m_v_component_of_wind" in key:
                    new_data[key] *= -1
            if "time" in new_data.coords:
                new_data["time"] = new_data.time + np.timedelta64(183, "D")

        elif rotation == "revlon":
            new_data = _flip_lon_xr(new_data)
            for key in new_data.data_vars:
                if "u_component_of_wind" in key and "10m" not in key:
                    new_data[key] *= -1
                if "10m_u_component_of_wind" in key:
                    new_data[key] *= -1

        elif rotation == "rotlon":
            new_data = _shift_lon_xr(new_data)
            if "time" in new_data.coords:
                new_data["time"] = new_data.time + np.timedelta64(12, "h")

    else:  # numpy array
        if rotation == "revlat":
            new_data = _flip_lat_np(new_data)
            # You could optionally flip signs on channels representing v_wind if known
        elif rotation == "revlon":
            new_data = _flip_lon_np(new_data)
            # You could optionally flip signs on channels representing u_wind if known
        elif rotation == "rotlon":
            new_data = _shift_lon_np(new_data)

    return new_data


def flip_back(data: xarray.Dataset, rotation: str) -> xarray.Dataset:
    """
    Reverses the flipping of an xarray Dataset based on the rotation type.

    Parameters
    ----------
    data : xarray.Dataset
        Input dataset that has been flipped.
    rotation : str
        One of: 'rotlon', 'revlat', 'revlon'
        (legacy aliases '180', 'equatorial', 'axis' are also accepted)

    Returns
    -------
    xarray.Dataset
        Dataset flipped back to the original orientation.
    """
    rotation = _canonical_rotation(rotation)
    new_data = data.copy()


    def _flip_lat(ds):
        if isinstance(ds, (xarray.DataArray, xarray.Dataset)):
            # Flip latitude axis in xarray
            flipped = ds.sel(latitude=slice(None, None, -1))
            flipped = flipped.assign_coords(latitude=-flipped.latitude.values)
            return flipped
        elif ds.__module__.startswith('jaxlib.'):
            return jnp.flip(ds, axis=-1)
        else:
            raise TypeError("Unsupported input type. Must be xarray DataArray/Dataset or JAX array.")

    def _flip_lon(ds):
        flipped = ds.sel(longitude=slice(None, None, -1))
        flipped = flipped.assign_coords(longitude=flipped.longitude.values[::-1])
        return flipped

    def _shift_lon_back(ds):
        if isinstance(ds, (xarray.DataArray, xarray.Dataset)):
            n = ds.sizes["longitude"]
            return ds.roll(longitude=n // 2, roll_coords=False)
        elif hasattr(ds, '__module__') and ds.__module__.startswith('jaxlib.'):
            n = ds.shape[1]
            return jnp.roll(ds, shift=n // 2, axis=1)
        else:
            raise TypeError("Unsupported input type. Must be xarray DataArray/Dataset or JAX array.")

    if rotation == "revlat":
        new_data = _flip_lat(new_data)

        if isinstance(data, xarray.Dataset):
            new_data = data.copy(deep=True)
            for key in new_data.data_vars:
                if "v_component_of_wind" in key and "10m" not in key:
                    new_data[key] *= -1
                if "10m_v_component_of_wind" in key:
                    new_data[key] *= -1
            return new_data
        if isinstance(data, dict) and all(is_jax_array(v) for v in data.values()):
            new_data = {}
            for key, val in data.items():
                if "v_component_of_wind" in key and "10m" not in key:
                    new_data[key] = -val
                elif "10m_v_component_of_wind" in key:
                    new_data[key] = -val
                else:
                    new_data[key] = val
            return new_data


        #if "time" in new_data.coords:
            #new_data["time"] = new_data.time - np.timedelta64(183, "D")

    elif rotation == "revlon":
        new_data = _flip_lon(new_data)

        for key in new_data.data_vars:
            if "u_component_of_wind" in key and "10m" not in key:
                new_data[key] *= -1
            if "10m_u_component_of_wind" in key:
                new_data[key] *= -1

    elif rotation == "rotlon":
        new_data = _shift_lon_back(new_data)

        #if "time" in new_data.coords:
            #new_data["time"] = new_data.time - np.timedelta64(12, "h")

    return new_data