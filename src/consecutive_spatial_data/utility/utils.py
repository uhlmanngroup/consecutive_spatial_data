import numpy
import pathlib
from spatialdata import SpatialData
from spatialdata.models import SpatialElement
from xarray import DataTree
import warnings
import logging
import tifffile

log = logging.getLogger(__name__)


def remove_array_overlap(array, array_to_remove):
    """Remove elements from array_a that are in array_b"""
    assert array.ndim == 2 and array_to_remove.ndim == 2, "Only works for 2D arrays."

    # Convert to a set of nested tuples
    set_a = set(tuple(map(tuple, array)))
    set_b = set(tuple(map(tuple, array_to_remove)))

    # Remove elements
    new_set_a = set_a - set_b

    # Covnert back to a list of lists
    new_set_a = list(map(list, new_set_a))

    return numpy.array(new_set_a)

def combine_dicts(original, new):
    """Append a new dictionary to a pre-existing one.
    
    Useful for combining the output of a function that creates a 
    dictionary of results, such as an object feature extractor"""
    for key, value in new.items():
        if key in original:
            # If the value is already a list, extend the list with the new value
            if isinstance(new[key], list):
                original[key].extend(value)
            else:
                # If it's not already a list, make it one and extend
                original[key].extend(list(value))
        else:
            # If the key is not in the original dictionary, add it to the original
            original[key] = list(value)

    return original

def get_spatial_element(
    element_dict: dict[str, SpatialElement],
    key: str | None = None,
    return_key: bool = False,
    as_spatial_image: bool = False,
) -> SpatialElement | tuple[str, SpatialElement]:
    """Gets an element from a SpatialData object.

    Args:
        element_dict: Dictionnary whose values are spatial elements (e.g., `sdata.images`).
        key: Optional element key. If `None`, returns the only element (if only one).
        return_key: Whether to also return the key of the element.
        as_spatial_image: Whether to return the element as a `SpatialImage` (if it is a `DataTree`)

    Returns:
        If `return_key` is False, only the element is returned, else a tuple `(element_key, element)`
    """
    assert len(element_dict), "No spatial element was found in the dict."

    if key is not None:
        assert key in element_dict, f"Spatial element '{key}' not found."
        return _return_element(element_dict, key, return_key, as_spatial_image)

    assert (
        len(element_dict) > 0
    ), "No spatial element found. Provide an element key to denote which element you want to use."
    assert (
        len(element_dict) == 1
    ), f"Multiple valid elements found: {', '.join(element_dict.keys())}. Provide an element key to denote which element you want to use."

    key = next(iter(element_dict.keys()))

    return _return_element(element_dict, key, return_key, as_spatial_image)

def _return_element(
    element_dict: dict[str, SpatialElement], key: str, return_key: bool, as_spatial_image: bool
) -> SpatialElement | tuple[str, SpatialElement]:
    element = element_dict[key]

    if as_spatial_image and isinstance(element, DataTree):
        element = next(iter(element["scale0"].values()))

    return (key, element) if return_key else element


def add_spatial_element(
    sdata: SpatialData,
    element_name: str,
    element: SpatialElement,
    overwrite: bool = True,
):
    assert isinstance(element_name, str)
    assert (
        overwrite or element_name not in sdata._shared_keys
    ), f"Trying to add {element_name=} but it is already existing and {overwrite=}"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*already exists. Overwriting it in-memory.")
        sdata[element_name] = element

    if sdata.is_backed():
        try:
            sdata.write_element(element_name, overwrite=overwrite)
        except Exception as e:
            if overwrite:  # force writing the element
                sdata.delete_element_from_disk(element_name)
                sdata.write_element(element_name, overwrite=overwrite)
            else:
                log.error(f"Error while saving {element_name} on disk: {e}")


def relate_keys(keys, unique_id, n_matches=2):
    if not isinstance(unique_id, list):
        unique_id = [unique_id]

    output = {}

    for ui in unique_id:
        matches = [i for i in keys if ui in i]
        if len(matches) == n_matches:
            output[ui] = matches

    return output

def _strip_ome_tiff(s, return_stem: bool = True):
    # Cast to str
    s = str(s)

    s = s.split(".ome.t")[0]

    if return_stem:
        s = pathlib.Path(s)
        return s.stem
    else:
        return s

def reorder_array(array: numpy.ndarray, dim_dict: dict, target_order: str, idx: int = 0) -> numpy.ndarray:
    """
    Transpose an array to a provided dimension order.
    If there are missing dimensions in the target dimension order, drop them by selecting the 0th index.
    
    Args:
        array: Input numpy array
        dim_dict: Dictionary mapping dimension names to their axis indices in the array
        target_order: String specifying the desired order of dimensions (e.g., 'xyz', 'yxc')
    
    Returns:
        Reordered array with only the dimensions specified in target_order
    """
    # Create indexer - select 0th index for dimensions not in target_order

    indexer = [idx] * array.ndim
    
    # Set slice(None) for dimensions we want to keep
    for dim_char in target_order:
        if dim_char in dim_dict:
            indexer[dim_dict[dim_char]] = slice(None)
        else:
            raise ValueError(f"Dimension '{dim_char}' not found in dim_dict")
    
    # Apply indexing to drop unwanted dimensions
    reduced_array = array[tuple(indexer)]
    
    # Get the axes order for transposition
    axes_order = []
    for dim_char in target_order:
        # Find which axis this dimension is now at (after dropping)
        kept_dims = [i for i, idx in enumerate(indexer) if idx == slice(None)]
        original_axis = dim_dict[dim_char]
        new_axis = kept_dims.index(original_axis)
        axes_order.append(new_axis)
    
    # Transpose to target order
    result = numpy.transpose(reduced_array, axes_order)
    
    return result


def get_ome_tiff_shape(file_path):
    with tifffile.TiffFile(file_path) as tif:
        first_series = tif.series[0]  # First image series
        shape = first_series.shape  # Shape of the image array
        return shape
