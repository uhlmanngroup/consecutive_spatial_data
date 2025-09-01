import logging
import os
import pathlib
import re
import typing
from pathlib import Path

import anndata
import dask
import geopandas
import numpy
from dask_image.imread import imread
from multiscale_spatial_image import to_multiscale
from spatial_image import to_spatial_image
from tqdm import tqdm
from xarray import DataArray

import spatialdata

from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import set_transformation
from spatialdata.transformations.transformations import Identity

from .write import MultiscaleImageWriter
from .omero_annotations import yaml_to_polygon
from ..registration import ValisAlignment
from ..segmentation import tissue_detection
from ..spec import Constants, Opts, Xenium
from ..utility.utils import (
    _strip_ome_tiff,
    get_spatial_element,
    reorder_array,
)

log = logging.getLogger(__name__)


class ImageAlign:
    def __init__(
        self,
        source_images: list[dict[str, str]],
        save_path: str,
        save_full_slide: bool = True,
        alignment_method: typing.Literal[Opts.alignment_valis] = Opts.alignment_valis,
        alignment_kwargs: dict = {},
        transformation: str | None = None,
        dimensions: str = {"c": 0, "x": 1, "y": 2},
        overwrite: bool = False,
    ):
        """_summary_

        Args:
            source_images (str | list[str]): _description_
            alignment_method (typing.Literal[Opts.alignment_valis], optional): _description_. Defaults to Opts.alignment_valis.
            mask_ids (typing.List[str], optional): _description_. Defaults to None.
            source_image_mask_downsample_scale (typing.Tuple[int], optional): _description_. Defaults to None.
            mask_to_align (str | list[str] | None, optional): _description_. Defaults to None.
            source_image_mask_order_method (typing.Literal[], optional): _description_. Defaults to "left_right".
            annotations (dict[str, str] | None, optional): _description_. Defaults to None.
            masking_kwargs (dict, optional): _description_. Defaults to {}.
            dimensions (_type_, optional): _description_. Defaults to {"c": 0, "x": 1, "y": 2}.
        """

        # Image to be transformed
        self.source_images = source_images
        self.processed_source_images = []

        self.annotations = {}
        self.tables = {}

        # Information on axes
        self.dimensions = dimensions
        # Path to image to be aligned to
        self.alignment_method = alignment_method
        self.save_path = save_path
        self.alignment_kwargs = alignment_kwargs
        self.cache = save_path / Constants.histology_features_cache
        self.transformation = transformation
        self.save_full_slide = save_full_slide
        self.overwrite = overwrite

        self.target_image_path = None
        self.target_sdata_path = None

        self._validate_input()

    def _validate_input(self):
        assert isinstance(
            self.source_images, list
        ), "Source images is expected to be a list."

        for img_info in self.source_images:
            assert isinstance(
                img_info, dict
            ), "Elements of source_images are expected to be a dictionary"
            assert (
                "path" in img_info.keys()
            ), f"Image dict does not possess 'path' key, which is required: {img_info}"
            if (img_info["path"].endswith(".zarr") or img_info["path"].startswith("spatialdata:")) and (
                "mask_ids" in img_info.keys() or "mask_to_align" in img_info.keys()
            ):
                raise ValueError(
                    "Found mask information for a spatial data object, which is not supported."
                )
            if img_info["path"].endswith(".zarr") or img_info["path"].startswith("spatialdata:"):
                if img_info.get("spatialdata_element_name", None) is not None:
                    assert isinstance(
                        img_info["spatialdata_element_name"], list
                    ), "spatialdata_element_name must be a list."
                    assert (
                        len(img_info["spatialdata_element_name"]) == 1
                    ), "Only one spatialdata_element_name per SpatialData zarr is supported."
            elif img_info["path"].endswith(tuple(Constants.image_file_extensions)):
                if img_info.get("target_image_path"):
                    raise ValueError(
                        "Only SpatialData objects can be used as the target image for alignment."
                    )
                assert isinstance(img_info["path"], str), "Path should be a str"
                assert len(set(img_info["mask_ids"])) == len(
                    img_info["mask_ids"]
                ), "Duplicate mask_ids found."
                if (
                    img_info["mask_to_align"] is not None
                    and img_info["mask_ids"] is None
                ):
                    raise ValueError(
                        "mask_to_align is provided by mask_ids is None. Both need to be provided."
                    )
                elif (
                    img_info["mask_to_align"] is None
                    and img_info["mask_ids"] is not None
                ):
                    raise ValueError(
                        "mask_to_align is not provided by mask_ids is provided. Both need to be provided."
                    )
                elif img_info.get("target_image", None) is not None:
                    raise ValueError(
                        "Currently, only spatial images as target_image is supported."
                    )
                elif img_info.get("spatialdata_element_name", None) is not None:
                    assert isinstance(
                        img_info["spatialdata_element_name"], list
                    ), "spatialdata_element_name must be a list."
                    assert len(img_info["spatialdata_element_name"]) == len(
                        img_info["mask_to_align"]
                    ), "spatialdata_element_name must have same length of mask_to_align"

    def _set_alignment_method(self):
        if self.alignment_method.casefold() == Opts.alignment_valis:
            self.alignment_method = ValisAlignment(
                source_image=self.processed_source_images,
                target_image=self.target_image_path,
                spatial_data=self.annotations,
                save_path=self.cache,
                save_full_slide=self.save_full_slide,
                transformation=self.transformation,
                alignment_kwargs=self.alignment_kwargs,
                overwrite=self.overwrite,
            )
        else:
            raise ValueError(
                f"Alignment method {self.alignment_method} not recognised."
            )

    def _get_registered_image(self):
        log.info(
            f"Aligning the following images:\n{chr(10).join(self.processed_source_images)}"
        )

        output = self.alignment_method.process()

        return output

    def _get_image(
        self, img_path, source_image_mask_downsample_scale: tuple[int] = [1, 1, 1]
    ):
        histology_image = imread(
            img_path,
        )  # .squeeze().transpose(2, 0, 1) # TODO: Remove this squeeze and transpose

        histology_image = reorder_array(
            histology_image, dim_dict=self.dimensions, target_order="cyx"
        )

        if source_image_mask_downsample_scale is not None:
            assert (
                len(source_image_mask_downsample_scale) == histology_image.ndim
            ), f"Number of scaling elements ({len(source_image_mask_downsample_scale)}) must match the number of image dimensions ({histology_image.ndim})"

            # Perform simple downsampling to speed up tissue masking
            histology_image = histology_image[
                # *_get_scale_slices(self.source_image_mask_downsample_scale)
                tuple(_get_scale_slices(source_image_mask_downsample_scale))
            ]

        return histology_image

    def _get_masks(self):
        for img_info in self.source_images:
            mask_to_align = img_info.get("mask_to_align", None)
            if mask_to_align is not None:
                img_path = img_info["path"]

                source_image_mask_downsample_scale = img_info.get(
                    "source_image_mask_downsample_scale", None
                )
                masking_kwargs = img_info.get("masking_kwargs", {})
                source_image_mask_order_method = img_info.get(
                    "source_image_mask_order_method"
                )
                mask_ids = img_info.get("mask_ids")
                annotations = img_info.get("annotations", None)

                # If ome.tiff, calculate masks
                if img_path.endswith(tuple(Constants.image_file_extensions)):
                    # Path to save the calculated masks to
                    img_path = pathlib.Path(img_path)
                    # Save the mask path with the same name as the image it was derived from
                    mask_save_path = pathlib.Path(
                        self.cache, img_path.name, Constants.masks_save_filename
                    )
                    mask_save_path.parent.mkdir(parents=True, exist_ok=True)
                    # Check if it's already been created
                    if not mask_save_path.is_file():
                        log.info("Computing masks")
                        image = self._get_image(
                            img_path, source_image_mask_downsample_scale
                        )
                        masks = tissue_detection(
                            # cv2 expects XYC channel order
                            # but histology_features uses cyx
                            image.transpose(1, 2, 0).compute(),
                            outer_contours_only=True,
                            return_poly=True,
                            **masking_kwargs,
                        )

                        masks = geopandas.GeoDataFrame({"geometry": masks})

                        # Scale up the masks to the original size
                        if source_image_mask_downsample_scale:
                            masks["geometry"] = masks["geometry"].scale(
                                xfact=source_image_mask_downsample_scale[
                                    self.dimensions["x"]
                                ],
                                yfact=source_image_mask_downsample_scale[
                                    self.dimensions["y"]
                                ],
                                origin=(0, 0),
                            )

                        # Order the masks
                        masks = sort_polygons(masks, source_image_mask_order_method)

                        # Save the cache
                        masks.to_file(mask_save_path)

                    # Load the mask polygons since they've previously been calculated
                    else:
                        log.info(f"Loading masks from cache: {mask_save_path}")
                        # Load the cache
                        masks = geopandas.read_file(mask_save_path)

                assert len(masks) == len(
                    mask_ids
                ), f"Number of masks detected in the source image ({len(masks)}) did not match the number of provided mask IDs ({len(mask_ids)}). Adjust masking_kwargs."

                # Define variable that will later represent the computed dask array
                image = None

                # Process each mask
                for i, mta in enumerate(mask_to_align):
                    # If an alternative name is available, use this. The saved
                    # file name is the same name that will be used as a SpatialData
                    # key.
                    if img_info.get("spatialdata_element_name", None) is not None:
                        save_name = (
                            img_info.get("spatialdata_element_name")[i]
                            + Constants.image_extension
                        )
                    else:
                        save_name = (
                            _strip_ome_tiff(img_path.name)
                            + f"_{mta}"
                            + Constants.image_extension
                        )

                    save_path = pathlib.Path(self.cache, save_name)

                    # If the cropped images do not already exist, create them
                    if not save_path.is_file():
                        if image is None:
                            # Load and compute the image once
                            image = self._get_image(img_path).compute()
                        # Get the relevant mask
                        # Keep as list for adding to geopandas later
                        mask = masks.loc[mask_ids.index(mta), "geometry"]

                        # Get the crop bbox
                        bbox = _get_mask_bounding_box(mask, pad=0.1)

                        y_slice, x_slice = _bbox_to_slice(bbox)

                        cropped_image = image[:, y_slice, x_slice]

                        # Save the cropped image. Valis needs a path, so we will provide this one
                        log.info(
                            f"Saving mask crop {mta} with shape {cropped_image.shape} to {save_path}"
                        )

                        if annotations is not None:
                            # Load the annotations that are found in this mask ROI
                            loaded_annotations = self._get_annotations(
                                annotations, y_slice, x_slice
                            )

                            if len(loaded_annotations) > 0:
                                # Cache the annotations, since calculating the masks
                                # is computationally intensive and we need the masks for subsetting.
                                loaded_annotations.to_file(save_path.with_suffix(".geojson"))
                                self.annotations[save_path] = loaded_annotations
                            else:
                                log.warning(
                                    f"No annotations found to be associated with {save_path.name}"
                                )

                        self.processed_source_images.append(
                            str(_save_image(cropped_image, save_path))
                        )
                    # Cropped histology already exists. Add its path to processed_source_images
                    # which can be loaded later
                    else:
                        self.processed_source_images.append(str(save_path))
                        # Load annotations if they exist
                        if (save_path.with_suffix(".geojson")).is_file():
                            self.annotations[str(save_path)] = geopandas.read_file(
                                save_path.with_suffix(".geojson")
                            )
                        else:
                            log.warning(
                                f"No annotations found to be associated with {save_path.name}"
                            )

    def _get_spatial_annotations(self, sdata_path):
        """Load GeoDataFrames from any spatial experiments.
        Extend any existing annotations"""

        sdata = spatialdata.read_zarr(
            sdata_path,
            selection=["shapes"],
        )
        spatial_annotations = get_spatial_element(
            sdata.shapes, Constants.shapes_cell_boundaries
        )

        # Convert spatial coords from micron scale to pixel scale
        spatial_annotations["geometry"] = spatial_annotations.scale(
            xfact=1 / Xenium.pixel_size.value,
            yfact=1 / Xenium.pixel_size.value,
            origin=(0, 0),
        )

        # Update the SpatialData transform to identity.
        # When reading Xenium polygons, spatialdata_io adds a micron -> scale transform.
        # We are removing this transform here.
        set_transformation(sdata[Constants.shapes_cell_boundaries], Identity())

        # spatial_annotations = {}

        # sdata = spatialdata.read_zarr(
        #     sdata_path,
        #     selection=["shapes", "points"],
        # )

        # geo_keys = {
        #     "shapes": Constants.shapes_cell_boundaries,
        #     "shapes": Constants.shapes_cell_circles,
        #     "shapes": Constants.shapes_nucleus_boundaries,
        #     # "points": Constants.points_transcripts,
        # }

        # for group, gk in geo_keys.items():
        #     geo = get_spatial_element(
        #         getattr(sdata, group), gk
        #     )

        #     if gk == Constants.points_transcripts:
        #         # By default, Xenium transcripts do not contain a geo
        #         # column. Instead, they have XYZ columns. Let's fix this.
        #         geo['geometry'] = geo.apply(lambda row: shapely.Point(row['x'], row['y']), axis=1)

        #         # Loads into memory. Ideally, this should stay lazy until saving.
        #         geo = geopandas.GeoDataFrame(geo)
        #         geo = geo.set_geometry("geometry")

        #     # Convert spatial coords from micron scale to pixel scale
        #     geo["geometry"] = geo.scale(
        #         xfact=1/Xenium.pixel_size.value,
        #         yfact=1/Xenium.pixel_size.value,
        #         origin=(0, 0),
        #     )

        #     spatial_annotations[f"{group}_{geo}"] = geo

        return spatial_annotations

    def _get_spatial_table(self, sdata_path):
        """Load GeoDataFrames from any spatial experiments.
        Extend any existing annotations"""

        sdata = spatialdata.read_zarr(
            sdata_path,
            selection=["table", "tables"],
        )
        table = get_spatial_element(sdata.tables, Constants.table_key)
        return table

    def _get_concatenated_tables(self):
        tables = {pathlib.Path(k).name: v for k, v in self.tables.items()}

        tables = anndata.concat(tables, label="anndata_source")

        return tables

    def _get_annotations(
        self, annotation_source: str, y_slice: slice, x_slice: slice
    ):
        """Load annotations from various sources: yaml directory, parquet file, or SpatialData shapes key."""
        
        if os.path.isdir(annotation_source):
            # Directory of YAML files (original behavior)
            return self._get_omero_annotations(annotation_source, y_slice, x_slice)
        elif annotation_source.endswith('.parquet'):
            # Parquet file containing geodataframe
            return self._get_parquet_annotations(annotation_source, y_slice, x_slice)
        elif annotation_source.startswith('spatialdata:'):
            # SpatialData shapes key format: "spatialdata:path/to/file.zarr:shapes_key"
            return self._get_spatialdata_annotations(annotation_source, y_slice, x_slice)
        else:
            raise ValueError(f"Unsupported annotation source format: {annotation_source}")

    def _get_omero_annotations(
        self, annotation_path: str, y_slice: slice, x_slice: slice
    ):
        """Load annotations from directory of YAML files (original implementation)."""
        
        polygons = {
            "geometry": [],
            Constants.PolygonID: [],
            Constants.PolygonName: [],
            Constants.AnnotationImagePath: [],
        }
        for f in os.listdir(annotation_path):
            # TODO: Move this
            _polygon_data = yaml_to_polygon(os.path.join(annotation_path, f))
            polygons["geometry"].append(_polygon_data["shapely_polygon"])
            polygons[Constants.PolygonID].append(_polygon_data[Constants.PolygonID])
            polygons[Constants.PolygonName].append(_polygon_data[Constants.PolygonName])
            polygons[Constants.AnnotationImagePath].append(
                _polygon_data[Constants.AnnotationImagePath]
            )

        polygons = geopandas.GeoDataFrame({**polygons})

        polygons = _subset_polygons(polygons, y_slice, x_slice, reset_origin=True)

        return polygons

    def _get_parquet_annotations(
        self, parquet_path: str, y_slice: slice, x_slice: slice
    ):
        """Load annotations from a parquet file containing a geodataframe."""
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
        polygons = geopandas.read_parquet(parquet_path)
        
        # Ensure it has a geometry column
        if 'geometry' not in polygons.columns:
            raise ValueError(f"Parquet file must contain a 'geometry' column: {parquet_path}")
        
        polygons = _subset_polygons(polygons, y_slice, x_slice, reset_origin=True)

        return polygons

    def _get_spatialdata_annotations(
        self, spatialdata_source: str, y_slice: slice, x_slice: slice
    ):
        """Load annotations from a SpatialData shapes key."""
        
        # Parse the spatialdata source format: "spatialdata:path/to/file.zarr:shapes_key"
        parts = spatialdata_source.split(':', 2)
        if len(parts) != 3 or parts[0] != 'spatialdata':
            raise ValueError(f"Invalid SpatialData source format. Expected 'spatialdata:path/to/file.zarr:shapes_key', got: {spatialdata_source}")
        
        _, sdata_path, shapes_key = parts
        
        if not os.path.exists(sdata_path):
            raise FileNotFoundError(f"SpatialData file not found: {sdata_path}")
        
        sdata = spatialdata.read_zarr(sdata_path, selection=["shapes"])
        
        if shapes_key not in sdata.shapes:
            available_keys = list(sdata.shapes.keys())
            raise KeyError(f"Shapes key '{shapes_key}' not found in SpatialData. Available keys: {available_keys}")
        
        polygons = sdata.shapes[shapes_key]
        
        # Convert to regular GeoDataFrame if it's not already
        if not isinstance(polygons, geopandas.GeoDataFrame):
            polygons = geopandas.GeoDataFrame(polygons)
        
        polygons = _subset_polygons(polygons, y_slice, x_slice, reset_origin=True)

        return polygons

    def _parse_spatialdata_path(self, spatialdata_source: str):
        """Parse spatialdata: format path and return sdata_path and key."""
        parts = spatialdata_source.split(':', 2)
        if len(parts) != 3 or parts[0] != 'spatialdata':
            raise ValueError(f"Invalid SpatialData source format. Expected 'spatialdata:path/to/file.zarr:key', got: {spatialdata_source}")
        
        _, sdata_path, key = parts
        return sdata_path, key

    def _get_spatial_images(self):
        """Save spatial images to disk for registration"""
        for img_info in tqdm(self.source_images):
            img_path = img_info.get("path")
            save_name = img_info.get("spatialdata_element_name", None)[0]
            if img_path.endswith(".zarr") or img_path.startswith("spatialdata:"):
                
                # Handle spatialdata: format
                if img_path.startswith("spatialdata:"):
                    sdata_path, image_key = self._parse_spatialdata_path(img_path)
                    actual_img_path = pathlib.Path(sdata_path)
                else:
                    actual_img_path = pathlib.Path(img_path)
                    image_key = Constants.images_morphology_focus  # Default key
                
                if save_name is None:
                    save_name = (
                        actual_img_path.stem + "_" + Constants.spatial_image_save_filename,
                    )
                save_path = pathlib.Path(
                    self.cache, save_name + Constants.image_extension
                )

                if img_info.get("target_image", False):
                    self.target_image_path = str(save_path)
                    self.target_sdata_path = actual_img_path
                self.processed_source_images.append(str(save_path))

                # Use the cache, if present
                if not save_path.is_file():
                    log.info(
                        "Converting SpatialData images to ome.tiff for registration"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    sdata = spatialdata.read_zarr(
                        actual_img_path,
                        selection=["images"],
                    )
                    # Get the multiscale spatial image using specified or default key
                    image = get_spatial_element(
                        sdata.images,
                        image_key,
                        as_spatial_image=True,
                    )
                    log.info(f"Saving spatial image for registration at {save_path}")
                    _ = _save_image(image, save_path)

                # Get the shapes and points associated with this spatialdata object
                annotations = self._get_spatial_annotations(actual_img_path)
                self.annotations[save_path] = annotations

                table = self._get_spatial_table(actual_img_path)
                self.tables[save_path] = table

    def _associate_annotations_with_masks(self):
        """Link loaded annotations with masks"""

        self.annotations.sjoin(
            self.masks, predicate="intersects", rsuffix="parent_mask"
        )

    def _create_sdata(self, output):
        if not self.save_full_slide:
            log.warning(
                f"""save_full_slide is {self.save_full_slide}, so if registration has not 
            previously saved slides, they will not be added to the output SpatialData object. This can result
            in a SpatialData object with only a table element."""
            )
        images = _get_dict_subset(output, Constants.concat_sdata_image_key)
        shapes = _get_dict_subset(output, Constants.concat_sdata_shapes_key)

        if len(images) == 0:
            log.info("No full slide alignments detected. Exiting.")
            return

        images = {
            _replace_space(k): Image2DModel.parse(
                imread(v),
                chunks=(1, 4096, 4096),
                scale_factors=[2, 2, 2, 2],
            )
            for k, v in images.items()
        }

        shapes = {
            _replace_space(k): ShapesModel.parse(
                v,
            )
            for k, v in shapes.items()
        }

        # Add tables taken from multiple SpatialData objects. They will all
        # be concatenated into a single table and added to the final sdata object
        if self.tables:
            table = self._get_concatenated_tables()
            table = {"table": TableModel.parse(table)}
        else:
            table = {}

        # if self.target_sdata_path is not None:
        #     """
        #     TODO: Make this work.

        #     Currently, calling add_spatial_element supports addition of a single
        #     data object (eg. anndata). This need to be updated to support
        #     writing of multiple data objects.
        #     """
        #     sdata = spatialdata.read_zarr(self.target_sdata_path)
        #     log.info(f"Writing aligned data to existing SpatialData object: {self.target_sdata_path}")
        #     # Add images
        #     add_spatial_element(
        #         sdata = sdata,
        #         element_name = "images",
        #         element = images,
        #         overwrite = False,
        #     )
        #     # Add shapes
        #     add_spatial_element(
        #         sdata = sdata,
        #         element_name = "shapes",
        #         element = shapes,
        #         overwrite = False,
        #     )
        #     # Add table, which is concatenated
        #     add_spatial_element(
        #         sdata = sdata,
        #         element_name = "tables",
        #         element = table,
        #         overwrite = True,
        #     )
        # else:
        #     sdata = spatialdata.SpatialData(
        #         images=images,
        #         shapes=shapes,
        #         tables=table,
        #     )

        #     return sdata

        # For now, we will just create a new SpatialData object
        sdata = spatialdata.SpatialData(
            images=images,
            shapes=shapes,
            tables=table,
        )

        return sdata

    def process(self):
        # Find masks. This will also crop and save the ROI, which
        # ensures only the cropped ROI is used for registration.
        self._get_masks()

        # For any spatial data, load and save the image as an ome.tiff
        self._get_spatial_images()

        # Perform image registration
        self._set_alignment_method()

        output = self._get_registered_image()

        save_path = Path(self.save_path)
        save_path = Path(save_path, save_path.stem + ".zarr")

        if self.save_full_slide:
            # Only construct the spatialdata if it can be saved.
            # ie. Doesn't exist or can be overwritten.
            if not save_path.exists() or self.overwrite:
                sdata = self._create_sdata(output)
                sdata.write(save_path, overwrite=True)
            else:
                log.warning("Overwrite is False so SpatialData will not be saved.")
        else:
            log.info(
                f"save_full_slice is {self.save_full_slide} so SpatialData will not be saved."
            )

def _get_scale_slices(scale: typing.Union[int, typing.List[int]]):
    if isinstance(scale, int):
        scale = [scale]
    slices = []
    for s in scale:
        assert s >= 1, "Scale cannot be less than 1."
        slices.append(slice(None, None, s))

    return slices


def _subset_polygons(
    gdf: geopandas.GeoDataFrame, y_slice, x_slice, reset_origin: bool = True
):
    gdf = gdf.cx[x_slice, y_slice]
    if reset_origin:
        gdf["geometry"] = gdf["geometry"].translate(
            xoff=-x_slice.start, yoff=-y_slice.start
        )

    return gdf


def _get_mask_bounding_box(mask, pad=None):
    bbox = mask.envelope
    if pad:
        if isinstance(pad, float):
            # Pad by a fraction of the major axis
            xmin, ymin, xmax, ymax = bbox.bounds
            width = xmax - xmin
            height = ymax - ymin
            pad = max(width, height) * pad
        bbox = bbox.buffer(pad)
    return bbox


def _get_dict_subset(data, subset):
    filtered_data = {
        key[len(subset) :]: value
        for key, value in data.items()
        if key.startswith(subset)
    }

    return filtered_data


def _replace_space(s: str) -> str:
    return re.sub(r"\s+", "_", s)


def _bbox_to_slice(bbox):
    bounds = numpy.array(bbox.bounds).astype(int)
    # Negative slices are set to 0
    bounds = numpy.where(bounds < 0, 0, bounds)
    xmin, ymin, xmax, ymax = bounds
    y, x = slice(ymin, ymax), slice(xmin, xmax)
    return y, x


def _save_image(image, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, (numpy.ndarray, dask.array.Array)):
        image = to_spatial_image(image, dims=("c", "y", "x"))

    if isinstance(image, DataArray):
        # Convert to DataTree, but only with one scale.
        image = to_multiscale(
            image, [2] * (Constants.num_image_levels - 1)  # - 1 since we already have 1
        )

    MultiscaleImageWriter(image, pixel_size=1, tile_width=1024).write(save_path)

    return save_path


def sort_polygons(
    gdf: geopandas.GeoDataFrame,
    method: typing.Literal["left_right", "right_left", "top_bottom", "bottom_top"],
):
    if method.casefold() == "left_right":
        accessor = "minx"
    elif method.casefold() == "right_left":
        accessor = "maxx"
    elif method.casefold() == "top_bottom":
        accessor = "maxy"
    elif method.casefold() == "bottom_top":
        accessor = "miny"
    else:
        raise ValueError(f"Method {method} not recognised.")

    gdf[accessor] = getattr(gdf.bounds, accessor)
    gdf = gdf.sort_values(by=accessor)
    gdf = gdf.drop(columns=accessor).reset_index(drop=True)

    return gdf
