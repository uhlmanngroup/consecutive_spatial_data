from pathlib import Path

import numpy
import shapely
import geopandas
import dask
import logging
import typing
from dask.diagnostics import ProgressBar

from ..spec import Constants
from ..utility import _strip_ome_tiff, get_ome_tiff_shape


log = logging.getLogger(__name__)

class ValisAlignment:
    def __init__(
        self,
        source_image: str | list[str],
        save_path: str,
        save_full_slide: bool = True,
        target_image: str | None = None,
        spatial_data: dict[str, geopandas.GeoDataFrame] | None = None,
        transformation: typing.Literal["rigid", "non_rigid", "both"] = "rigid",
        alignment_kwargs: dict = {}, 
        overwrite: bool = False,
    ):
        if target_image is not None:
            assert target_image in source_image, "Target image was not found in the image list"
        self.source_image = source_image
        self.target_image = target_image
        self.spatial_data = spatial_data
        if transformation.casefold() in ["non-rigid", "non_rigid"]:
            self.do_rigid = [False]
        elif transformation.casefold() == "rigid":
            self.do_rigid = [True]
        elif transformation.casefold() == "both":
            self.do_rigid = [True, False]
        self.output = {}
        self.save_path = save_path
        self.save_full_slide = save_full_slide
        self.alignment_kwargs = alignment_kwargs
        self.overwrite = overwrite
        
        log.info(f"Alignment cache: {self.save_path}")

        self._init_valis()

    def _init_valis(self):
        try:
            from valis import registration
            # from valis import feature_detectors
            # from valis.micro_rigid_registrar import MicroRigidRegistrar
        except ImportError:
            import traceback
            traceback.print_exc()
            raise ImportError("Please install Valis")

        self.registrar = registration.Valis(
            "",
            self.save_path,

            reference_img_f=self.target_image,
            align_to_reference=True if self.target_image else False,
            # align_to_reference=True,

            img_list=self.source_image,
            # max_processed_image_dim_px=1024,
            # Always start from a rigid transformation
            do_rigid = True, 
            create_masks=False,
            # max_non_rigid_registration_dim_px=15_000,
            # max_non_rigid_registration_dim_px=5_000,
            # max_non_rigid_registration_dim_px=5_00,
            # max_processed_image_dim_px=1024,
            # max_processed_image_dim_px=2048, # Higher rigid registration resolution
            # max_image_dim_px=256, # If you don't set this, the min max_processed_image_dim_px is 1024

            ### Can't pass the processor_dict to MicroRigidRegistrar so for multichannel
            ### images it doesn't work as expected. Seems to be a limitation of Valis.
            # micro_rigid_registrar_cls=MicroRigidRegistrar,
            # micro_rigid_registrar_params = {"scale": 1**3, "tile_wh": 2**12},

            **self.alignment_kwargs,
        )

    def process(self):
        log.info("Running Valis for alignment")
        self._register()
        if self.save_full_slide:
            # Don't warp gdf if not saving the full slide. 
            # This is because we want to assess registration 
            # performance at the image level first. 
            self._warp_gdf()
            self._save_slides()

        return self.output

    def _register(self, **kwargs):
        from valis import preprocessing

        # Check if an image is RGB or not based on the 0th channel dimension
        rgb_info = [get_ome_tiff_shape(i)[0] == 3 for i in self.source_image]

        print("self.source_image", self.source_image)

        # For .zarr-derived images, select the first channel (DAPI), otherwise use all channels for RGB images (histology)
        # This isn't ideal since a user may require other than the 0th channel. Can later
        # add the channel selection as a kwarg
        processor_dict = {
                   img_name: (
                        preprocessing.HEDeconvolution if is_rgb
                        else [preprocessing.ChannelGetter, {"channel": 0, "adaptive_eq": True}]
                    )
                for img_name, is_rgb in zip(self.source_image, rgb_info)
            }

        self.registrar.register(
            processor_dict = processor_dict,
        )

        # self.registrar.register_micro(
        #     processor_dict = processor_dict,
        #     max_non_rigid_registration_dim_px=2048, 
        #     align_to_reference=False
        # )


        # TODO: should we enable this?
        # self.registrar.register_micro(
        #     max_non_rigid_registration_dim_px=2048,
        #     processor_dict=processor_dict,
        # )

    def _warp_gdf(
        self,
        ):

        if self.registrar is None:
            raise ValueError("Run Valis.register before warp_gdf.")

        if self.spatial_data is None:
            raise ValueError("No spatial_data found. Can't run warping.")

        for do_rigid in self.do_rigid:
            for source_image, gdf in self.spatial_data.items():
                source_image = str(source_image)
                # Don't warp the target image polygons since they have not
                # been transformed
                if source_image != self.target_image:
                    log.info(f"Warping shapes for {source_image}")
                    warped_gdf = gdf.copy()
                    polygons = warped_gdf["geometry"].values
                    delayed_polygons = [
                        _warp_polygon(
                            pg=pg, 
                            registrar=self.registrar,
                            source_image=source_image, 
                            rigid=do_rigid,
                        ) for pg in polygons
                    ]

                    with ProgressBar(minimum=1):
                        new_pg = dask.compute(*delayed_polygons)

                    log.info(f"Shape warping complete for {source_image}")

                    warped_gdf["geometry"] = new_pg

                    self.output[Constants.concat_sdata_shapes_key + _strip_ome_tiff(source_image)] = warped_gdf
                else:
                    self.output[Constants.concat_sdata_shapes_key + "target_image_shapes_" + _strip_ome_tiff(source_image)] = gdf


    def _save_slides(self):
        for do_rigid in self.do_rigid:
            for img_name in self.registrar.original_img_list:
                img_name = str(img_name)
                # Don't warp the target image since it hasn't been transformed
                if img_name != self.target_image:
                    save_path = Path(
                        self.save_path,
                        f"{Constants.non_rigid_image_key if not do_rigid else Constants.rigid_image_key}_"
                        + Path(img_name).stem
                        + ".tiff",
                    )
                    
                    # Don't recreate if the file already exists
                    if not save_path.is_file() or self.overwrite:
                        log.info(f"Saving {save_path}")
                        # Get the registered slide
                        slide_obj = self.registrar.get_slide(img_name)
                        
                        slide_obj.warp_and_save_slide(
                            save_path, 
                            non_rigid=not do_rigid,
                            # Here, we perform an reference crop. This ensures that the aligned
                            # images all have the same shape. This is useful when performing
                            # a bounding box query and you need to slice multiple images with 
                            # the same XY coordinates.
                            # Additionally, it saves space by discarding parts of the image that
                            # do not overlap.
                            crop = "reference" if self.target_image is not None else False
                        )
                    else:
                        log.warning(f"Registered image already exists. Will not save {save_path}")

                    self.output[Constants.concat_sdata_image_key +  _strip_ome_tiff(save_path)] = save_path
                else:
                    self.output[Constants.concat_sdata_image_key + "target_image_" + _strip_ome_tiff(img_name)] = img_name



@dask.delayed
def _warp_polygon(
    pg, 
    registrar, 
    source_image, 
    rigid: bool
    ):

    source_slide = registrar.get_slide(source_image)

    xy = numpy.array(pg.exterior.coords.xy)

    warped_xy = source_slide.warp_xy(
        xy.T, 
        non_rigid = not rigid,
        # We do not set the crop arg here as the Valis object
        # initialisation crop method will be used. We do not perform
        # "overlap" cropping as there is no concept of overlapping image
        # regions for XY coordinates. To perform "overlap" cropping will
        # lead to an erroneous shift of the polygons.
    )
    return shapely.Polygon(warped_xy)
