class Constants:
    ImageKey = "image"
    MaskKey = "masks"
    AnnotationKey = "annotations"

    PolygonID = "roi_id"
    PolygonName = "roi_name"
    AnnotationImagePath = "raw_image_path"

    histology_features_cache = ".histology_features_cache"
    registration_cache = ".registration_cache"

    histology_save_filename = "histology.ome.tiff"
    spatial_image_save_filename = "spatial_image"
    masks_save_filename = "masks.geojson"
    image_extension = ".ome.tiff"

    rigid_image_key = "rigid_image"
    non_rigid_image_key = "non_rigid_image"
    rigid_annotations_key = "rigid_annotations"
    non_rigid_annotations_key = "non_rigid_annotations"

    image_file_extensions = [".ome.tiff", "ome.tif", ".tiff", ".tiff"]

    shapes_cell_boundaries = "cell_boundaries"
    shapes_cell_circles = "cell_circles"
    shapes_nucleus_boundaries = "nucleus_boundaries"
    points_transcripts = "transcripts"
    images_morphology_focus = "morphology_focus"
    table_key = "table"

    num_image_levels = 5

    concat_sdata_image_key = "images_"
    concat_sdata_shapes_key = "shapes_"


class Opts:
    alignment_valis = "valis"
    mask_ordering_methods = ["left_right", "right_left", "top_bottom", "bottom_top"]
