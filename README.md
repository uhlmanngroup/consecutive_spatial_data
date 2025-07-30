# consecutive_spatial_data

`consecutive_spatial_data` provides a CLI, configurable with a `config.toml` file, to align histology images with spatial transcriptomics data. `consecutive_spatial_data` has been designed with Xenium spatial transcriptomics in mind, but it could feasibly work for any other SpatialData compatible method.

The package can be used for the following scenarios:
- Automated alignment of H&E stained image following spatial transcriptomics
- If adajcent sections of tissue have been stained with H&E and you would like to align them to a nearby spatial transcriptomics slice. This enables 3D reconstruction of tissue and acquired spatial transcriptomics data

Some features of the package:
- Automated region of interest identification
  - If there are multiple sections per image, you can configure the `config.toml` so that each section is segmented and aligned individually. Sections can also be skipped, if required (eg. low quality).
- Supports registration of annotation polygons
  - For example, omero annotations of histology are aligned to the spatail transcriptomics dataset
- Using the Valis backend, both rigid and non-rigid registrations are performed. Both forms are saved to the SpatialData object.

The package centers around the [SpatialData](https://github.com/scverse/spatialdata) format and aligned images are saved within a single SpatialData object.

Currently, `consecutive_spatial_data` uses Valis as the backend for registration, but any other registration method can be added as long as the created classes possesses a `process()` method.

## Usage

For handling the Valis dependency, it's best to run `consecutive_spatial_data` inside the Valis container [found here](https://hub.docker.com/r/cdgatenbee/valis-wsi/tags). Valis 1.1.0 was used when developing `consecutive_spatial_data`.

Example config can be [found here](config/example_config.toml).

```python
consecutive_spatial_data register config.toml
```