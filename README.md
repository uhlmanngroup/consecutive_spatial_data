# `consecutive_spatial_data`

`consecutive_spatial_data` is a CLI tool for aligning histology images with spatial transcriptomics data. It is configurable via a `config.toml` file and designed primarily with Xenium spatial transcriptomics in mind, though it is compatible with any method supported by the [SpatialData](https://github.com/scverse/spatialdata) ecosystem.

## Key Use Cases

- **Align H&E-stained images to spatial transcriptomics data**  
  Automate the alignment of histology images acquired following spatial transcriptomics protocols.
  
- **Align adjacent tissue sections for 3D reconstruction**  
  Register histological sections from adjacent tissue slices to reconstruct a 3D representation of the tissue alongside spatial transcriptomics data.

## Features

- **Automated Region of Interest (ROI) Detection**  
  - Automatically detect and segment multiple tissue sections per image.  
  - Fine-grained control via the `config.toml`: skip poor-quality sections or align sections individually.

- **Annotation Polygon Support**  
  - Align polygonal annotations (e.g., from Omero) to spatial transcriptomics data.

- **Flexible Registration Options**  
  - Uses [Valis](https://github.com/choosehappy/valis) for both rigid and non-rigid image registration.  
  - Outputs are saved directly to a SpatialData object.

- **Modular Backend Design**  
  - While Valis is the default backend, alternative registration methods can be integrated by implementing a `process()` method in a compatible class.

## Getting Started

We recommend running `consecutive_spatial_data` inside the Valis Docker container for consistent results and dependency management. The container used during development was: [cdgatenbee/valis-wsi:1.1.0](https://hub.docker.com/r/cdgatenbee/valis-wsi/tags)

### Example Usage

```bash
git clone github.com/callum-jpg/consecutive_spatial_data
cd consecutive_spatial_data
pip install -e .

consecutive_spatial_data register config.toml
```

An example configuration file can be found in [`config/example_config.toml`](config/example_config.toml).
