import pathlib
import tomllib

import typer
from InquirerPy import inquirer

from ._search import search


app = typer.Typer()


@app.command()
def register(config: str):
    from ..io.image import ImageAlign

    config = pathlib.Path(config)

    if config.is_dir():
        config_path = inquirer.fuzzy(
            "Select a config file", choices=search(config, ".toml")
        ).execute()
    elif config.suffix == ".toml":
        config_path = config
    else:
        raise ValueError

    config = tomllib.load(open(config_path, "rb"))

    source_images = config.get("image", None)

    assert (
        source_images is not None
    ), "[[image]] must be provided in the config toml. Got None."

    # TODO: Move this to validate_input
    target_image = None
    for img in source_images:
        if img.get("target_image", None) is not None:
            assert (
                target_image is None
            ), "Multiple target images found. Can only have 1 target image"
            target_image = img

    save_path = config["alignment"].get("save_path", None)
    transformation = config["alignment"].get("transformation", None)

    save_full_slide = config["alignment"].get("save_full_slide", False)

    save_path = pathlib.Path(save_path)
    if not save_path.suffix == ".zarr":
        save_path = save_path.parent / (save_path.name + ".zarr")

    alignment_kwargs = config["alignment"].get("alignment_kwargs", {})

    aligner = ImageAlign(
        source_images=source_images,
        alignment_method="valis",
        save_full_slide=save_full_slide,
        alignment_kwargs=alignment_kwargs,
        save_path=save_path,
        transformation=transformation,
        overwrite=config["alignment"].get("overwrite", False),
        dimensions=config["alignment"].get("dimensions", {"c": 0, "x": 1, "y": 2}),
    )

    aligner.process()
