import numpy
import shapely
import yaml

def yaml_to_polygon(yaml_file):
    if isinstance(yaml_file, str):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
    elif isinstance(yaml_file, dict):
        data = yaml_file
    else:
        raise ValueError(
            f"Cannot load yaml of type {type(yaml_file)}. Expected path str or dict"
        )

    polygon = shapely.wkt.loads(data["poly"])

    data["shapely_polygon"] = polygon

    data["numpy_polygon"] = numpy.array(polygon.exterior.coords)

    return data
