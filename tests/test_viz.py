import zarr
from correspondence_viz import create_annotation_layer_from_points

default_url = 's3://aind-open-data/exaSPIM_3163606_2023-11-17_12-54-51_alignment_2024-01-09_05-00-44/'

def test_viz():
    points = create_annotation_layer_from_points(default_url)