import polars as pl

MATCH_SCHEMA = {
    "image_id_self": pl.String,
    "point_id_self": pl.Int64,
    "image_id_other": pl.String,
    "point_id_other": pl.Int64,
}

IP_SCHEMA = MATCH_SCHEMA | {
    "point_loc_xyz": pl.Array(pl.Float64, 3),
    "point_intensity": pl.Float32,
}
