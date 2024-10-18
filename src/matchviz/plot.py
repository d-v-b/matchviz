import matplotlib.pyplot as plt
import polars as pl
import numpy as np


def plot_matches_gird(
    *, data: pl.DataFrame, dataset_name: str, invert_x, invert_y
) -> plt.Figure:
    fig_w = 8
    fig_h = 8

    fig, axs = plt.subplots(figsize=(fig_w, fig_h))

    if invert_y:
        axs.invert_yaxis()
    if invert_x:
        axs.invert_xaxis()

    axs.spines[["right", "top"]].set_visible(False)
    axs.grid(True, alpha=0.5)
    completed_pairs = set()
    completed_points = set()
    axs.set_xlabel("Image x coordinate (nm)")
    axs.set_ylabel("Image y coordinate (nm)")

    for row in data.rows():
        row_model = dict(zip(data.schema, row))
        id_self = row_model["image_id_self"]
        id_other = row_model["image_id_other"]
        name_self = row_model["image_name_self"]
        coords_self = row_model["image_origin_self"][:2]

        if name_self not in completed_points:
            axs.scatter(*coords_self, label=name_self, marker=f"${id_self}$", s=200)
            completed_points.add(name_self)

        if (id_self, id_other) not in completed_pairs:
            rows_other = data.filter(
                pl.col("image_id_self") == row_model["image_id_other"]
            )
            coords_other = rows_other.select("image_origin_self").to_numpy()[0][0][:2]

            # ensure that we don't display symmetric pairs
            completed_pairs.add((id_self, id_other))
            completed_pairs.add((id_other, id_self))

            line_x = np.array([coords_self[0], coords_other[0]])
            line_y = np.array([coords_self[1], coords_other[1]])
            axs.plot(line_x, line_y, color=(0.75, 0.75, 0.75), zorder=0)

            axs.text(
                line_x[0] * 0.65 + line_x[1] * 0.35,
                line_y[0] * 0.65 + line_y[1] * 0.35,
                row_model["num_matches"],
                horizontalalignment="center",
                verticalalignment="top",
                rotation_mode="anchor",
            )
    axs.set_title(f"Number of matches found across tiles in {dataset_name}", wrap=True)
    fig.legend(loc=8, mode="expand", ncols=2, markerscale=0.5)
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    return fig
