import matplotlib.pyplot as plt
import polars as pl
import numpy as np

def plot_matches_grid(
    *, 
    images: dict[str, np.ndarray], 
    point_df: pl.DataFrame, 
    dataset_name: str, 
    invert_x, 
    invert_y
) -> plt.Figure:
    fig_w = 8.5
    fig_h = 11

    fig, axs_all = plt.subplots(
        figsize=(fig_w, fig_h),
        nrows=2,
        height_ratios=(1.5,1)
    )
    axs = axs_all[0]
    axs.spines[["right", "top"]].set_visible(False)

    image_ids = sorted(
        point_df['image_id_self'].unique().to_list(), 
        key=lambda v: int(v))
    conn_mat = np.zeros(
        (len(image_ids), ) * 2
         )

    axs.set_xlabel("Image x coordinate (nm)")
    axs.set_ylabel("Image y coordinate (nm)")
    cmap = plt.cm.viridis_r
    custom_lines = [
        plt.Line2D([0], [0], color=cmap(0.), lw=8),
        plt.Line2D([0], [0], color=cmap(.5), lw=8),
        plt.Line2D([0], [0], color=cmap(1.), lw=8)]
    
    point_df_sorted = point_df.sort(
        pl.col('image_id_self').cast(pl.Int64), 
        pl.col('image_id_other').cast(pl.Int64))
    
    grouped = point_df_sorted.group_by(
        "image_id_self", "image_id_other", maintain_order=True)
    observed_pairs = set()
    observed_points = set()

    for pair, subtable in grouped:
        id_self, id_other = pair
        if id_self not in observed_points:
            img = images[id_self]
            x_coord, y_coord = img.coords['x'], img.coords['y']
            coords_self = x_coord.mean(), y_coord.mean()
            num_matches = subtable['num_matches'][0]
            axs.add_patch(
                    plt.Circle(
                        coords_self, 
                        (x_coord[-1] - x_coord[0]) / 8 , 
                        facecolor='w', 
                        fill=True, 
                        lw=1, 
                        edgecolor='gray', 
                        zorder=2))
            axs.text(
                *coords_self,
                id_self, 
                verticalalignment='center', 
                horizontalalignment='center', 
                fontsize='x-large',
                zorder=3)            

            img.plot.imshow(
                ax=axs, 
                cmap="gray_r", 
                add_colorbar=False, 
                robust=True)

            axs.add_patch(plt.Rectangle(
                    (x_coord[0], y_coord[0]), 
                    x_coord[-1] - x_coord[0], 
                    y_coord[-1] - y_coord[0], 
                    color='y', fill=False, lw=1))
            observed_points.add(id_self)
        if pair not in observed_pairs:
            img_other = images[id_other]
            conn_mat[image_ids.index(id_other), image_ids.index(id_self)] = num_matches
            conn_mat[image_ids.index(id_self), image_ids.index(id_other)] = num_matches
            coords_other = img_other.coords['x'].mean(), img_other.coords['y'].mean()

            line_x = np.array([coords_self[0], coords_other[0]])
            line_y = np.array([coords_self[1], coords_other[1]])

            color = cmap(0.)
            
            if num_matches > 100:
                color = cmap(0.5)
            
            if num_matches > 1000:
                color = cmap(1.0)
            
            axs.plot(
                line_x, 
                line_y, 
                color=color,
                lw=8,
                zorder=1)
            observed_pairs.add(pair)
            observed_pairs.add(pair[::-1])

    axs.set_xlim(min((s.coords['x'][0] for s in images.values())), max((s.coords['x'][-1] for s in images.values())))
    axs.set_ylim(min((s.coords['y'][0] for s in images.values())), max((s.coords['y'][-1] for s in images.values())))
    axs.set_aspect('equal')
    axs.set_title(f"Number of matches found across tiles in {dataset_name}", wrap=True)

    if invert_y:
        axs.invert_yaxis()
    if invert_x:
        axs.invert_xaxis()
    axs.legend(custom_lines, ['0 - 99', '100 - 999', '1000+'])
    imshowed = axs_all[1].imshow(conn_mat, cmap='gray_r')
    for gap in range(len(image_ids)):
        axs_all[1].axvline(gap + 0.5, color=(0.3, 0.3, 0.3), lw=1)   
        axs_all[1].axhline(gap + 0.5, color=(0.3, 0.3, 0.3), lw=1)   
    axs_all[1].set_xlabel('Image ID')
    axs_all[1].set_ylabel('Image ID')
    axs_all[1].set_xticks(range(len(image_ids)), labels=image_ids)
    axs_all[1].set_yticks(range(len(image_ids)), labels=image_ids)
    axs_all[1].title.set_text(f"Pairwise match counts")
    cbar = plt.colorbar(imshowed, ax=axs_all[1], location='right')
    cbar.set_label("Number of matches")    
    return fig
