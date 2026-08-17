import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

def generate_prediction_visual(
    ndvi_array: np.ndarray,
    grass_mask_2d: np.ndarray,
    grass_percentage: float,
    confidence: float,
    lat: float,
    lon: float,
    tree_mask_2d: np.ndarray = None,
    tree_percentage: float = 0.0,
    non_veg_percentage: float = 0.0,
    prediction: str = "Unknown"
) -> str:
    """
    Generates a color-coded PNG with NDVI map + 3-class ML prediction mask.
    Green = Grass, Dark Green = Tree, Brown = Non-Vegetation
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # --- Left: NDVI heatmap ---
    ax1 = axes[0]
    ndvi_plot = ax1.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(ndvi_plot, ax=ax1, label='NDVI Value')
    ax1.set_title('NDVI Map', color='white', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Pixel (X)', color='white')
    ax1.set_ylabel('Pixel (Y)', color='white')
    ax1.tick_params(colors='white')

    # --- Right: 3-class prediction map ---
    ax2 = axes[1]
    color_map = np.zeros((*ndvi_array.shape, 3))

    # Default: everything is Non-Vegetation (brown)
    color_map[:] = [0.6, 0.4, 0.2]

    # Grass pixels — light green
    color_map[grass_mask_2d] = [0.2, 0.8, 0.2]

    # Tree pixels — dark green (only for CNN)
    if tree_mask_2d is not None:
        color_map[tree_mask_2d] = [0.0, 0.4, 0.1]

    ax2.imshow(color_map)
    ax2.set_title('Vegetation Detection Map (ML)', color='white', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Pixel (X)', color='white')
    ax2.set_ylabel('Pixel (Y)', color='white')
    ax2.tick_params(colors='white')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#33cc33', label=f'Grass ({grass_percentage:.1f}%)'),
        mpatches.Patch(color='#006622', label=f'Tree ({tree_percentage:.1f}%)'),
        mpatches.Patch(color='#996633', label=f'Non-Veg ({non_veg_percentage:.1f}%)'),
    ]
    ax2.legend(handles=legend_patches, loc='lower right',
               facecolor='#1a1a2e', labelcolor='white', fontsize=9)

    # Title
    title_color = '#33cc33' if 'Grass' in prediction else ('#006622' if 'Tree' in prediction else '#ff4444')
    fig.suptitle(
        f'Prediction: {prediction}  |  Confidence: {confidence}%\n'
        f'Coordinates: ({lat}, {lon})',
        color=title_color, fontsize=14, fontweight='bold', y=1.02
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"prediction_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    logger.info(f"Visualization saved: {output_path}")
    return output_path