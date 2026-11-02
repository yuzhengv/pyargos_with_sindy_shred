"""Recolour the SST reconstruction map of the Fig. 8 schematic to the field colour
map of Fig. 9 (Yuzheng, 2026-11-02: "if we choose to use another colormap as in
Fig. 9, this one should align").

The diagrams.net export schematic-plots/pyargos_with_sindy_shred.pdf embeds the
map "SST Reconstruction (t = 100)" as a raster (6016 x 3022 px, RGB with a binary
land mask) drawn in matplotlib's coolwarm, and its vertical colour bar as a
second raster (151 x 3021 px). Both are palette images of 256 colours. This
script reads the two images from the page, inverts the colour map (each colour
to the nearest entry of the 256-step coolwarm table; the colour bar matches the
table exactly, the map's colours are a palette quantisation of it), applies
house_style.FIELD_CMAP (viridis, Fig. 9c) at the same 256 levels, and writes the
two images as PNG files that compose_fig8_schematic.sh lays exactly over the
originals. The images are stored upside down on the page (placed with a negative
vertical scale), so they are flipped here and placed upright. Nothing else in
the drawing changes; the values of the field are untouched (the map's title,
the colour-bar ticks "Normalized SST" 0 to 1 and the land mask stay).

Run:  python recolour_fig8_map.py <schematic.pdf> [--out-dir figures/house]
      (writes fig8_map_viridis.png and fig8_cbar_viridis.png; seconds)
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
from PIL import Image
from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plot-style"))
import house_style as hs  # noqa: E402

SOURCE_CMAP = "coolwarm"        # the colour map of the drawing's raster
MAP_SIZE = (6016, 3022)         # the map image (width, height) as embedded
BAR_SIZE = (151, 3021)          # the colour-bar image


def table(name):
    return (matplotlib.colormaps[name](np.arange(256) / 255.0)[:, :3] * 255).round().astype(np.uint8)


def recolour(rgb):
    """rgb (h, w, 3) uint8 in SOURCE_CMAP -> (h, w, 3) uint8 in hs.FIELD_CMAP; also the
    distance (RGB units) of each pixel's colour to its nearest source-table entry."""
    src = table(SOURCE_CMAP).astype(float)
    dst = table(hs.FIELD_CMAP)
    flat = rgb.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    uniq, inv = np.unique(packed, return_inverse=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(float)
    d2 = ((cols[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    dist = np.sqrt(d2.min(axis=1))
    return dst[idx][inv].reshape(rgb.shape), dist[inv].reshape(rgb.shape[:2])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("schematic", help="schematic-plots/pyargos_with_sindy_shred.pdf of the manuscript repository")
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "house"))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    page = PdfReader(a.schematic).pages[0]
    images = {im.image.size: np.array(im.image) for im in page.images}
    if MAP_SIZE not in images or BAR_SIZE not in images:
        sys.exit(f"expected a {MAP_SIZE} map and a {BAR_SIZE} colour bar on the page; found {list(images)}")
    m = images[MAP_SIZE]
    if m.shape[2] != 4:
        sys.exit("the map has no land mask (expected RGBA)")
    rgb, alpha = m[..., :3], m[..., 3]
    ocean = alpha == 255
    new, dist = recolour(rgb)
    out = np.dstack([new, alpha])
    out[~ocean, :3] = 0                                    # masked pixels: as in the source
    Image.fromarray(np.flipud(out), "RGBA").save(os.path.join(a.out_dir, "fig8_map_viridis.png"), optimize=True)
    bar = images[BAR_SIZE][..., :3]
    bar_new, bar_dist = recolour(bar)
    Image.fromarray(np.flipud(bar_new), "RGB").save(os.path.join(a.out_dir, "fig8_cbar_viridis.png"), optimize=True)
    print(f"map: {ocean.mean():.1%} ocean pixels; distance to the {SOURCE_CMAP} table: "
          f"median {np.median(dist[ocean]):.1f}, max {dist[ocean].max():.1f} (RGB units); "
          f"colour bar: max {bar_dist.max():.1f}")
    print("wrote", os.path.join(a.out_dir, "fig8_map_viridis.png"), "and fig8_cbar_viridis.png", f"({hs.FIELD_CMAP})")


if __name__ == "__main__":
    main()
