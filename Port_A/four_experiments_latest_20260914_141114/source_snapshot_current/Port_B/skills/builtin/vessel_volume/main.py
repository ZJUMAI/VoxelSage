"""vessel_volume: 计算血管体积。

从 mask_dir 中读取血管掩码，调用 Tool_Box.liver_analysis.compute_vessel_volume
计算体积（mm³/cm³）。
"""

from Tool_Box.liver_analysis import compute_vessel_volume
from Tool_Box.mask_resolution import resolve_mask_path
from skills.utils import convert_numpy


def run(ctx):
    vessel_names = ctx.params.get("vessel_names", ["hepatic", "portal"])

    results = {}
    for name in vessel_names:
        try:
            resolved = resolve_mask_path(ctx.mask_dir, name)
            ctx.log(f"Computing volume for {name}...")
            vol = compute_vessel_volume(resolved.path)
            results[name] = {**vol, "mask_variant": resolved.variant}
        except FileNotFoundError:
            ctx.log(f"Vessel mask '{name}' not found, skipping")
            results[name] = {"error": f"mask '{name}' not found"}

    return convert_numpy({"vessels": results})
