"""Side-by-side panels: photo | RGB without / with LiDAR depth loss | LiDAR | depth without / with.

Usage: compare_depth.py SCENE BASE_CONFIG BASE_CKPT DEPTH_CONFIG DEPTH_CKPT OUT_DIR VIEW,VIEW,...
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.cm as cm
from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.dataparsers import parse_scene
from sim_splat.data.input_dataset import InputDataset
from sim_splat.engine.checkpoints import load_model_weights
from sim_splat.models.sim_splat_model import SimSplatModel

scene_cfg, base_cfg, base_ck, depth_cfg, depth_ck, out, views = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], Path(sys.argv[6]), [int(v) for v in sys.argv[7].split(",")]
scene = parse_scene(load_scene_config(Path(scene_cfg))); ds = InputDataset(scene)
renders = {}
for tag, cf, ck in (("base", base_cfg, base_ck), ("depth", depth_cfg, depth_ck)):
    cfg = load_train_config(Path(cf)); m = SimSplatModel(cfg.model, scene, cfg.device)
    load_model_weights(Path(ck), m, cfg.device); m.eval(); m.set_step(10**9)
    with torch.no_grad():
        for v in views:
            o = m.get_outputs(scene.cameras[v]); renders[(tag, v)] = (o["rgb"].clamp(0, 1).cpu().numpy(), o["depth"][..., 0].cpu().numpy())
    del m; torch.cuda.empty_cache()

def colour(d, vmax):
    return (cm.turbo(np.clip(d / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)

out.mkdir(parents=True, exist_ok=True)
rows = []
for v in views:
    b = ds[v]; photo = (b.image.numpy() * 255).astype(np.uint8)
    H, W = photo.shape[:2]; vmax = 40.0
    lidar = np.zeros((H, W, 3), np.uint8) + 30
    if b.lidar is not None:
        uv, z = b.lidar[:, :2].numpy().astype(int), b.lidar[:, 2].numpy()
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        c = colour(z[ok], vmax)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                lidar[np.clip(uv[ok, 1] + dy, 0, H - 1), np.clip(uv[ok, 0] + dx, 0, W - 1)] = c
    panels = [("photo", photo), ("RGB, photo loss only", (renders[("base", v)][0] * 255).astype(np.uint8)),
              ("RGB, + LiDAR depth loss", (renders[("depth", v)][0] * 255).astype(np.uint8)),
              ("LiDAR (ground truth)", lidar), ("depth, photo loss only", colour(renders[("base", v)][1], vmax)),
              ("depth, + LiDAR depth loss", colour(renders[("depth", v)][1], vmax))]
    tiles = []
    for name, img in panels:
        t = Image.fromarray(img).resize((W // 2, H // 2)); d = ImageDraw.Draw(t)
        d.rectangle((0, 0, 8 + 6 * len(name), 16), fill="black"); d.text((4, 3), name, fill="white"); tiles.append(t)
    w, h = tiles[0].size
    grid = Image.new("RGB", (3 * w + 8, 2 * h + 4), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 3) * (w + 4), (i // 3) * (h + 4)))
    grid.save(out / f"view_{v:04d}.jpg", quality=88); rows.append(grid)
print(json.dumps({"views": views, "out": str(out)}))
