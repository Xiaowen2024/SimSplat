# Neural rendering for an excavator simulator: benchmark, failure modes, recommendation

![](hero_joint_push.gif)

*A nuScenes work zone (scene-0061) reconstructed with 3D Gaussian splatting. Every clearly visible barrier has been
rebuilt as a simulator asset and rendered back into the neural background. Left: t = 0. Right: a MuJoCo excavator
pushes the joint between two barriers; both move and the rest of the row stays put.*

## TL;DR

- **Fidelity.** On 7 nuScenes scenes the scene-graph model beats vanilla 3DGS by **+2.68 dB PSNR**: 21.35 → 24.03 dB
  mean, better on all 7 scenes.
  - At the same 30k-step budget the gain is +2.87 dB (3 scenes). Rigid actors account for 1.65 dB, the sky model
    for 0.77 and per-camera colour for 0.42; the actors' share ranges from 0.3 to 3.2 dB depending on traffic.
- **Integration.** Reconstructed objects that must be *interactive* are rebuilt as conventional sim assets. Shape
  comes from SAM photo masks and texture from the fleet photos. The assets become MuJoCo rigid bodies and are
  rendered back into the neural background.
  - 21 of the 65 barriers in the work zone are converted. The other 44 are occluded hard cases and are left in
    the background.
- **Does it transfer to autonomy?** Across 370 (view, object) pairs, the detector's outcome on the simulator's
  render matches its outcome on the real photo **90.5%** of the time. The unedited reconstruction matches 92.4%.
  - A camera-driven closed loop (detect → localize → push 1.2 m) ends 4.8 cm from the goal; the oracle-state
    loop ends 5.4 cm away.
  - This takes three fixes for failures that appear only in closed loop.
- **Geometry.** Photos alone give good images over wrong depth. Adding LiDAR depth supervision reduces depth error
  about 7× (median AbsRel 0.14 → 0.02) for −0.4 dB PSNR. It roughly halves the failed asset fits and cuts
  localization error from 0.45 to 0.29 m.
- **Terrain.** On an off-road site (RELLIS-3D) the excavator digs a trench in a height-field soil model rendered into
  the reconstruction, then pushes a barrier. The cut soil was never photographed, so its look has to be authored.
- **Recommendation.**
  - Use neural rendering for everything the autonomy stack only *looks at*: background, terrain appearance, parked
    equipment.
  - Use photo-textured conventional assets for everything it *touches*.
  - Measure the gap with a downstream detector, not with PSNR alone.

## Pipeline

```mermaid
flowchart LR
    A["Fleet data<br/>6 cameras + LiDAR"] --> B["3DGS scene graph<br/>static field + rigid actors<br/>+ sky + per-camera colour"]
    B --> C{"Will the robot<br/>touch it?"}
    C -- no --> D["Stays neural"]
    C -- yes --> E["Look-alike asset<br/>SAM photo-mask fit<br/>+ photo texture"]
    E --> F["Carve + inpaint<br/>the neural background"]
    E --> G["MuJoCo rigid body<br/>(convex colliders)"]
    F --> H["gsplat render:<br/>neural background<br/>+ assets at sim poses"]
    G --> H
    H --> I["Perception eval<br/>real vs recon vs sim"]
    H --> J["Closed loop<br/>detect → localize → push"]
```

| Stage | Script |
| --- | --- |
| Reconstruct | `train.py` (scene graph: static field + per-track rigid actors, sky map, per-camera affine colour) |
| Rebuild objects as assets | `build_lookalike_asset.py --category`, checked by `check_assets.py` |
| Remove them from the background | `inpaint_removed.py --asset-root` |
| Simulate | `simulate_excavator.py` (MuJoCo 3, excavator arm with IK + position servos) |
| Render the simulation | `render_physics.py` |
| Measure | `eval.py`, `eval_scene_perception.py`, `closed_loop_push.py` |

![](assets_contact_sheet.jpg)

*The 21 rebuilt barriers. Panels per asset: photo | reconstruction | asset alone | asset in place. Shape is fitted
from SAM masks on the fleet photos; texture is projected from them.*

## 1. Benchmark

### 1.1 Visual fidelity (held-out views, 7 scenes)

| Scene | PSNR: vanilla 3DGS → ours | SSIM | LPIPS ↓ |
| --- | --- | --- | --- |
| 0061 (work zone) | 20.87 → **23.59** | 0.698 → **0.743** | 0.485 → **0.388** |
| 0103 | 20.12 → **22.91** | 0.714 → **0.750** | 0.502 → **0.397** |
| 0553 | 27.12 → **29.83** | 0.937 → **0.946** | **0.084** → 0.088 |
| 0655 | 20.59 → **23.01** | 0.750 → **0.780** | 0.475 → **0.378** |
| 0757 | 22.53 → **26.68** | 0.771 → **0.822** | 0.354 → **0.254** |
| 0796 | 18.26 → **20.54** | 0.550 → **0.587** | 0.748 → **0.638** |
| 0916 | 19.99 → **21.64** | 0.675 → **0.717** | 0.516 → **0.401** |
| **Mean** | 21.35 → **24.03** | 0.728 → **0.763** | 0.452 → **0.363** |

**Equal-budget ablation.** The table above compares a 50k-step model with 30k-step vanilla 3DGS, and four changes
at once. Here every model gets the vanilla budget: 30k steps, densification until 15k. One component is removed at
a time. PSNR on held-out views; *static* means pixels outside annotated objects.

| Scene | vanilla 3DGS | full (30k) | − rigid actors | − sky | − per-camera colour | full (50k) |
| --- | --- | --- | --- | --- | --- | --- |
| 0061 | 20.87 | **23.15** | 22.85 (−0.30) | 22.54 (−0.62) | 21.87 (−1.28) | 23.59 |
| 0553 | 27.12 | 29.65 | 28.21 (−1.45) | 28.54 (−1.12) | **30.17 (+0.51)** | 29.83 |
| 0757 | 22.53 | **26.34** | 23.14 (−3.20) | 25.76 (−0.59) | 25.84 (−0.50) | 26.68 |
| **Mean** | 23.51 | **26.38** | 24.73 (−1.65) | 25.61 (−0.77) | 25.96 (−0.42) | 26.70 |

- **At equal budget the gain is +2.87 dB**, so it is not an artefact of longer training (the extra 20k steps add
  0.32 dB).
- **The three components roughly add up** (1.65 + 0.77 + 0.42 = 2.84 dB).
- **Rigid actors matter most, but per scene.** They are worth 3.2 dB in 0757 (heavy traffic) and only 0.3 dB in the
  parked work zone 0061. Most of that gain is in *static* pixels (0757 static PSNR 27.67 vs 24.38): without actors,
  moving cars smear floaters across the road that every view sees.
- **Per-camera colour can hurt.** In 0553 (overcast, cameras consistent) it costs 0.5 dB. The affine transform
  absorbs some view-dependent appearance that should stay in the Gaussians. In 0061 (low sun) it is worth 1.3 dB.

### 1.2 Temporal consistency and speed

A simulator replays sequences, so a render must not flicker. `eval_temporal.py` takes 60 consecutive front-camera
frames and checks each pair:
1. lift every pixel of frame t+1 with the rendered depth;
2. move it into frame t with the known poses;
3. compare colours. The score is the mean absolute RGB residual (0–255) on unoccluded pixels.

The same warp is applied to the real photos as a reference. Their residual includes exposure changes, moving traffic
and warp error. A renderer is consistent when its residual is well below the photos' residual.

**Setup:** RTX 3080 Ti, idle GPU, 800 × 450 (nuScenes) and 960 × 600 (RELLIS), median frame time.

| Scene | Renderer | Warp residual ↓ | FPS | Gaussians |
| --- | --- | --- | --- | --- |
| 0061 | *real photos* | *6.51* | — | — |
| | vanilla 3DGS | 3.91 | 157 | 1.08 M |
| | ours, 30k | 3.43 | 93 | 1.16 M |
| | ours, 50k | 3.84 | 86 | 1.30 M |
| | ours, 50k + LiDAR depth | **3.07** | 83 | 1.31 M |
| 0553 | *real photos* | *2.96* | — | — |
| | vanilla 3DGS | 0.14† | 327 | 1.04 M |
| | ours, 30k / 50k | 0.94 / 1.05 | 176 / 165 | 0.66 / 0.69 M |
| 0757 (traffic) | *real photos* | *5.25* | — | — |
| | vanilla 3DGS | 5.17 | 108 | 1.60 M |
| | ours, 30k / 50k | **3.05** / 3.14 | 69 / 62 | 1.69 / 1.83 M |
| RELLIS (off-road) | *real photos* | *9.54* | — | — |
| | ours, no depth loss | 1.35 | 253 | 0.15 M |
| | ours + LiDAR depth | **1.06** | 218 | 0.16 M |

- **No renderer flickers more than reality.** All residuals are at or below the real photos' residual.
- **Vanilla 3DGS flickers in traffic.** In 0757 its residual (5.17) is as large as the photos'. Moving cars have no
  actor, so they are baked as view-dependent smears that change from frame to frame. Rigid actors cut it to 3.05.
- **Depth supervision makes frames more consistent** (0061: 3.84 → 3.07; RELLIS: 1.35 → 1.06). The colours now sit
  on the right surface, so reprojecting them holds.
- **Speed.** Everything is real-time at camera resolution. The scene graph costs 1.5–2× vanilla's frame time: per-actor
  transforms, the sky pass and more Gaussians.

† Not yet explained. The metric uses each renderer's own depth, so a renderer whose depth is wrong in a smooth way can
score well. The depth-error metrics (§2) are the check for that.

### 1.3 Does fidelity transfer to perception?

The scenes are scored with an off-the-shelf open-vocabulary detector:
- **Detector:** Grounding DINO tiny, run on the full frame plus tiles.
- **Ground truth:** the visible boxes of the 21 rebuilt barriers.
- **Images compared:**
  - *real*: the fleet photo;
  - *reconstruction*: the unedited 3DGS render;
  - *sim*: the inpainted background plus all assets, as the simulator renders it.

| 348 views, 370 (view, object) pairs | real | reconstruction | sim |
| --- | --- | --- | --- |
| Recall @IoU 0.3 | 27.6% | 21.1% | 25.4% |
| Recall @IoU 0.5 | 14.1% | 8.1% | 11.1% |
| Same outcome as on the real photo | — | 92.4% | **90.5%** |
| Sim gap: detected on real, missed in render | — | 6.8% | **6.2%** |
| Row of barriers boxed as one object | 84% | 92% | 93% |
| Depth-localization error, median | 0.47 m* | 0.45 m | 0.31 m |

\* The photo has no depth, so it uses the reconstruction's depth as a LiDAR stand-in.

![](perception_row_merge.jpg)

*The main sim gap, panels photo | reconstruction | sim (green: ground truth, red: detections). On the photo the
detector boxes the orange barrier alone (IoU 0.81). On both renders it merges it with the neighbour, because the
dark seam between abutting barriers is softer.*

**Does better fidelity mean better transfer?** Same 370 pairs, rendered by every model of the ablation (unedited
reconstructions, no assets):

| Renderer (0061) | crop PSNR | crop LPIPS ↓ | Same outcome as real | Sim gap | Recall @0.3 | Rows merged | Depth error, median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla 3DGS (30k) | 23.12 | 0.307 | 90.0% | 8.4% | 21.1% | 92% | 0.38 m |
| − rigid actors (30k) | 23.38 | 0.305 | 91.4% | 7.3% | 22.2% | 89% | 0.53 m |
| − sky (30k) | 24.60 | 0.214 | 93.5% | 5.7% | 23.2% | 92% | 0.39 m |
| − per-camera colour (30k) | 24.70 | 0.227 | 93.5% | 5.9% | 21.9% | 91% | 0.48 m |
| full (30k) | 24.80 | 0.223 | 93.2% | 6.2% | 20.8% | 92% | 0.51 m |
| full (50k) | **25.36** | **0.196** | 92.4% | 6.8% | 21.1% | 92% | 0.45 m |
| *real photo* | — | — | — | — | 27.6% | 84% | 0.47 m* |

- **Fidelity helps, but saturates.** Going from vanilla to any model with LPIPS ≤ 0.23 cuts the sim gap from 8.4% to
  about 6%. Beyond that, a further 0.6 dB and 0.03 LPIPS buy nothing. One pair is 0.27 points, so the differences
  among the top four rows are noise.
- **The remaining gap is structural, not photometric.** Every renderer merges rows of barriers more often than the
  photo (89–92% vs 84%), and recall @0.3 is stuck at 21–23% vs 27.6%. That is the missing seam shadow of §1.3, which
  more training does not recover.
- **Depth error does not follow PSNR.** Localization from rendered depth is best for vanilla 3DGS (0.38 m), whose
  images are the worst. A likely reason (not verified) is that its depth is smoother, so the median over a box is
  less contaminated by background. This is why rendered depth needs its own supervision (§2, LiDAR depth).
- Speed is the cost: 224 FPS (vanilla) vs 91–120 FPS with actors and sky (shared GPU; see §1.2).

### 1.4 Closed loop

- **Task:** push the barrier 1.2 m and leave it upright.
- **Each cycle:**
  1. render the simulator view;
  2. detect the barrier;
  3. localize it from depth;
  4. command the bucket.

**Setup:**
- **Scene:** the inpainted background, all 21 barrier assets, and the simulated neighbours.
- **Observation camera:** one fleet camera (view 195).
- **Detector:** Grounding DINO tiny.

Each perception or control fix was added one at a time, so every row isolates the failure it addresses.

| Controller input | Result | Final error | What happened |
| --- | --- | --- | --- |
| Oracle state (simulator truth) | ✅ 2 pushes | 5.4 cm | — |
| Camera, as first written | ❌ toppled | — | Depth sees the near face: the estimate is 0.24–0.47 m ahead, so the bucket comes down *on* the barrier |
| + surface → centre correction | ❌ lost track | — | Detector boxes the whole row as one: no detection near the last estimate |
| + 0.5 m clearance before lowering | ⚠ ends in tolerance, never converges | 5.3 cm | Estimate stays 0.21 m ahead, oscillates until max cycles |
| + predicted-box gating | ❌ toppled | — | Tracks the right unit; the 0.23 m bias remains, and the 0.15 m gap is too small |
| gating + clearance | ❌ stops short | 22 cm | Bias makes it think the goal is reached |
| **gating + clearance + centre** | ✅ **2 pushes** | **4.8 cm** | Perception error 3.4 cm, same as the oracle |

How each fix works:
- **Predicted-box gating:** project the last estimate with the known object size, pick the detection that covers
  that box, and clip a merged row-box to it.
- **Clearance:** lower the bucket farther behind the object than the perception error.
- **Surface → centre:** move the depth point by half the object's depth along the ray.

No single fix is enough. The two perception failures, a biased depth point and merged boxes, only show up in
closed loop, and the controller has to be robust to the error that remains.

| Camera as first written: barrier toppled | All three fixes: tracked unit, 3 cm error |
| --- | --- |
| ![](closed_loop_toppled.jpg) | ![](closed_loop_tracked.jpg) |

## 2. Failure modes found, and fixes

Each of these was found by a measurement, not by eye alone.

| Failure | Found by | Fix | Effect |
| --- | --- | --- | --- |
| Static objects are baked into the background: the barrier's own actor Gaussians are transparent (12 of 6011 opaque) | inspection | Rebuild as an asset instead of moving Gaussians | — |
| Rendered depth around off-road objects is a blur along viewing rays | `check_assets` flags + side-view point plots | Fit shape from SAM photo masks across views | Sanity checks passed: 5 of 65 → 21 built, 0 flagged. The first barrier had been 25% too short |
| nuScenes per-frame poses drift up to ±0.3 m in height | spread of mask edges between views | Per-frame height correction, used by every renderer | — |
| Blotchy textures: background and pedestrians in the samples | contact sheet | Sample inside the photo mask and reject what disagrees with the static render | Photo-textured surface 17% → 34% |
| A vehicle parked in front gets baked into the texture | contact sheet | Skip objects whose views disagree on height (IQR > 15 cm) | 44 hard cases left in the background |
| The asset shows through a half-transparent veil | strict crop PSNR + a green-asset test | Inpainting: only Gaussians *behind* the asset learn the fill | Hidden pixels 27–48% → 4–11% |
| Neighbours start sunk in or floating (±21 cm) and pop at t = 0 | idle-control simulation | Snap simulated bodies to the ground; render only their simulated *motion* on the photo-fitted pose; shrink annotation colliders; 1 s settle | Idle drift 6–9 cm → ~1e-10 m |
| Closed loop topples the barrier, loses it, or stops short | closed loop vs oracle (§1.4) | Predicted-box gating + 0.5 m clearance + surface → centre | Fail → 4.8 cm, oracle 5.4 cm |
| Rendered depth is wrong although the images are good (photometric loss only) | LiDAR depth error; depth-only asset fits; localization error (§1.3) | LiDAR depth supervision (§2.1) | Depth error 0.14 → 0.02 (median AbsRel); depth-fit flags 40 → 22 of 64 |
| Wrong prompt: the whole row (and the excavator) boxed as "road barrier" | closed loop | Prompt "an orange plastic barrier." | — |
| Rendered panel captioned "original scene" | demo GIF | Variable shadowing in `render_physics` | — |

### 2.1 Fix in depth: LiDAR depth supervision

The reconstruction is trained on photos only, so nothing makes its depth correct. Its images can be good while its
geometry is fog along the viewing rays. Three earlier measurements traced back to that:
- the depth-only asset fit flagged 40 of 64 barriers;
- localization from rendered depth was off by 0.45 m;
- the ablation renderers' depth error did not follow PSNR (§1.3).

**Fix:**
- project the LiDAR sweep nearest in time into each image (`add_lidar_depth.py`);
- drop returns hidden behind nearer ones (the LiDAR sits higher than the camera, so it sees past occluders);
- add a relative L1 loss on rendered depth at static pixels, weight 0.1.

Scene 0061, 50k steps, same model otherwise:

| | without | with LiDAR depth |
| --- | --- | --- |
| Depth AbsRel vs held-out LiDAR, mean / median ↓ | 0.216 / 0.141 | **0.068 / 0.020** |
| Depth within 25% of LiDAR (δ < 1.25) | 68% | **92%** |
| PSNR / LPIPS | **23.59 / 0.388** | 23.15 / 0.398 |
| Depth-only asset fit: barriers flagged | 40 of 64 | **22 of 64** |
| Fitted height / annotated height, median | 0.54 | **0.81** (the photo-mask fit gives ≈ 0.85) |
| Localization error from rendered depth, median | 0.45 m | **0.29 m** |
| Detector outcome same as on the photo | 92.4% | **94.1%** |
| Temporal warp residual | 3.84 | **3.07** |

On RELLIS-3D (off-road, §2.2) the same loss takes depth AbsRel from 0.161 to **0.045** and δ < 1.25 from 71% to 97%,
at −0.26 dB PSNR.

![](lidar_depth_0061.jpg)

*Scene 0061, held-out view. Top: photo and renders without / with the depth loss; they are almost the same. Bottom:
LiDAR, then rendered depth without / with the loss (turbo colour map, 0–40 m). With photos only, the road has
blotches nearer or farther than the surface: the "fog" that broke the asset fits. With LiDAR the road is a smooth
gradient and the kerb line of barriers is a sharp depth edge.*

![](lidar_depth_rellis.jpg)

*RELLIS-3D trail. With photos only, the grass ground is patchy and its depth flickers between nearby values. With
LiDAR it matches the scan. The RGB barely changes (the ruts are slightly crisper); the blur is the separate
underfitting problem of §2.2.*

**Trade-off:** about 0.3–0.4 dB PSNR for depth that is roughly 3–7× more accurate. For a simulator that is the right
trade. Every downstream geometric use improves: asset fitting, localization, temporal consistency and the terrain
surface. The photometric loss alone cannot see the error.

### 2.2 Construction scene: deformable terrain on an off-road site (RELLIS-3D)

nuScenes is paved road. To test digging on real soil, the same pipeline runs on **RELLIS-3D** sequence 00001: an
off-road trail of dirt ruts and grass, one camera, an Ouster LiDAR and SLAM poses.
- `convert_rellis.py` builds the scene: 300 frames, undistorted, with LiDAR depth.
- RELLIS has no annotations, so an authored barrier marks the work zone.

The site was picked from data: flat, LiDAR-covered ground beside the trail, seen by about 50 views at 7–20 m.

**Soil model** (`terrain.py`). The bucket's motion comes from MuJoCo; the soil is kinematic.
- **Height field:** 5 cm cells, initialized from LiDAR.
- **Cut:** soil above the bucket bottom goes into the bucket, up to its capacity.
- **Dump:** the load is spilled as a mound.
- **Settle:** slopes relax to the angle of repose.
- **Rendering:** fixed-size surfel slots, so the soil updates in place without flicker. The reconstructed ground
  Gaussians over changed cells are hidden.

**Scripted job** (`simulate_dig.py`): four passes dig a trench, then the bucket pushes the barrier out of the way. The
barrier is a MuJoCo rigid body.

| Trench depth | Trench area | Excavated | Volume balance | Barrier pushed | Barrier tilt |
| --- | --- | --- | --- | --- | --- |
| 0.63 m | 2.78 m² | 0.66 m³ | ~1e-16 m³ | 0.60 m | 3.6° (upright) |

![](rellis_dig_push.gif)

*Left: the reconstruction along the captured trail. Right: the same camera path with the simulated excavator, the
soil height field (trench and spoil pile) and the pushed barrier.*

- **Views along the captured trail render well.** The road path in the figure is interpolated between captured views.
- **Novel viewpoints off the trail do not.** A camera 5 m to the side of the trail, or a cab camera looking down,
  sees ground the reconstruction only ever observed at grazing angles. It renders as grey fog. A machine-mounted
  camera would need captures from the machine's own viewpoints, or a ground prior.

**Findings:**
- **Newly exposed soil has no photos.** The cut surface was never photographed, so its appearance has to be authored
  (here: a measured soil colour, shading and depth darkening). This is a hard limit of reconstruction for any
  machine that changes the scene.
  - The fix that scales is a *material library*: soil textures captured once per site type and applied by the soil
    model.
- **The ground must have the right depth** for the hidden-ground swap to work: only Gaussians near the true ground
  are removed. Without LiDAR depth the ground Gaussians are spread along the viewing rays, so the swap would be expected to leave some of them over the trench or cut holes elsewhere (not measured separately).
- **Off-road needs denser Gaussians.** Grass and ruts are high-frequency. With the road-tuned densification threshold
  the model had 155k Gaussians and LPIPS 0.60.
  - A 2.5× lower threshold gives 275k Gaussians and LPIPS 0.584.
  - Per-image pose refinement, a learned SE(3) correction per view (`pose_refine`), makes it *worse* (LPIPS 0.667).
  - Training views are as blurry as held-out ones (train PSNR 28.2, eval 28.1). So the model underfits views that
    disagree with each other, and the cause is not poses. The likely causes are wind-blown grass and motion blur.
    The real photos' own warp residual (9.5) is 2–3× nuScenes'.
  - Handling this needs a per-image appearance/transient model, or masking vegetation from the loss.

### Construction-specific challenges: status

| Challenge | Status |
| --- | --- |
| Dynamic objects | Done: per-track rigid actors; moving actors removed or replaced |
| Harsh lighting | *Partly.* Lighting is baked into the Gaussians and the assets, so pushed assets keep their original shading and cast no new shadow. The missing contact shadow between abutting barriers is the main remaining perception gap (row merging 93% vs 84%) |
| Deformable terrain | *Done, kinematic* (§2.2). Height-field soil (cut, dump, angle of repose, volume conserved) driven by the MuJoCo bucket. It is rendered into the reconstruction on an off-road RELLIS-3D site. Soil reaction forces on the bucket are not modelled |
| Dust | Not yet. Plan: participating-medium Gaussians or screen-space dust, evaluated by detector degradation against real dusty frames |

## 3. Recommendation: where neural rendering should (and should not) replace sim assets

1. **Background and anything only observed: neural.** It is the only practical way to get real-site appearance, and
   it is cheap to update from fleet logs. Detector outcomes on reconstructions match the real photos 92% of the time.
2. **Anything the machine touches: conventional assets textured from the same fleet photos.** Gaussians baked into
   the background cannot move. Static objects are never separated from it, so they must be rebuilt.
   - Simulation needs collision geometry and mass, which the reconstruction does not provide.
   - Mixing the two costs little: 90.5% vs 92.4% agreement.
3. **Occluded or ambiguous objects: leave them neural.** The pipeline should detect when it cannot build a
   faithful asset (here, disagreement between views) rather than guess.
4. **Physics and rendering need different truths.** Physics needs a consistent world, and rendering needs alignment
   with each photo. Keep them separate and apply simulated *motion* to photo-aligned poses.
5. **Measure with the downstream task.** PSNR ranks renderers, but the gap that matters showed up only in detector
   behaviour: rows merged into one box, and depth that is too clean.

## 4. Limitations

- **The detector is a weak proxy.** Grounding DINO tiny is open-vocabulary, has low recall even on real photos, and
  merges rows of barriers. A driving-trained 3D detector would be the real test.
- **Lighting is baked.** There are no relighting, cast shadows or time-of-day changes.
- **The excavator is untextured boxes.** Its appearance is not part of the evaluation.
- **No real excavation site.** The scenes are a nuScenes road work zone and an off-road trail (RELLIS-3D). Neither has real
  machines or freshly dug soil to compare against.
- **Soil is kinematic.** The bucket does not feel the soil, and the cut surface's appearance is authored.

## Reproduce

```bash
S=configs/scenes/nuscenes_scene0061_sweeps.yaml; M=configs/methods/nuscenes_scene0061_sweeps.yaml
python -m sim_splat.scripts.train --scene $S --config $M
python -m sim_splat.scripts.build_lookalike_asset --scene $S --config $M --checkpoint <ckpt> --category movable_object.barrier --surfels 40000 --output-dir <assets>
python -m sim_splat.scripts.check_assets <assets>
python -m sim_splat.scripts.inpaint_removed --scene $S --config $M --checkpoint <ckpt> --asset-root <assets> --output-dir <bg>
python -m sim_splat.scripts.eval_scene_perception --scene $S --config $M --checkpoint <ckpt> --background-checkpoint <bg>/final.pt --asset-root <assets> --max-views 400 --output-dir <out>
python -m sim_splat.scripts.simulate_excavator --scene $S --track 808a988e --asset-dir <assets>/808a988e --asset-root <assets> --push-sign -1 --lateral-offset 1.0 --asset-mass 60 --travel 1.6 --seconds 6 --output push.pt
python -m sim_splat.scripts.render_physics --scene $S --config $M --checkpoint <bg>/final.pt --physics push.pt --asset-root <assets> --views 79 85 109 --skip-depth --output-dir <render>
```
