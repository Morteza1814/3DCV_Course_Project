# render_novel_views.py — stabilized + multi-frame splat + bg mask + z-dolly

import os, glob, numpy as np, cv2
from tqdm import tqdm
import subprocess, shlex
import math

FRAMES_DIR = "Frames/babywalk1"
DEPTH_DIR  = "Depth_any/babywalk1"
OUT_DIR    = "NovelView_any_2/novel_frames_babywalk1"
FPS        = 24

# ==== quality & motion knobs ====
SS = 1.5                  # supersample scale (1.0 off). 1.25–1.5 recommended
SPLAT_RADIUS = 2          # 0: single pixel; 1–2: better hole fill
INPAINT = True            # OpenCV inpaint for leftover holes
PARALLAX_GAIN = 0.6       # 0.5–0.7: smaller => more parallax
FOCAL_SCALE = 0.95        # <1 widens FOV slightly (more 3D feel)
USE_CONSTANT_VIEW = True  # if False, yaw sweeps across clip

# Camera motion
YAW_DEG = 8.0             # 6–12 is good
TX = 0.04                 # small lateral dolly
USE_TZ_SIN = True         # sinusoidal push-pull along Z
TZ_AMPL = 0.06            # amplitude for Z motion; try 0.05–0.08

# Temporal stability
USE_MULTI_FRAME = True    # splat neighbors t-1 and t+1
TEMP_EMA_ALPHA = 0.8      # depth_t = alpha*prev + (1-alpha)*raw; 0.75–0.9
BG_Z = 12.0               # constant background Z (meters-ish) for white bg

# ===== helpers =====
def intrinsics(w, h, f=None, fscale=1.0):
    if f is None: f = 1.1 * max(w, h)
    f = f * fscale
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float32)

def yaw_rt(deg, tx=0.0, tz=0.0):
    a = np.deg2rad(deg)
    R = np.array([[ np.cos(a),0,np.sin(a)],
                  [ 0,        1,0        ],
                  [-np.sin(a),0,np.cos(a)]], dtype=np.float32)
    t = np.array([tx,0,tz], dtype=np.float32)
    return R, t

def to_point_cloud(rgb, depth, K):
    h, w = depth.shape
    j, i = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth.astype(np.float32)
    z[z<=0] = np.nan
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    pts  = np.stack([x,y,z], axis=-1).reshape(-1,3)
    cols = (rgb.reshape(-1,3)/255.0).astype(np.float32)
    m = np.isfinite(pts).all(axis=1)
    return pts[m], cols[m]

def read_image_rgb(path, W, H):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: return None
    if (bgr.shape[1], bgr.shape[0]) != (W, H):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def read_depth_any(path, W, H):
    """Robust depth loader:
       - If 16-bit PNG: scale to 0..1.
       - If 8-bit grayscale: use as-is 0..255 then map later.
       - If 3-channel (colored depth): convert to gray.
    """
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None: return None, False
    # Resize to render size
    if d.ndim == 2:
        if (d.shape[1], d.shape[0]) != (W, H):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        if d.dtype == np.uint16:
            d = d.astype(np.float32)
            # Avoid divide by zero if image is constant
            m = float(d.max()) if d.max() > 0 else 1.0
            d = d / m  # 0..1 normalized
            return d, True  # normalized float already
        else:
            # 8-bit grayscale
            d = d.astype(np.float32) / 255.0
            return d, True
    else:
        # 3-channel colored depth: make grayscale proxy (still not ideal)
        if (d.shape[1], d.shape[0]) != (W, H):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        if d.shape[2] == 3:
            gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return gray, True
        # Fallback
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return d, True

def map_depth_relative_to_z(dn, gain=PARALLAX_GAIN):
    """Map relative depth in [0,1] to pseudo-Z (meters-ish)."""
    # keep within [eps,1]
    dn = np.clip(dn, 1e-5, 1.0)
    z = 1.0 / (0.1 + 0.9*dn)
    z *= gain
    return z

def white_bg_mask(rgb, thr=245):
    """Return background mask for near-white cyclorama."""
    # in RGB; consider pixel background if all channels high
    m = (rgb[:,:,0] >= thr) & (rgb[:,:,1] >= thr) & (rgb[:,:,2] >= thr)
    mask = m.astype(np.uint8) * 255
    # clean up mask a bit
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask  # 255 = background, 0 = foreground

def splat_points_into(out, zbuf, pts_cam, cols, K, radius):
    H, W, _ = out.shape
    Z = pts_cam[:,2]
    valid = Z > 0.05
    Xp = pts_cam[valid]; C = cols[valid]; Z = Z[valid]
    uv = Xp[:,:2] / Z[:,None]
    u = (uv[:,0]*K[0,0] + K[0,2]).round().astype(int)
    v = (uv[:,1]*K[1,1] + K[1,2]).round().astype(int)

    r = int(radius)
    if r <= 0:
        for uu, vv, zz, c in zip(u, v, Z, C):
            if 0 <= uu < W and 0 <= vv < H and zz < zbuf[vv, uu]:
                zbuf[vv, uu] = zz
                out[vv, uu] = (c * 255).astype(np.uint8)
    else:
        rr = r * r
        for uu, vv, zz, c in zip(u, v, Z, C):
            if zz <= 0: continue
            x0 = max(0, uu - r); x1 = min(W - 1, uu + r)
            y0 = max(0, vv - r); y1 = min(H - 1, vv + r)
            for x in range(x0, x1 + 1):
                dx = x - uu; dx2 = dx * dx
                for y in range(y0, y1 + 1):
                    dy = y - vv
                    if dx2 + dy*dy > rr: continue
                    if zz < zbuf[y, x]:
                        zbuf[y, x] = zz
                        out[y, x] = (c * 255).astype(np.uint8)

# ===== main =====
os.makedirs(OUT_DIR, exist_ok=True)
frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.*")))
depths = sorted(glob.glob(os.path.join(DEPTH_DIR , "*.*")))
assert frames, f"No frames in {FRAMES_DIR}"
assert len(frames) == len(depths), "frames/ and depth/ counts must match"

num = len(frames)
yaw_path = np.linspace(-YAW_DEG, YAW_DEG, num) if not USE_CONSTANT_VIEW else np.full(num, YAW_DEG)

# Render size and intrinsics from first frame
probe = cv2.imread(frames[0], cv2.IMREAD_COLOR)
assert probe is not None, f"Failed to read {frames[0]}"
h0, w0 = probe.shape[:2]
W = int(round(w0 * SS)); H = int(round(h0 * SS))
K = intrinsics(W, H, f=None, fscale=FOCAL_SCALE)

prev_depth_ema = None  # for temporal smoothing (current frame only)

for idx in tqdm(range(num), total=num):
    # Camera motion for this output frame
    yaw = float(yaw_path[idx])
    tz = TZ_AMPL * math.sin(2 * math.pi * idx / max(1, num - 1)) if USE_TZ_SIN else 0.0
    R, t = yaw_rt(yaw, tx=TX, tz=tz)

    # Output buffers
    out  = np.zeros((H, W, 3), dtype=np.uint8)
    zbuf = np.full((H, W), 1e9, dtype=np.float32)

    # Neighbor window for multi-frame splat
    rels = [-1, 0, 1] if USE_MULTI_FRAME else [0]

    # For temporal median (optional small boost) gather raw depths for center
    raw_center_depth = None
    raw_prev_depth   = None
    raw_next_depth   = None

    for rel in rels:
        j = min(max(idx + rel, 0), num - 1)

        # Load inputs at render size
        rgb = read_image_rgb(frames[j], W, H)
        dn, ok = read_depth_any(depths[j], W, H)
        assert rgb is not None and ok, f"Failed to read {frames[j]} / {depths[j]}"

        # Background mask from white cyclorama
        bg_mask = white_bg_mask(rgb, thr=245)  # 255 = background
        fg_mask = (bg_mask == 0)

        # Map relative depth -> pseudo Z
        z = map_depth_relative_to_z(dn, gain=PARALLAX_GAIN)

        # Set background to constant far Z to avoid huge holes
        z = np.where(bg_mask == 255, BG_Z, z)

        # Store raw depths for center if we want a tiny temporal median
        if rel == 0:
            raw_center_depth = z.copy()
        elif rel == -1:
            raw_prev_depth = z.copy()
        elif rel == 1:
            raw_next_depth = z.copy()

        # For neighbors: we can use as-is; for center we’ll EMA/median after loop
        if rel != 0:
            pts, cols = to_point_cloud(rgb, z, K)
            # Transform to target camera and splat
            Xp = (R @ pts.T).T + t
            splat_points_into(out, zbuf, Xp, cols, K, SPLAT_RADIUS)

    # Apply temporal smoothing to the center frame depth
    if raw_center_depth is None:
        continue
    depth_center = raw_center_depth

    # Optional tiny temporal median using prev/next raw depths if available
    stack = [depth_center]
    if raw_prev_depth is not None: stack.append(raw_prev_depth)
    if raw_next_depth is not None: stack.append(raw_next_depth)
    if len(stack) > 1:
        depth_center = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    # EMA with previous frame’s smoothed depth (foreground only, keep background constant)
    if prev_depth_ema is None:
        depth_smooth = depth_center
    else:
        # only smooth foreground; keep background at BG_Z
        fg = (depth_center < BG_Z - 1e-6).astype(np.float32)
        depth_smooth = TEMP_EMA_ALPHA * prev_depth_ema * fg + (1.0 - TEMP_EMA_ALPHA) * depth_center * fg
        depth_smooth += (1.0 - fg) * BG_Z
    prev_depth_ema = depth_smooth.copy()

    # Now splat the smoothed center frame last (so it "wins" on color for current time)
    rgb_center = read_image_rgb(frames[idx], W, H)
    pts_c, cols_c = to_point_cloud(rgb_center, depth_smooth, K)
    Xp_c = (R @ pts_c.T).T + t
    splat_points_into(out, zbuf, Xp_c, cols_c, K, SPLAT_RADIUS)

    # quick cleanups
    out = cv2.medianBlur(out, 3)
    if INPAINT:
        hole_mask = (zbuf == 1e9).astype(np.uint8) * 255
        out = cv2.inpaint(out, hole_mask, 3, cv2.INPAINT_TELEA)

    # downscale if supersampled
    if SS != 1.0:
        out_small = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)
    else:
        out_small = out

    cv2.imwrite(os.path.join(OUT_DIR, f"nv_{idx:05d}.png"),
                cv2.cvtColor(out_small, cv2.COLOR_RGB2BGR))

# --- encode to mp4 via ffmpeg ---
cmd = "~/ffmpeg/ffmpeg -y -r {} -f image2 -i {}/nv_%05d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart novel_view.mp4".format(FPS, OUT_DIR)
subprocess.check_call(shlex.split(cmd))
print("Wrote novel_view.mp4")
