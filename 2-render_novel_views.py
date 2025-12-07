# render_novel_views.py  —  beefed up parallax + hole handling

import os, glob, numpy as np, cv2, imageio.v2 as imageio
from tqdm import tqdm
import subprocess, shlex

FRAMES_DIR = "frames"
DEPTH_DIR  = "depth"
OUT_DIR    = "novel_frames"
FPS        = 24

# ==== knobs you can tweak easily ====
YAW_DEG = 12.0        # try 8–15; larger shows more rotation
TX = 0.05             # small lateral dolly (0.02–0.08 is common)
PARALLAX_GAIN = 0.7   # scale depth (0.5–0.9); smaller -> more parallax
SPLAT_RADIUS = 1      # 0 = single pixel; 1 or 2 fills holes better
INPAINT = True        # use OpenCV inpaint to clean remaining holes
USE_CONSTANT_VIEW = True  # True = same offset for whole clip
SS = 1.25             # supersample scale (1.0 = off). Try 1.25–1.5

# ==== camera + projection helpers ====
def intrinsics(w,h,f=None):
    if f is None: f = 1.1 * max(w,h)
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float32)

def yaw_rt(deg, tx=0.0, tz=0.0):
    a = np.deg2rad(deg)
    R = np.array([[ np.cos(a),0,np.sin(a)],
                  [ 0,        1,0        ],
                  [-np.sin(a),0,np.cos(a)]], dtype=np.float32)
    t = np.array([tx,0,tz], dtype=np.float32)
    return R, t

def to_point_cloud(rgb, depth, K):
    h,w = depth.shape
    j,i = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth.astype(np.float32)
    z[z<=0] = np.nan
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    pts  = np.stack([x,y,z], axis=-1).reshape(-1,3)
    cols = (rgb.reshape(-1,3)/255.0).astype(np.float32)
    m = np.isfinite(pts).all(axis=1)
    return pts[m], cols[m]

os.makedirs(OUT_DIR, exist_ok=True)
frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.*")))
depths = sorted(glob.glob(os.path.join(DEPTH_DIR , "*.*")))
assert frames, f"No frames in {FRAMES_DIR}"
assert len(frames) == len(depths), "frames/ and depth/ counts must match"

num = len(frames)
yaw_path = np.linspace(-YAW_DEG, YAW_DEG, num) if not USE_CONSTANT_VIEW else np.full(num, YAW_DEG)

for idx,(fpath,dpath) in enumerate(tqdm(list(zip(frames, depths)), total=num)):
    rgb_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
    d8      = cv2.imread(dpath, cv2.IMREAD_GRAYSCALE)
    assert rgb_bgr is not None and d8 is not None, f"Failed to read {fpath} / {dpath}"

    # match resolutions (MiDaS output can differ)
    h, w, _ = rgb_bgr.shape
    d8 = cv2.resize(d8, (w, h), interpolation=cv2.INTER_LINEAR)

    # optional supersample render size
    W = int(round(w*SS)); H = int(round(h*SS))
    rgb = cv2.cvtColor(cv2.resize(rgb_bgr, (W, H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    d8  = cv2.resize(d8, (W, H), interpolation=cv2.INTER_LINEAR)

    K = intrinsics(W, H)

    # map 0..255 MiDaS relative depth to pseudo-metric, then apply parallax gain
    dn = d8.astype(np.float32)/255.0
    depth = 1.0 / (0.1 + 0.9*dn)
    depth *= PARALLAX_GAIN

    pts, cols = to_point_cloud(rgb, depth, K)

    # camera move
    R, t = yaw_rt(yaw_path[idx], tx=TX, tz=0.0)
    Xp = (R @ pts.T).T + t

    Z = Xp[:,2]
    valid = Z > 0.05
    Xp, C, Z = Xp[valid], cols[valid], Z[valid]

    uv = Xp[:,:2] / Z[:,None]
    u = (uv[:,0]*K[0,0] + K[0,2]).round().astype(int)
    v = (uv[:,1]*K[1,1] + K[1,2]).round().astype(int)

    out  = np.zeros((H,W,3), dtype=np.uint8)
    zbuf = np.full((H,W), 1e9, dtype=np.float32)

    # splat with z-buffer
    r = int(SPLAT_RADIUS)
    if r <= 0:
        for uu,vv,zz,c in zip(u,v,Z,C):
            if 0<=uu<W and 0<=vv<H and zz < zbuf[vv,uu]:
                zbuf[vv,uu] = zz
                out[vv,uu] = (c*255).astype(np.uint8)
    else:
        rr = r*r
        for uu,vv,zz,c in zip(u,v,Z,C):
            if zz <= 0: continue
            x0 = max(0, uu - r); x1 = min(W-1, uu + r)
            y0 = max(0, vv - r); y1 = min(H-1, vv + r)
            for x in range(x0, x1+1):
                dx = x - uu
                dx2 = dx*dx
                for y in range(y0, y1+1):
                    dy = y - vv
                    if dx2 + dy*dy > rr: continue
                    if zz < zbuf[y, x]:
                        zbuf[y, x] = zz
                        out[y, x] = (c*255).astype(np.uint8)

    # quick cleanups
    out = cv2.medianBlur(out, 3)
    if INPAINT:
        hole_mask = (zbuf==1e9).astype(np.uint8)*255
        out = cv2.inpaint(out, hole_mask, 3, cv2.INPAINT_TELEA)

    # downscale if supersampled
    if SS != 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(OUT_DIR, f"nv_{idx:05d}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

# --- encode to mp4 via your local ffmpeg (avoids imageio tiff fallback) ---
cmd = "~/ffmpeg/ffmpeg -y -r {} -f image2 -i {}/nv_%05d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart novel_view.mp4".format(FPS, OUT_DIR)
subprocess.check_call(shlex.split(cmd))
print("Wrote novel_view.mp4")
