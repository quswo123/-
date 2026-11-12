"""Sleeping cloud — baby-like breathing loop (v5: natural breath rhythm + breath-linked float + shading).

Usage:
    python breathe.py                       # renders renders/out.mp4 (same look as v5)
    python breathe.py --v3                  # previous v3 rhythm/float for comparison
    python breathe.py --no-shading          # v2 look
    python breathe.py --morph 14 --shade 0.6 --light -45 -o renders/test.mp4
    python breathe.py --alpha               # also writes transparent PNG sequence (for Premiere/AE)
    python breathe.py --webm                # also writes a transparent WebM (VP9 alpha, for web/overlays)

Every motion is a whole-number harmonic of the loop length, so the clip loops seamlessly.
"""
import argparse, math, os, shutil, subprocess, tempfile
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("-i", "--input", default="assets/sleeping_cloud.png")
p.add_argument("-o", "--output", default="renders/out.mp4")
p.add_argument("--fps", type=int, default=30)
p.add_argument("--breaths", type=int, default=2, help="breaths per loop (loop = breaths * period)")
p.add_argument("--period", type=float, default=2.5, help="seconds per breath")
p.add_argument("--breath-y", type=float, default=0.040, help="vertical swell of the deepest breath (fraction)")
p.add_argument("--breath-x", type=float, default=0.016, help="horizontal swell of the deepest breath (fraction)")
p.add_argument("--inhale", type=float, default=0.38, help="inhale share of each breath")
p.add_argument("--rest", type=float, default=0.06, help="still pause at the bottom of each breath (share)")
p.add_argument("--breath-var", type=float, default=0.15, help="every other breath is this much shallower (0 = identical)")
p.add_argument("--breath-swing", type=float, default=0.06, help="deep breath is this much longer, shallow one shorter (0 = even)")
p.add_argument("--morph", type=float, default=9.0, help="outline morph amount (px)")
p.add_argument("--bob", type=float, default=3.5, help="slow float up/down over the whole loop (px)")
p.add_argument("--bob-breath", type=float, default=5.0, help="extra lift that rides each breath, a beat late (px)")
p.add_argument("--tilt", type=float, default=0.5, help="slow rocking (deg)")
p.add_argument("--tilt-breath", type=float, default=0.25, help="head side lifts on each inhale (deg)")
p.add_argument("--blush", type=float, default=0.35, help="how much the cheeks warm up on inhale")
p.add_argument("--v3", action="store_true", help="use the v3 rhythm/float/size (even breaths, independent float)")
p.add_argument("--tail", type=float, default=4.0, help="tail sway (deg)")
p.add_argument("--shade", type=float, default=0.42, help="shadow strength (0 = off)")
p.add_argument("--sheen", type=float, default=0.38, help="highlight strength")
p.add_argument("--light", type=float, default=-135.0, help="light direction deg (-135 = upper-left, -45 = upper-right)")
p.add_argument("--no-shading", action="store_true")
p.add_argument("--no-shadow", action="store_true", help="remove ground contact shadow")
p.add_argument("--size", type=int, default=1080)
p.add_argument("--alpha", action="store_true", help="also export transparent PNG frames to renders/alpha_frames/")
p.add_argument("--webm", action="store_true", help="also export a transparent VP9 WebM next to the output (<output>.webm)")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
if a.v3:
    a.breath_y, a.breath_x, a.inhale, a.rest, a.breath_var, a.breath_swing = 0.042, 0.018, 0.4, 0.1, 0.0, 0.0
    a.bob, a.bob_breath, a.tilt, a.tilt_breath = 7.0, 0.0, 0.7, 0.0

src = cv2.imread(a.input, cv2.IMREAD_UNCHANGED).astype(np.float32)
assert src.shape[2] == 4, "input must be a transparent PNG (RGBA)"
H, W = src.shape[:2]

# Landmarks measured on the 1254x1254 artwork; scaled if the input differs.
sc = W / 1254.0
top, base, cx = 357.0 * sc, 905.0 * sc, 616.0 * sc
FPS = a.fps
DUR = a.breaths * a.period
w1 = 2 * math.pi / DUR
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
h = np.clip((base - yy) / (base - top), 0, 1)


def soft_disc(cx_, cy_, rx, ry, feather):
    d = np.sqrt(((xx - cx_ * sc) / (rx * sc)) ** 2 + ((yy - cy_ * sc) / (ry * sc)) ** 2)
    return np.clip((1.0 - d) / feather + 0.5, 0, 1)


face = soft_disc(810, 640, 190, 110, 0.5)     # keep eyes/mouth crisp
tail = soft_disc(140, 640, 150, 130, 0.6)     # curl on the left
pillow = soft_disc(970, 790, 230, 130, 0.5)   # small cloud under the cheek
cheeks = (soft_disc(679, 713, 62, 50, 0.6) + soft_disc(975, 620, 76, 44, 0.6))[..., None]
BLUSH = np.array([-18, -8, 6], np.float32)       # BGR nudge toward a warmer pink

# ---- outline morph: smooth, loopable noise, strongest near the edge, weak on the face
alpha_m = (src[..., 3] > 20).astype(np.uint8)
dist = cv2.distanceTransform(alpha_m, cv2.DIST_L2, 5)
edge_w = np.where(alpha_m > 0, 1 - np.clip(dist / (140 * sc), 0, 1) * 0.75, 1.0).astype(np.float32)
morph_w = edge_w * (1 - 0.9 * face)
rng = np.random.default_rng(a.seed)


def field(sigma=170):
    n = rng.standard_normal((H // 8, W // 8)).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), sigma * sc)
    return n / (np.abs(n).max() + 1e-6)


F = [field() for _ in range(8)]

# ---- volume shading: soft height map from the fill (NOT distanceTransform -> radial ridges)
bgr, A = src[..., :3], src[..., 3]
gray = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
facezone = soft_disc(810, 650, 230, 140, 0.4) > 0.5
line = (((gray < 175) & (A > 120)) & ~facezone).astype(np.uint8)  # outlines split the lobes
fill = ((A > 200) & (line == 0)).astype(np.float32)
hgt = 0.55 * cv2.GaussianBlur(fill, (0, 0), 35 * sc) + 0.45 * cv2.GaussianBlur(fill, (0, 0), 85 * sc)
hgt = cv2.GaussianBlur(hgt * fill, (0, 0), 4 * sc)
gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=5) / 32.0
gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=5) / 32.0
k = 140.0 * sc
nx, ny, nz = -gx * k, -gy * k, np.ones_like(gx)
nn = np.sqrt(nx * nx + ny * ny + nz * nz)
nx, ny, nz = nx / nn, ny / nn, nz / nn
fillw = cv2.GaussianBlur((A > 200).astype(np.float32) * (1 - line), (0, 0), 1.2)
SHADOW_COL = np.array([196, 160, 182], np.float32)  # BGR lavender, matches the artwork
SHEEN_COL = np.array([228, 246, 255], np.float32)   # BGR warm cream


def shade_src(t, fb):
    out = src.copy()
    out[..., :3] = np.clip(out[..., :3] + (a.blush * fb) * cheeks * BLUSH, 0, 255)  # blush deepens on inhale
    if a.no_shading:
        return out
    th = math.radians(a.light + 18 * math.sin(w1 * t))   # light drifts slowly
    el = math.radians(38 + 6 * math.sin(w1 * t + 1.2))
    lx, ly, lz = math.cos(el) * math.cos(th), math.cos(el) * math.sin(th), math.sin(el)
    lam = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    flat = math.sin(el)
    dark = np.clip((flat - lam) / flat, 0, 1)
    lite = np.clip((lam - flat) / (1 - flat), 0, 1) ** 1.6
    S = a.shade * dark ** 1.2 * fillw
    L = (a.sheen + 0.14 * fb) * lite * fillw              # highlight brightens on inhale
    c = out[..., :3]
    c = c + (SHADOW_COL - c) * S[..., None]
    c = c + (SHEEN_COL - c) * L[..., None]
    out[..., :3] = np.clip(c, 0, 255)
    return out


# Breath schedule: alternate a deeper, slightly longer breath with a shallower, shorter one,
# so the two breaths in the loop are not carbon copies. Durations sum to DUR -> still loops.
_d = [a.period * (1 + a.breath_swing * (1 if k % 2 == 0 else -1)) for k in range(a.breaths)]
DURS = [d * DUR / sum(_d) for d in _d]
DEPTH = [1.0 - a.breath_var * (k % 2) for k in range(a.breaths)]
STARTS = np.cumsum([0.0] + DURS[:-1]).tolist()


def smoothstep(u):
    return u * u * (3 - 2 * u)


def breath(ph):
    # gentle inhale -> relaxed exhale that lets go quickly then drifts down slowly -> short still pause
    ri, re_ = a.inhale, 1.0 - a.rest
    if ph < ri:
        return 0.5 - 0.5 * math.cos(math.pi * ph / ri)
    if ph < re_:
        return 1.0 - smoothstep(((ph - ri) / (re_ - ri)) ** 0.75)
    return 0.0


def B(t, lag=0.0):
    tt = (t - lag) % DUR
    k = max(i for i in range(len(STARTS)) if STARTS[i] <= tt + 1e-9)
    return DEPTH[k] * breath((tt - STARTS[k]) / DURS[k])


pivot_tail = (235.0 * sc, 720.0 * sc)
tmp = tempfile.mkdtemp()
alpha_dir = os.path.join(os.path.dirname(a.output) or ".", "alpha_frames")
if a.alpha:
    os.makedirs(alpha_dir, exist_ok=True)
rgba_dir = alpha_dir if a.alpha else os.path.join(tmp, "rgba")
if a.webm:
    os.makedirs(rgba_dir, exist_ok=True)

n_frames = int(round(FPS * DUR))
for i in range(n_frames):
    t = i / FPS
    # 1) float + tilt: a slow drift over the loop, plus a little lift and head-side rise riding each breath
    fb = B(t)
    up = a.bob * math.sin(w1 * t) + a.bob_breath * B(t, 0.22)        # px, positive = up
    tilt = math.radians(a.tilt) * math.sin(w1 * t + 0.9) - math.radians(a.tilt_breath) * B(t, 0.1)
    X, Y = xx.copy(), yy + up
    c, s = math.cos(-tilt), math.sin(-tilt)
    dx, dy = X - cx, Y - base
    X, Y = cx + c * dx - s * dy, base + s * dx + c * dy
    # 2) breathing (bottom anchored); pillow follows a beat later and softer
    fp = B(t, 0.18)
    fl = fb * (1 - pillow) + 0.6 * fp * pillow
    Y = base - (base - Y) / (1 + a.breath_y * fl)
    X = cx + (X - cx) / (1 + a.breath_x * fl * (0.3 + 0.7 * h))
    # 3) tail sway
    ang = math.radians(a.tail) * math.sin(w1 * 2 * t - 1.3) * tail
    c2, s2 = np.cos(-ang), np.sin(-ang)
    dx, dy = X - pivot_tail[0], Y - pivot_tail[1]
    X, Y = pivot_tail[0] + c2 * dx - s2 * dy, pivot_tail[1] + s2 * dx + c2 * dy
    # 4) outline morph
    a1, b1, a2, b2 = math.cos(w1 * t), math.sin(w1 * t), math.cos(2 * w1 * t), math.sin(2 * w1 * t)
    X = X - (F[0] * a1 + F[1] * b1 + 0.5 * (F[2] * a2 + F[3] * b2)) * a.morph * sc * morph_w
    Y = Y - (F[4] * a1 + F[5] * b1 + 0.5 * (F[6] * a2 + F[7] * b2)) * a.morph * sc * morph_w

    out = cv2.remap(shade_src(t, fb), X.astype(np.float32), Y.astype(np.float32), cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out = np.clip(out, 0, 255)

    # soft contact shadow that shrinks as the cloud floats up
    sh = 0.0
    if not a.no_shadow:
        span = 2 * a.bob + a.bob_breath
        lift = (up + a.bob) / span if span else 0.5
        sh = (soft_disc(625, 918, 440 * (1 - 0.08 * lift), 22, 1.6) * (0.10 - 0.035 * lift))[..., None]
    al = out[..., 3:4] / 255.0

    if a.alpha or a.webm:
        # transparent frame: cloud over a black shadow layer (alpha = sh) -> identical to the MP4 on white
        A_t = al + sh * (1 - al)
        rgb_t = np.where(A_t > 1e-4, out[..., :3] * al / np.maximum(A_t, 1e-4), 0)
        rgba = np.concatenate([rgb_t, A_t * 255], axis=2)
        cv2.imwrite(os.path.join(rgba_dir, f"f{i:03d}.png"), np.clip(rgba + 0.5, 0, 255).astype(np.uint8))

    # white background
    comp = out[..., :3] * al + 255 * (1 - sh) * (1 - al)
    cv2.imwrite(os.path.join(tmp, f"w{i:03d}.png"), np.clip(comp, 0, 255).astype(np.uint8))
    print(f"\rframe {i + 1}/{n_frames}", end="", flush=True)

print()
os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "w%03d.png"),
                "-vf", f"scale={a.size}:{a.size}:flags=lanczos", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.output], check=True)
print("saved", a.output)
if a.webm:
    webm = os.path.splitext(a.output)[0] + ".webm"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(rgba_dir, "f%03d.png"),
                    "-vf", f"scale={a.size}:{a.size}:flags=lanczos,format=yuva420p", "-c:v", "libvpx-vp9",
                    "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28", "-row-mt", "1", "-auto-alt-ref", "0", webm], check=True)
    print("saved", webm)
shutil.rmtree(tmp)
if a.alpha:
    print("saved", alpha_dir)
