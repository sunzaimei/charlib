"""Face expressions on a painted texture.

1. `detect_landmarks` finds the eyes and mouth in the front-view face render.
2. Every expression is drawn in a fixed 1024px *reference* face space (tuned on the Red Lady character),
   then warped onto the real face with the affine transform defined by the three landmarks.
3. `project` bakes each painted face image back into the UV texture through the texel map.
No Blender needed here (numpy / Pillow / OpenCV only).
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REF = 1024
REF_LM = np.float32([[242, 700], [786, 695], [512, 925]])   # left eye, right eye, mouth in reference space
SS = 4
LASH = (28, 14, 14, 255); BROW = (38, 22, 20, 255); LIP = (196, 96, 90, 255); MOUTH_IN = (120, 36, 44, 255)
TONGUE = (222, 120, 120, 255); TEETH = (250, 246, 244, 255); BLUSH = (240, 130, 140, 255)
L_EYE = dict(c=(245, 700), x0=92, x1=392); R_EYE = dict(c=(785, 695), x0=636, x1=935)
BROW_ERASE = [('line', ([(140, 545), (230, 527), (395, 548)], 36)), ('line', ([(620, 545), (720, 518), (898, 530)], 34))]
MOUTH_ERASE = [('ell', ((512, 925), (85, 22)))]
EYES_ERASE = [('ell', ((240, 697), (168, 112), 4)), ('ell', ((786, 692), (168, 112), -4))]
MOUTH_REGION = (425, 880, 600, 1000)


# ------------------------------------------------------------------ landmarks
def _skin_mask(rgb):
    f = rgb.astype(np.float32)
    h, w = f.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    bright = (hsv[..., 2] > 150) & (hsv[..., 1] < 90)
    centre = bright[int(h * .55):int(h * .85), int(w * .35):int(w * .65)]
    if centre.sum() < 50:
        raise RuntimeError("could not find skin in the face render; set face_center_z/face_scale or landmarks in the config")
    skin = np.median(f[int(h * .55):int(h * .85), int(w * .35):int(w * .65)][centre], axis=0)
    return (np.abs(f - skin).max(-1) < 32), skin


def _eye_candidates(gray, x0f, x1f, thr=70):
    h, w = gray.shape
    y0, y1 = int(.40 * h), int(.92 * h); X0, X1 = int(x0f * w), int(x1f * w)
    d = (gray[y0:y1, X0:X1] < thr).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d)
    out = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if .12 * w < bw < .42 * w and .07 * h < bh < .30 * h and a > .004 * w * h:
            out.append((float(a), np.float32([x + X0 + bw / 2, y + y0 + bh / 2])))
    return out


def detect_landmarks(rgb, alpha):
    """Return (left_eye, right_eye, mouth) centres in face-image pixels.

    Eyes = big dark blobs (iris + lashes) left/right of centre, paired by symmetry around the head bone
    (image centre). If only one side is clean (hair or earrings touching the other), it is mirrored."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cx = w / 2
    best = None
    for thr in (60, 75, 90):
        Ls = _eye_candidates(gray, .03, .50, thr); Rs = _eye_candidates(gray, .50, .97, thr)
        for la, lc in Ls:
            for ra, rc in Rs:
                sym = abs((lc[0] + rc[0]) / 2 - cx) / w + abs(lc[1] - rc[1]) / h
                if sym < .06 and (best is None or sym < best[0]):
                    best = (sym, lc, rc)
        if best:
            break
        if best is None and (Ls or Rs):          # mirror the cleanest single eye
            if Ls:
                lc = max(Ls, key=lambda t: t[0])[1]; best = (1, lc, np.float32([2 * cx - lc[0], lc[1]]))
            else:
                rc = max(Rs, key=lambda t: t[0])[1]; best = (1, np.float32([2 * cx - rc[0], rc[1]]), rc)
            break
    if best is None:
        hull_dark = (gray[int(.4 * h):int(.92 * h), int(.2 * w):int(.8 * w)] < 90).mean()
        if hull_dark < 0.01:
            raise RuntimeError("the face looks blank (no painted eyes/mouth on the texture). Expressions need a painted face; "
                               "re-run with --allow-no-face to build actions only")
        raise RuntimeError("could not find the eyes; set left_eye/right_eye/mouth in the config (see previews/face_layout.png)")
    _, L, R = best
    ed = float(np.linalg.norm(R - L))
    guess = (L + R) / 2 + np.float32([0, 0.42 * ed])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = ((hsv[..., 0] < 12) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 60) & (hsv[..., 2] > 80)
    y0 = int(max(L[1], R[1]) + 0.25 * ed); y1 = int(min(h - 1, guess[1] + 0.2 * ed))
    x0 = int(guess[0] - 0.12 * ed); x1 = int(guess[0] + 0.12 * ed)
    sub = red[y0:y1, x0:x1]
    M = np.float32([guess[0], (np.nonzero(sub)[0].mean() + y0) if sub.sum() > 15 else guess[1]])
    return L, R, M


def affine(lm_target, res):
    """Reference(1024) -> target face image transform."""
    return cv2.getAffineTransform(REF_LM, np.float32(lm_target))


# ------------------------------------------------------------------ painting helpers (reference space)
def curve(p0, p1, p2, n=24):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in np.linspace(0, 1, n)]


class Layer:
    def __init__(s):
        s.im = Image.new('RGBA', (REF * SS, REF * SS), (0, 0, 0, 0)); s.d = ImageDraw.Draw(s.im)

    def P(s, pts): return [(x * SS, y * SS) for x, y in pts]

    def stroke(s, pts, w0, w1, col):
        n = len(pts)
        for i in range(n - 1):
            t = i / (n - 2) if n > 2 else 0.5; w = w0 + (w1 - w0) * math.sin(math.pi * t)
            s.d.line(s.P([pts[i], pts[i + 1]]), fill=col, width=max(1, int(w * SS)))
            r = w * SS / 2; x, y = pts[i + 1]; s.d.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=col)

    def poly(s, pts, col): s.d.polygon(s.P(pts), fill=col)

    def ell(s, c, r, col): s.d.ellipse([(c[0] - r[0]) * SS, (c[1] - r[1]) * SS, (c[0] + r[0]) * SS, (c[1] + r[1]) * SS], fill=col)

    def ref(s, blur=0):
        im = s.im.resize((REF, REF), Image.LANCZOS)
        return im.filter(ImageFilter.GaussianBlur(blur)) if blur else im


def brows(L, mode):
    if mode == 'sad': lb = [(160, 530), (260, 522), (372, 505)]; rb = [(640, 505), (760, 522), (885, 532)]
    elif mode == 'angry': lb = [(160, 515), (270, 528), (378, 560)]; rb = [(632, 560), (740, 528), (888, 512)]
    elif mode == 'up': lb = [(155, 515), (235, 490), (380, 512)]; rb = [(630, 510), (720, 485), (890, 500)]
    else: lb = [(158, 540), (240, 528), (378, 535)]; rb = [(635, 535), (730, 522), (890, 532)]
    for b in (lb, rb): L.stroke(curve(*b), 3, 11, BROW)


def closed_eye(L, e, happy=False, side='L'):
    x0, x1 = e['x0'] + 25, e['x1'] - 25; cy = e['c'][1] + (10 if not happy else 0)
    mid = ((x0 + x1) / 2, cy - 45 if happy else cy + 35)
    pts = curve((x0, cy), mid, (x1, cy), 30)
    L.stroke(pts, 5, 16 if happy else 13, LASH)
    if not happy:
        for t in ((0.25, 0.45, 0.65) if side == 'L' else (0.35, 0.55, 0.75)):
            x, y = pts[int(t * 29)]; L.stroke([(x, y), (x + (-14 if side == 'L' else 14), y + 18)], 2, 4, LASH)
        if side == 'L': L.stroke([(x0, cy), (x0 - 22, cy - 8)], 2, 6, LASH)
        else: L.stroke([(x1, cy), (x1 + 22, cy - 8)], 2, 6, LASH)


def mouth(L, kind):
    c = (512, 928)
    if kind == 'smile_open':
        top = curve((452, 915), (512, 928), (572, 915), 20); bot = curve((572, 915), (512, 985), (452, 915), 20)
        L.poly(top + bot, MOUTH_IN)
        L.poly(curve((470, 955), (512, 940), (554, 955), 12) + curve((554, 955), (512, 982), (470, 955), 12), TONGUE)
        L.poly(curve((458, 917), (512, 930), (566, 917), 16) + curve((566, 917), (512, 940), (458, 917), 16), TEETH)
        L.stroke(top, 2, 4, LIP)
    elif kind == 'smile': L.stroke(curve((455, 915), (512, 948), (570, 915)), 2, 6, LIP)
    elif kind == 'frown': L.stroke(curve((470, 940), (512, 915), (555, 940)), 2, 6, LIP)
    elif kind == 'angry':
        L.poly(curve((465, 935), (512, 915), (560, 935), 14) + curve((560, 935), (512, 950), (465, 935), 14), MOUTH_IN)
        L.poly(curve((470, 933), (512, 918), (555, 933), 14) + curve((555, 933), (512, 934), (470, 933), 14), TEETH)
        L.stroke(curve((465, 935), (512, 915), (560, 935)), 2, 4, LIP)
    elif kind == 'O':
        L.ell(c, (24, 30), LIP); L.ell(c, (19, 25), MOUTH_IN); L.ell((512, 945), (12, 8), TONGUE)
    elif kind == 'A':
        L.poly(curve((465, 915), (512, 905), (560, 915), 16) + curve((560, 915), (512, 985), (465, 915), 16), LIP)
        L.poly(curve((470, 917), (512, 909), (555, 917), 16) + curve((555, 917), (512, 978), (470, 917), 16), MOUTH_IN)
        L.ell((512, 960), (24, 13), TONGUE)
    elif kind == 'I':
        L.poly(curve((455, 922), (512, 912), (570, 922), 16) + curve((570, 922), (512, 950), (455, 922), 16), LIP)
        L.poly(curve((460, 923), (512, 915), (565, 923), 16) + curve((565, 923), (512, 946), (460, 923), 16), MOUTH_IN)
        L.poly(curve((462, 923), (512, 916), (563, 923), 16) + curve((563, 923), (512, 932), (462, 923), 16), TEETH)
    elif kind == 'U': L.ell(c, (15, 17), LIP); L.ell(c, (10, 12), MOUTH_IN)
    elif kind == 'E':
        L.poly(curve((452, 920), (512, 910), (572, 920), 16) + curve((572, 920), (512, 962), (452, 920), 16), LIP)
        L.poly(curve((457, 921), (512, 913), (567, 921), 16) + curve((567, 921), (512, 957), (457, 921), 16), MOUTH_IN)
        L.poly(curve((460, 921), (512, 914), (564, 921), 16) + curve((564, 921), (512, 930), (460, 921), 16), TEETH)
    elif kind == 'wavy':
        xs = np.linspace(478, 546, 20); L.stroke([(x, 928 + 5 * math.sin((x - 478) / 68 * 3 * math.pi)) for x in xs], 2, 5, LIP)
    elif kind == 'cry':
        L.poly(curve((462, 940), (512, 905), (562, 940), 16) + curve((562, 940), (512, 965), (462, 940), 16), MOUTH_IN)
        L.stroke(curve((462, 940), (512, 905), (562, 940)), 2, 4, LIP)


def blush(L, strong=False):
    a = 150 if strong else 75
    for c in ((215, 835), (810, 830)): L.ell(c, (95, 40), (*BLUSH[:3], a))


def blush_lines(L):
    for cx in (215, 810):
        for k in range(4):
            x = cx - 45 + k * 28; L.stroke([(x + 10, 812), (x - 6, 850)], 2, 4, (205, 80, 95, 230))


def tears(L):
    for x in (190, 300, 730, 840):
        for k in range(40):
            L.ell((x + 2 * math.sin(k / 6), 745 + k * 4), (5 + k * 0.18, 5), (150, 200, 245, int(200 * (1 - k / 48))))
        L.ell((x - 2, 760), (3, 8), (255, 255, 255, 200))


# ------------------------------------------------------------------ expression recipes
# each recipe: (erase shapes, [(layer_painter, blur)])
ANGRY_LIDS = [('poly', [(130, 590), (400, 630), (400, 702), (260, 655), (130, 642)]),
              ('poly', [(895, 585), (625, 630), (625, 697), (780, 650), (895, 637)])]


def recipe(name):
    if name == 'neutral': return [], []
    if name == 'happy':
        return EYES_ERASE + MOUTH_ERASE, [(lambda L: blush(L), 20),
                                          (lambda L: (closed_eye(L, L_EYE, True, 'L'), closed_eye(L, R_EYE, True, 'R'), mouth(L, 'smile_open')), 0)]
    if name == 'smile': return MOUTH_ERASE, [(lambda L: blush(L), 20), (lambda L: mouth(L, 'smile'), 0)]
    if name == 'sad': return BROW_ERASE + MOUTH_ERASE, [(lambda L: (brows(L, 'sad'), mouth(L, 'frown')), 0)]
    if name == 'angry':
        def p(L):
            brows(L, 'angry'); mouth(L, 'angry')
            L.stroke(curve((120, 640), (260, 650), (392, 700)), 4, 12, LASH); L.stroke(curve((905, 635), (770, 645), (634, 695)), 4, 12, LASH)
        return BROW_ERASE + MOUTH_ERASE + ANGRY_LIDS, [(p, 0)]
    if name == 'surprised': return BROW_ERASE + MOUTH_ERASE, [(lambda L: (brows(L, 'up'), mouth(L, 'O')), 0)]
    if name == 'shy': return BROW_ERASE + MOUTH_ERASE, [(lambda L: blush(L, True), 20), (lambda L: (brows(L, 'sad'), mouth(L, 'wavy'), blush_lines(L)), 0)]
    if name == 'crying':
        return EYES_ERASE + BROW_ERASE + MOUTH_ERASE, [(lambda L: (brows(L, 'sad'), closed_eye(L, L_EYE, False, 'L'), closed_eye(L, R_EYE, False, 'R'), mouth(L, 'cry')), 0), (tears, 0)]
    if name == 'eyes_closed': return EYES_ERASE, [(lambda L: (closed_eye(L, L_EYE, False, 'L'), closed_eye(L, R_EYE, False, 'R')), 0)]
    if name == 'wink': return [EYES_ERASE[1]] + MOUTH_ERASE, [(lambda L: (closed_eye(L, R_EYE, True, 'R'), mouth(L, 'smile')), 0)]
    if name.startswith('mouth_'): return MOUTH_ERASE, [(lambda L, k=name[6:]: mouth(L, k), 0)]
    raise KeyError(name)


# ------------------------------------------------------------------ painting onto the real face
class FacePainter:
    def __init__(self, rgb, alpha, landmarks):
        self.rgb = rgb; self.alpha = alpha; self.res = rgb.shape[0]
        self.A = affine(landmarks, self.res)
        f = rgb.astype(np.float32)
        skin, self.skin_col = _skin_mask(rgb)
        # only skin below the brows (exclude forehead hair)
        ytop = int(min(landmarks[0][1], landmarks[1][1]) - 0.45 * np.linalg.norm(np.subtract(landmarks[1], landmarks[0])))
        skin[:max(ytop, 0)] = False
        S = skin.astype(np.float32)
        def nc(sig): return cv2.GaussianBlur(f * S[..., None], (0, 0), sig) / (cv2.GaussianBlur(S, (0, 0), sig)[..., None] + 1e-6)
        sc = self.res / REF
        self.plate = np.nan_to_num(np.where((cv2.GaussianBlur(S, (0, 0), 12 * sc) > 0.05)[..., None], nc(12 * sc), nc(40 * sc)))

    def warp(self, img_ref, interp=cv2.INTER_LINEAR):
        return cv2.warpAffine(img_ref, self.A, (self.res, self.res), flags=interp, borderValue=0)

    def erase_mask(self, shapes):
        m = np.zeros((REF, REF), np.uint8)
        for kind, args in shapes:
            if kind == 'ell': cv2.ellipse(m, args[0], args[1], args[2] if len(args) > 2 else 0, 0, 360, 255, -1)
            if kind == 'poly': cv2.fillPoly(m, [np.array(args, np.int32)], 255)
            if kind == 'line': cv2.polylines(m, [np.array(args[0], np.int32)], False, 255, args[1])
        m = self.warp(m.astype(np.float32) / 255)
        m = cv2.GaussianBlur(m, (0, 0), 3 * self.res / REF)
        return np.clip(m * 1.6, 0, 1)[..., None]

    def paint(self, name):
        shapes, layers = recipe(name)
        img = self.rgb.astype(np.float32)
        if shapes:
            mf = self.erase_mask(shapes)
            img = img * (1 - mf) + self.plate * mf
        for painter, blur in layers:
            L = Layer(); painter(L)
            lay = np.asarray(L.ref(blur)).astype(np.float32) / 255
            lay = self.warp(lay)
            a = lay[..., 3:4]
            img = img * (1 - a) + lay[..., :3] * 255 * a
        return img.clip(0, 255).astype(np.uint8)

    def mouth_region_mask(self):
        m = np.zeros((REF, REF), np.float32)
        x0, y0, x1, y1 = MOUTH_REGION
        cv2.rectangle(m, (x0, y0), (x1, y1), 1.0, -1)
        return cv2.GaussianBlur(self.warp(m), (0, 0), 6 * self.res / REF)

    def layout_preview(self, landmarks, path):
        im = Image.fromarray(self.rgb).convert('RGB'); d = ImageDraw.Draw(im)
        for (x, y), lab in zip(landmarks, ('L eye', 'R eye', 'mouth')):
            d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(0, 255, 0), width=3); d.text((x + 10, y - 10), lab, fill=(0, 255, 0))
        ov = self.paint('happy')
        both = Image.new('RGB', (self.res * 2, self.res)); both.paste(im, (0, 0)); both.paste(Image.fromarray(ov), (self.res, 0))
        both.resize((self.res, self.res // 2)).save(path)


# ------------------------------------------------------------------ bake into the UV texture
def _sample(img, x, y):
    n = len(x); k = -(-n // 4096) * 4096
    X = np.zeros(k, np.float32); Y = np.zeros(k, np.float32); X[:n] = x - 0.5; Y[:n] = y - 0.5
    r = cv2.remap(img, X.reshape(-1, 4096), Y.reshape(-1, 4096), cv2.INTER_LINEAR)
    return r.reshape(k, -1)[:n]


def _bleed(out, ty, tx, w, vals=None, iters=6):
    """Spread edits into neighbouring unmapped texels (UV gutters / cracks)."""
    H, W = out.shape[:2]
    y0, y1 = max(ty.min() - 8, 0), min(ty.max() + 9, H); x0, x1 = max(tx.min() - 8, 0), min(tx.max() + 9, W)
    sub = out[y0:y1, x0:x1]
    F = np.zeros(sub.shape[:2], np.float32); F[ty - y0, tx - x0] = 1
    Wm = np.zeros_like(F); Wm[ty - y0, tx - x0] = w
    V = sub * F.reshape(F.shape + (1,) * (sub.ndim - 2)); filled = F.copy(); Wf = Wm.copy()
    k = np.ones((3, 3), np.float32)
    for _ in range(iters):
        num = cv2.filter2D(V, -1, k); den = cv2.filter2D(filled, -1, k); wn = cv2.filter2D(Wf * filled, -1, k)
        new = (filled == 0) & (den > 0)
        V[new] = num[new] / (den[new][:, None] if V.ndim == 3 else den[new]); Wf[new] = wn[new] / den[new]; filled[new] = 1
    bl = (F == 0) & (filled == 1) & (Wf > 0.01)
    wb = Wf[bl][:, None] if sub.ndim == 3 else Wf[bl]
    sub[bl] = sub[bl] * (1 - wb) + V[bl] * wb
    out[y0:y1, x0:x1] = sub
    return out


def project(texture_rgb, face_neutral, face_new, fmap):
    ty, tx, px, py = fmap['ty'], fmap['tx'], fmap['px'], fmap['py']
    E = face_new.astype(np.float32); F0 = face_neutral.astype(np.float32)
    d = (np.abs(E - F0).max(-1) > 3).astype(np.float32)
    d = cv2.GaussianBlur(cv2.dilate(d, np.ones((5, 5))), (0, 0), 1.2)
    w = _sample(d, px, py)[:, 0]; c = _sample(E, px, py)
    out = texture_rgb.astype(np.float32).copy()
    out[ty, tx] = out[ty, tx] * (1 - w[:, None]) + c * w[:, None]
    return _bleed(out, ty, tx, w).clip(0, 255).astype(np.uint8)


def project_mask(mask_face, fmap, tex_shape):
    ty, tx, px, py = fmap['ty'], fmap['tx'], fmap['px'], fmap['py']
    w = _sample(mask_face.astype(np.float32), px, py)[:, 0]
    M = np.zeros(tex_shape[:2], np.float32); M[ty, tx] = w
    return (_bleed(M, ty, tx, np.ones_like(w)) * 255).clip(0, 255).astype(np.uint8)
