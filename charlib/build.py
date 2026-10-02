"""charlib build: rigged GLB -> character library.

    python -m charlib.build --glb character.glb --out ./out/MyChar --name MyChar [--config character.json]

Steps: import & normalise -> face render + texel map -> landmarks -> expression textures + mouth mask
-> hand-keyed actions -> GLB export -> library .blend with the panel -> preview sheets -> report."""
import argparse
import json
import os
import sys
import time

import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import scene_io, face, animate, actions_def
from .config import Config

PKG = os.path.dirname(__file__)


def log(*a):
    print('[charlib]', *a, flush=True)


def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
              '/Library/Fonts/Arial.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _sheet(tiles, cols, label_h=40, bg=(245, 242, 238)):
    w, h = tiles[0][1].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * w, rows * (h + label_h)), bg); d = ImageDraw.Draw(sheet); f = _font(22)
    for i, (name, im) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * (h + label_h)
        sheet.paste(im, (x, y), im if im.mode == 'RGBA' else None)
        d.text((x + w // 2, y + h + 8), name, fill=(60, 30, 30), font=f, anchor='mt')
    return sheet


def connect_bsdf(mat):
    nt = mat.node_tree
    out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    nt.links.new(bsdf.outputs[0], out.inputs['Surface'])
    em = nt.nodes.get('charlib_flat')
    if em: nt.nodes.remove(em)


def build(glb, out, cfg: Config, work=None):
    t0 = time.time()
    out = os.path.abspath(out); work = os.path.abspath(work or os.path.join(out, '_work'))
    for d in (out, work, f'{out}/textures', f'{out}/previews', f'{out}/source'):
        os.makedirs(d, exist_ok=True)
    report = {'glb': os.path.abspath(glb), 'name': cfg.name, 'config': json.loads(cfg.to_json())}

    # 1. import + normalise ------------------------------------------------------------------
    scene_io.reset()
    arm, body = scene_io.import_glb(glb)
    bmap = scene_io.bone_map(arm)
    arm, body = scene_io.normalise(arm, body, cfg.name)
    mat, img = scene_io.face_material(body)
    tw, th = img.size
    report.update(bones=len(arm.data.bones), material=mat.name, texture=[tw, th],
                  source_actions=[a.name for a in bpy.data.actions])
    log(f'imported: rig={arm.name} body={body.name} verts={len(body.data.vertices)} texture={tw}x{th}')
    tex_png = f'{work}/texture.png'
    scene_io.save_texture(img, tex_png)

    # 2. face render + texel map -------------------------------------------------------------
    cam = scene_io.face_camera(arm, body, bmap, cfg)
    report['face_camera'] = cam
    scene_io.flat_scene(cfg.face_res)
    scene_io.flat_material_on(mat)
    scene_io.ortho_cam(cam['cx'], cam['cz'], cam['scale'])
    scene_io.render(f'{work}/face_front.png')
    n = scene_io.build_face_map(body, cam, cfg.face_res, (tw, th), f'{work}/map.npz')
    log(f'face map: {n} texels')
    if n < 1000:
        raise RuntimeError('face map is almost empty - is the character facing -Y? check face_center_z/face_scale')

    # 3. landmarks -----------------------------------------------------------------------------
    fimg = np.array(Image.open(f'{work}/face_front.png').convert('RGBA'))
    rgb, alpha = fimg[..., :3].copy(), fimg[..., 3]
    if cfg.left_eye and cfg.right_eye and cfg.mouth:
        lm = [np.float32(cfg.left_eye), np.float32(cfg.right_eye), np.float32(cfg.mouth)]; src = 'config'
    else:
        try:
            lm = list(face.detect_landmarks(rgb, alpha)); src = 'auto'
        except RuntimeError as e:
            if not cfg.allow_no_face:
                raise
            log('WARNING', e, '-> building without expressions'); lm = None; src = 'none'
        if cfg.left_eye: lm[0] = np.float32(cfg.left_eye)
        if cfg.right_eye: lm[1] = np.float32(cfg.right_eye)
        if cfg.mouth: lm[2] = np.float32(cfg.mouth)
    tex = np.array(Image.open(tex_png).convert('RGB'))
    Image.fromarray(tex).save(f'{out}/textures/face_neutral.jpg', quality=cfg.texture_quality)
    if lm is None:
        cfg.expressions = ['neutral']
        report['landmarks'] = None
    else:
        report['landmarks'] = {'source': src, 'left_eye': lm[0].tolist(), 'right_eye': lm[1].tolist(), 'mouth': lm[2].tolist()}
        log('landmarks', report['landmarks'])
        painter = face.FacePainter(rgb, alpha, lm)
        painter.layout_preview(lm, f'{out}/previews/face_layout.png')
        neutral_face = painter.paint('neutral')
        fmap = dict(np.load(f'{work}/map.npz'))

    # 4. expression textures + mouth mask ------------------------------------------------------
    for name in cfg.expressions:
        if name == 'neutral':
            continue
        new = painter.paint(name)
        Image.fromarray(new).save(f'{work}/face_{name}.png')
        Image.fromarray(face.project(tex, neutral_face, new, fmap)).save(f'{out}/textures/face_{name}.jpg', quality=cfg.texture_quality)
        log('expression', name)
    if lm is not None:
        mask = face.project_mask(painter.mouth_region_mask(), fmap, tex.shape)
    else:
        mask = np.zeros(tex.shape[:2], np.uint8)
    Image.fromarray(mask).save(f'{out}/textures/mouth_mask.png', optimize=True)

    # 5. actions --------------------------------------------------------------------------------
    connect_bsdf(mat)
    made = animate.build_actions(arm, bmap, actions_def.ACTIONS, only=cfg.actions)
    report['actions'] = sorted(a.name for a in bpy.data.actions)
    log(f'actions: {len(made)} new, {len(bpy.data.actions)} total')

    # 6. GLB (neutral face, all actions) ------------------------------------------------------------
    for o in list(bpy.data.objects):
        if o.name == 'charlib_cam': bpy.data.objects.remove(o)
    bpy.ops.export_scene.gltf(filepath=f'{out}/{cfg.name}_library.glb', export_format='GLB', export_animation_mode='ACTIONS',
                              export_force_sampling=True, export_image_format='JPEG', export_jpeg_quality=cfg.texture_quality)
    log('glb exported')

    # 7. library .blend + panel ------------------------------------------------------------------
    sc = bpy.context.scene
    sc['charlib_name'] = cfg.name; sc['charlib_rig'] = arm.name; sc['charlib_mat'] = mat.name
    sc.render.fps = 24
    neutral = bpy.data.images.load(f'{out}/textures/face_neutral.jpg'); neutral.name = 'face_neutral'
    for nd in mat.node_tree.nodes:
        if nd.type == 'TEX_IMAGE': nd.image = neutral
    if img.users == 0: bpy.data.images.remove(img)
    blend = f'{out}/{cfg.name}_library.blend'
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    neutral.filepath = '//textures/face_neutral.jpg'
    moods = [e for e in cfg.expressions if not e.startswith('mouth_')]
    src = open(os.path.join(PKG, 'panel_template.py')).read()
    src = src.replace("MOODS = ['neutral', 'smile', 'happy', 'sad', 'angry', 'surprised', 'shy', 'crying', 'eyes_closed', 'wink']",
                      f"MOODS = {moods!r}")
    txt = bpy.data.texts.new('charlib_panel.py'); txt.write(src); txt.use_module = True
    g = {'__name__': 'charlib_panel'}
    exec(compile(src, 'charlib_panel.py', 'exec'), g)            # registers + builds the face/mouth nodes
    arm.hide_set(True)
    for p in body.data.polygons: p.use_smooth = True
    sc.frame_start, sc.frame_end = 0, 48
    if 'Idle_Breathing' in bpy.data.actions:
        g['set_action']('Idle_Breathing')
    bpy.ops.wm.save_mainfile()
    log('library blend saved')

    # 8. previews --------------------------------------------------------------------------------
    if cfg.previews:
        previews(out, cfg, arm, body, mat, cam, g)
    bpy.ops.wm.open_mainfile(filepath=blend)   # drop preview-only changes

    # 9. report + source copy ----------------------------------------------------------------------
    report['seconds'] = round(time.time() - t0, 1)
    with open(f'{out}/build_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(f'{out}/source/charlib_panel.py', 'w') as f:
        f.write(src)
    log(f'done in {report["seconds"]}s -> {out}')
    return report


def previews(out, cfg, arm, body, mat, cam, g):
    sc = bpy.context.scene
    arm.animation_data.action = None
    scene_io.rest_pose(arm)
    scene_io.flat_scene(cfg.preview_res, samples=6)
    scene_io.flat_material_on(mat)
    # expression sheet (head close-up)
    head_scale = (cam['top'] - cam['head_z']) * 1.25
    scene_io.ortho_cam(cam['cx'], cam['head_z'] + 0.48 * (cam['top'] - cam['head_z']), head_scale)
    tiles = []
    for e in cfg.expressions:
        if e.startswith('mouth_'):
            g['set_expression']('neutral'); g['set_mouth'](g['MOUTHS'].index(e[6:]))
        else:
            g['set_mouth'](0); g['set_expression'](e)
        p = f'{out}/_work/e_{e}.png'; scene_io.render(p); tiles.append((e, Image.open(p).convert('RGBA')))
    g['set_mouth'](0); g['set_expression']('neutral')
    _sheet(tiles, 5).save(f'{out}/previews/expressions_sheet.jpg', quality=90)
    # turnaround
    height = cam['top']
    sc.render.resolution_x = int(cfg.preview_res * 0.66); sc.render.resolution_y = cfg.preview_res
    tiles = []
    for v in ('front', 'q34', 'side', 'back'):
        c = scene_io.ortho_cam(0, height / 2, height * 1.1, v); c.data.sensor_fit = 'VERTICAL'
        p = f'{out}/_work/t_{v}.png'; scene_io.render(p); tiles.append((v, Image.open(p).convert('RGBA')))
    _sheet(tiles, 4).save(f'{out}/previews/turnaround_sheet.jpg', quality=90)
    # actions contact sheet (q34, up to 3 key frames each)
    sc.render.resolution_x = sc.render.resolution_y = int(cfg.preview_res * 0.6)
    scene_io.ortho_cam(0, height * 0.53, height * 1.25, 'q34')
    tiles = []
    for name, keys in actions_def.ACTIONS.items():
        if name not in bpy.data.actions: continue
        g['set_action'](name)
        frames = [f for f, _ in keys if f > 0][:3]
        for f in frames:
            sc.frame_set(f); p = f'{out}/_work/a_{name}_{f}.png'; scene_io.render(p)
            tiles.append((f'{name} f{f}', Image.open(p).convert('RGBA')))
    _sheet(tiles, 6, label_h=30).save(f'{out}/previews/actions_sheet.jpg', quality=85)
    log('previews rendered')


def main(argv=None):
    argv = argv if argv is not None else (sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    ap = argparse.ArgumentParser(prog='charlib.build', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glb', required=True, help='rigged character GLB (Mixamo-style skeleton, e.g. Meshy export)')
    ap.add_argument('--out', required=True, help='output library folder')
    ap.add_argument('--name', help='character name (object/file names)')
    ap.add_argument('--config', help='JSON config (see character.example.json)')
    ap.add_argument('--no-previews', action='store_true')
    ap.add_argument('--allow-no-face', action='store_true', help='build actions only when the face has no painted features')
    a = ap.parse_args(argv)
    cfg = Config.load(a.config, name=a.name, previews=False if a.no_previews else None,
                      allow_no_face=True if a.allow_no_face else None)
    build(a.glb, a.out, cfg)


if __name__ == '__main__':
    main()
