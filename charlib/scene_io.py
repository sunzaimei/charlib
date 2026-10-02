"""Blender-side steps: import the GLB, normalise it, render the face, map texels to the face image."""
import math
import os

import bpy
import numpy as np

CANON_BONES = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
               "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
               "RightShoulder", "RightArm", "RightForeArm", "RightHand",
               "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
               "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]
REQUIRED = ["Hips", "Spine", "Head", "LeftArm", "LeftForeArm", "RightArm", "RightForeArm", "LeftUpLeg", "RightUpLeg"]


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(path))
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and o.find_armature() is not None]
    if len(arms) != 1:
        raise RuntimeError(f"expected exactly one armature, found {len(arms)}")
    if not meshes:
        raise RuntimeError("no skinned mesh found")
    body = max(meshes, key=lambda o: len(o.data.vertices))
    return arms[0], body


def bone_map(arm):
    """Map canonical Mixamo names (Hips, LeftArm, ...) to the actual bone names (mixamorig:Hips, mixamorig_Hips, Hips...)."""
    names = [b.name for b in arm.data.bones]
    m = {}
    for c in CANON_BONES:
        cands = [n for n in names if n == c or n.endswith(':' + c) or n.endswith('_' + c) or n.endswith('.' + c)]
        if cands:
            m[c] = min(cands, key=len)
    missing = [c for c in REQUIRED if c not in m]
    if missing:
        raise RuntimeError(f"rig is not Mixamo-compatible, missing bones: {missing}")
    return m


def normalise(arm, body, name):
    """Rename objects, drop bone custom shapes (the 'big sphere' problem) and their helper meshes."""
    shapes = {pb.custom_shape for pb in arm.pose.bones if pb.custom_shape}
    for pb in arm.pose.bones:
        pb.custom_shape = None
    for o in shapes:
        if o and o.name in bpy.data.objects:
            bpy.data.objects.remove(o)
    for o in list(bpy.data.objects):
        if o.type == 'MESH' and o != body and o.find_armature() is None and len(o.data.vertices) < 200:
            bpy.data.objects.remove(o)          # leftover widget meshes
    arm.name = f"{name}_Rig"
    body.name = f"{name}_Body"
    rename = {'Running': 'Run_Loop', 'Walking': 'Walk_Loop'}
    for a in bpy.data.actions:
        a.use_fake_user = True
        if a.name in rename:
            a.name = rename[a.name]
        elif '|' in a.name or a.name.lower().startswith('armature'):
            a.name = 'Base_' + a.name.split('|')[-1]
    rest_pose(arm)
    return arm, body


def rest_pose(arm):
    if arm.animation_data:
        arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def face_material(body):
    """The material + image texture that holds the face (the material used most by front-facing head triangles)."""
    mats = [s.material for s in body.material_slots if s.material]
    if not mats:
        raise RuntimeError("body has no material")
    def img_of(m):
        for n in m.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image:
                return n.image
    cands = [(m, img_of(m)) for m in mats if img_of(m)]
    if not cands:
        raise RuntimeError("no image texture on the body material")
    if len(cands) == 1:
        return cands[0]
    head_z = max(v.co.z for v in body.data.vertices) * 0.75
    counts = {}
    for p in body.data.polygons:
        if p.center.z > head_z:
            counts[p.material_index] = counts.get(p.material_index, 0) + 1
    mi = max(counts, key=counts.get)
    m = body.material_slots[mi].material
    return m, img_of(m)


def world_coords(body):
    M = body.matrix_world
    co = np.empty(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    R = np.array(M.to_3x3()); t = np.array(M.translation)
    return co @ R.T + t


def face_camera(arm, body, bmap, cfg):
    """Ortho front camera framing the face: centre z and vertical extent (scale) in metres."""
    co = world_coords(body)
    top = co[:, 2].max()
    head = arm.matrix_world @ arm.data.bones[bmap['Head']].head_local
    h = top - head.z
    cz = cfg.face_center_z if cfg.face_center_z is not None else head.z + 0.29 * h
    s = cfg.face_scale if cfg.face_scale is not None else 0.64 * h
    cx = float(head.x)
    return dict(cx=cx, cz=float(cz), scale=float(s), head_z=float(head.z), top=float(top))


def flat_scene(res, samples=4):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = True
    sc.view_settings.view_transform = 'Standard'
    return sc


def flat_material_on(mat, image=None):
    """Temporarily make the material pure emission of its texture (exact colours for renders)."""
    nt = mat.node_tree
    tex = [n for n in nt.nodes if n.type == 'TEX_IMAGE'][0]
    if image is not None:
        tex.image = image
    out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    em = nt.nodes.get('charlib_flat') or nt.nodes.new('ShaderNodeEmission')
    em.name = 'charlib_flat'
    src = nt.nodes.get('FaceMix')
    nt.links.new((src.outputs[2] if src else tex.outputs['Color']), em.inputs['Color'])
    nt.links.new(em.outputs[0], out.inputs['Surface'])


def ortho_cam(cx, cz, scale, view='front', res_x=None, res_y=None):
    sc = bpy.context.scene
    cam = bpy.data.objects.get('charlib_cam')
    if not cam:
        cd = bpy.data.cameras.new('charlib_cam'); cam = bpy.data.objects.new('charlib_cam', cd)
        sc.collection.objects.link(cam)
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = scale
    d = 6.0
    pos = {'front': ((cx, -d, cz), (math.pi / 2, 0, 0)), 'side': ((d, 0, cz), (math.pi / 2, 0, math.pi / 2)),
           'q34': ((d * .7, -d * .7, cz), (math.pi / 2, 0, math.pi / 4)), 'back': ((cx, d, cz), (math.pi / 2, 0, math.pi))}
    cam.location, cam.rotation_euler = pos[view]
    sc.camera = cam
    return cam


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def save_texture(image, path):
    """Write the image's pixels to PNG without touching the datablock."""
    from PIL import Image as PILImage
    w, h = image.size
    px = np.empty(w * h * 4, np.float32); image.pixels.foreach_get(px)
    px = (px.reshape(h, w, 4)[::-1, :, :3].clip(0, 1) * 255 + 0.5).astype(np.uint8)
    PILImage.fromarray(px).save(path)


# ------------------------------------------------------------------ texel <-> face-image map
def _raster(tri2d, attrs, W, H):
    out = []
    for i in range(len(tri2d)):
        t = tri2d[i]
        x0, y0 = np.floor(t.min(0)).astype(int); x1, y1 = np.ceil(t.max(0)).astype(int)
        x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, W - 1); y1 = min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        a, b, c = t
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not m.any():
            continue
        vals = np.stack([w0[m], w1[m], w2[m]], -1) @ attrs[i]
        out.append(((ys[m] - 0.5).astype(int), (xs[m] - 0.5).astype(int), vals))
    return out


def build_face_map(body, cam, res, tex_size, path):
    """For every texel on the front of the face: which face-image pixel it shows. Saves map.npz."""
    mesh = body.data
    mesh.calc_loop_triangles()
    co = world_coords(body)
    nt = len(mesh.loop_triangles)
    tv = np.zeros(nt * 3, int); tl = np.zeros(nt * 3, int)
    mesh.loop_triangles.foreach_get('vertices', tv); mesh.loop_triangles.foreach_get('loops', tl)
    tv = tv.reshape(-1, 3); tl = tl.reshape(-1, 3)
    luv = np.zeros(len(mesh.loops) * 2); mesh.uv_layers.active.data.foreach_get('uv', luv); luv = luv.reshape(-1, 2)
    S, CX, CZ = cam['scale'], cam['cx'], cam['cz']
    P = co[tv]
    zlo = CZ - S / 2 - 0.05
    hi = np.where(P[:, :, 2].max(1) > zlo)[0]
    Ph = P[hi]
    I2 = np.stack([(Ph[..., 0] - CX) / S * res + res / 2, (CZ + S / 2 - Ph[..., 2]) / S * res], -1)
    D = np.full((res, res), np.inf)
    for ys, xs, v in _raster(I2, Ph[:, :, 1:2], res, res):
        np.minimum.at(D, (ys, xs), v[:, 0])
    n = np.cross(Ph[:, 1] - Ph[:, 0], Ph[:, 2] - Ph[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    fi = np.where(n[:, 1] < -0.15)[0]
    TW, TH = tex_size
    UV = luv[tl[hi[fi]]]
    UVp = np.stack([UV[..., 0] % 1.0 * TW, (1 - UV[..., 1] % 1.0) * TH], -1)
    attrs = np.concatenate([I2[fi], Ph[fi][:, :, 1:2]], -1)
    acc = [[], [], [], [], []]
    for ys, xs, v in _raster(UVp, attrs, TW, TH):
        for a, x in zip(acc, (ys, xs, v[:, 0], v[:, 1], v[:, 2])):
            a.append(x)
    TY, TX, PX, PY, DY = [np.concatenate(a) for a in acc]
    inb = (PX >= 0) & (PX < res - 1) & (PY >= 0) & (PY < res - 1)
    TY, TX, PX, PY, DY = [a[inb] for a in (TY, TX, PX, PY, DY)]
    vis = DY <= D[PY.astype(int), PX.astype(int)] + 0.004 * (S / 0.36)
    np.savez(path, ty=TY[vis], tx=TX[vis], px=PX[vis], py=PY[vis], res=res, tex_w=TW, tex_h=TH)
    return int(vis.sum())
