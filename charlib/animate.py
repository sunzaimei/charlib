"""Hand-keyed actions for any Mixamo-style rig.

Poses are written as rotations in degrees around the *world* axes (character faces -Y, her left = +X),
relative to a canonical relaxed A-pose. Arm bones are corrected first, so the same pose data works for
T-pose and A-pose rigs. See actions_def.py for the conventions."""
import math

import bpy
from mathutils import Euler, Vector

CANON_DIR = {  # canonical rest directions (armature space) the pose data was authored against
    'LeftArm': Vector((0.534, 0.071, -0.842)), 'LeftForeArm': Vector((0.615, -0.065, -0.786)),
    'RightArm': Vector((-0.534, 0.071, -0.842)), 'RightForeArm': Vector((-0.615, -0.065, -0.786)),
}
PARENT = {'LeftForeArm': 'LeftArm', 'RightForeArm': 'RightArm'}


class Poser:
    def __init__(self, arm, bmap):
        self.arm, self.bmap = arm, bmap
        self.corr = {}
        for c, canon in CANON_DIR.items():
            b = arm.data.bones[bmap[c]]
            child = [x for x in b.children]
            tail = child[0].head_local if child else b.tail_local
            rest = (tail - b.head_local).normalized()
            self.corr[c] = rest.rotation_difference(canon.normalized())
        self.world_inv = arm.matrix_world.to_3x3().inverted()

    def pb(self, canon):
        return self.arm.pose.bones[self.bmap[canon]]

    def set_rot(self, canon, x=0, y=0, z=0, mirror=False):
        if canon not in self.bmap:
            return
        if mirror: y, z = -y, -z
        R = Euler((math.radians(x), math.radians(y), math.radians(z)), 'XYZ').to_quaternion()
        C = self.corr.get(canon)
        if C is not None:
            P = self.corr.get(PARENT.get(canon))
            R = (P.inverted() @ R @ C) if P is not None else (R @ C)
        pb = self.pb(canon); Bq = pb.bone.matrix_local.to_quaternion()
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = Bq.inverted() @ R @ Bq

    def apply(self, pose):
        for pb in self.arm.pose.bones:
            pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
        for c in self.corr:            # arms to the canonical relaxed pose even if not mentioned
            self.set_rot(c)
        for k, v in pose.items():
            if k == 'hips_loc':
                pb = self.pb('Hips'); Bq = pb.bone.matrix_local.to_quaternion()
                pb.location = Bq.inverted() @ (self.world_inv @ Vector(v)); continue
            if k.startswith('LR.'):
                self.set_rot('Left' + k[3:], *v); self.set_rot('Right' + k[3:], *v, mirror=True)
            elif k.startswith('L.'): self.set_rot('Left' + k[2:], *v)
            elif k.startswith('R.'): self.set_rot('Right' + k[2:], *v, mirror=True)
            else: self.set_rot(k, *v)

    def key(self, frame):
        for pb in self.arm.pose.bones:
            pb.keyframe_insert('rotation_quaternion', frame=frame)
            if pb.name == self.bmap['Hips']:
                pb.keyframe_insert('location', frame=frame)


def build_actions(arm, bmap, actions, only=None):
    poser = Poser(arm, bmap)
    if arm.animation_data is None:
        arm.animation_data_create()
    made = []
    for name, keys in actions.items():
        if only and name not in only:
            continue
        if name in bpy.data.actions:
            bpy.data.actions.remove(bpy.data.actions[name])
        arm.animation_data.action = None
        for f, p in keys:
            poser.apply(p); poser.key(f)
        a = arm.animation_data.action; a.name = name; a.use_fake_user = True
        made.append(name)
    arm.animation_data.action = None
    poser.apply({})
    return made
