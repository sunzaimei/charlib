"""Action library: name -> list of (frame, pose). 24 fps.

Pose conventions (degrees, rotations about WORLD axes, character faces -Y, her left = +X):
  * keys: bone names without prefix ('Spine2', 'Head'); 'L.Arm' / 'R.Arm' one side; 'LR.Arm' both sides mirrored.
    In P(...) keyword form write '__' instead of '.', e.g. P(LR__Arm=(0, 12, 0)).
  * Arm raise sideways: L.Arm y=-57 -> horizontal. Swing an arm/leg forward: x negative.
  * Bow / lean forward: Spine x positive. Head: nod down x+, turn to her left z+, tilt y.
  * 'hips_loc': (x, y, z) offset of the hips in metres (jumps, sitting).
Every pose starts from the relaxed A-pose below (REST_ARMS)."""
REST_ARMS = {'LR.Arm': (0, 12, 0)}


def P(**kw):
    d = dict(REST_ARMS)
    d.update({k.replace('__', '.'): v for k, v in kw.items()})
    return d


R0 = P()
ACTIONS={
'Idle_Breathing':[(0,R0),(24,P(Spine1=(-2,0,0),Spine2=(-3,0,0),Head=(3,0,2),LR__Arm=(0,9,0),LR__ForeArm=(-6,0,0))),(48,R0)],
'Wave_Hello':[(0,R0),
  (8,P(R__Arm=(-10,-80,0),R__ForeArm=(0,-70,0),Head=(0,6,-8))),
  (14,P(R__Arm=(-10,-80,0),R__ForeArm=(0,-45,0),Head=(0,6,-8))),
  (20,P(R__Arm=(-10,-80,0),R__ForeArm=(0,-95,0),Head=(0,6,-8))),
  (26,P(R__Arm=(-10,-80,0),R__ForeArm=(0,-45,0),Head=(0,6,-8))),
  (32,P(R__Arm=(-10,-80,0),R__ForeArm=(0,-95,0),Head=(0,6,-8))),
  (40,P(R__Arm=(-10,-80,0),R__ForeArm=(0,-70,0),Head=(0,6,-8))),(48,R0)],
'Bow_Greeting':[(0,R0),
  (12,P(LR__Arm=(-40,22,0),LR__ForeArm=(-45,0,-55))),
  (24,P(LR__Arm=(-40,22,0),LR__ForeArm=(-45,0,-55),Spine=(12,0,0),Spine1=(12,0,0),Spine2=(8,0,0),Head=(12,0,0))),
  (40,P(LR__Arm=(-40,22,0),LR__ForeArm=(-45,0,-55),Spine=(13,0,0),Spine1=(13,0,0),Spine2=(9,0,0),Head=(13,0,0))),
  (52,P(LR__Arm=(-40,22,0),LR__ForeArm=(-45,0,-55))),(60,R0)],
'Point_Forward':[(0,R0),(10,P(R__Arm=(-75,-15,0),R__ForeArm=(-5,0,0),Head=(0,0,-10),Spine2=(0,0,-8))),(36,P(R__Arm=(-75,-15,0),R__ForeArm=(-5,0,0),Head=(0,0,-10),Spine2=(0,0,-8)))],
'Arms_Crossed':[(0,R0),(12,P(LR__Arm=(-55,20,0),LR__ForeArm=(-20,0,-100),Head=(-4,0,0))),(48,P(LR__Arm=(-55,20,0),LR__ForeArm=(-20,0,-100),Head=(-4,5,0)))],
'Hands_On_Hips_Angry':[(0,R0),(10,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(-6,0,0),Spine=(-4,0,0))),
  (18,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(4,0,0),R__UpLeg=(-30,0,0),R__Leg=(40,0,0))),
  (22,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(-6,0,0))),(48,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(-6,0,0)))],
'Sad_Slump':[(0,P(Spine=(6,0,0),Spine2=(8,0,0),Head=(22,0,-5),LR__Arm=(0,20,0),LR__Shoulder=(8,0,0))),
  (24,P(Spine=(7,0,0),Spine2=(9,0,0),Head=(24,0,-8),LR__Arm=(0,21,0),LR__Shoulder=(9,0,0))),
  (48,P(Spine=(6,0,0),Spine2=(8,0,0),Head=(22,0,-5),LR__Arm=(0,20,0),LR__Shoulder=(8,0,0)))],
'Happy_Jump':[(0,R0),
  (6,P(hips_loc=(0,0,-0.07),LR__UpLeg=(-28,0,0),LR__Leg=(50,0,0),LR__Foot=(-22,0,0),Spine=(10,0,0),LR__Arm=(25,5,0))),
  (12,P(hips_loc=(0,0,0.18),LR__Leg=(25,0,0),LR__Foot=(20,0,0),LR__Arm=(0,-110,0),LR__ForeArm=(0,-15,0),Head=(-10,0,0))),
  (16,P(hips_loc=(0,0,0.22),LR__Leg=(30,0,0),LR__Foot=(20,0,0),LR__Arm=(0,-115,0),LR__ForeArm=(0,-15,0),Head=(-10,0,0))),
  (22,P(hips_loc=(0,0,-0.06),LR__UpLeg=(-25,0,0),LR__Leg=(45,0,0),LR__Foot=(-20,0,0),Spine=(8,0,0),LR__Arm=(0,-40,0))),
  (30,R0)],
'Shy_Cover_Mouth':[(0,R0),(14,P(R__Arm=(-50,22,0),R__ForeArm=(-105,0,-35),Head=(10,-10,-12),Spine2=(3,0,-5),L__Arm=(-10,18,0),L__ForeArm=(-25,0,-40))),
  (48,P(R__Arm=(-50,22,0),R__ForeArm=(-105,0,-35),Head=(12,-12,-14),Spine2=(3,0,-5),L__Arm=(-10,18,0),L__ForeArm=(-25,0,-40)))],
'Think_Hand_Chin':[(0,R0),(14,P(R__Arm=(-35,18,0),R__ForeArm=(-115,0,-40),L__Arm=(-30,16,0),L__ForeArm=(-10,0,-80),Head=(5,-12,-8))),
  (48,P(R__Arm=(-35,18,0),R__ForeArm=(-115,0,-40),L__Arm=(-30,16,0),L__ForeArm=(-10,0,-80),Head=(3,-14,-14)))],
'Surprised_Jolt':[(0,R0),(5,P(hips_loc=(0,0.03,0.02),Spine=(-8,0,0),Head=(-10,0,0),LR__Arm=(-25,-30,0),LR__ForeArm=(-70,0,-10))),
  (24,P(hips_loc=(0,0.02,0.0),Spine=(-5,0,0),Head=(-6,0,0),LR__Arm=(-25,-28,0),LR__ForeArm=(-70,0,-10)))],
'Nod_Yes':[(0,R0),(6,P(Head=(15,0,0))),(12,R0),(18,P(Head=(15,0,0))),(24,R0)],
'Shake_No':[(0,R0),(8,P(Head=(0,0,22))),(16,P(Head=(0,0,-22))),(24,P(Head=(0,0,18))),(32,R0)],
'Talk_Explain':[(0,R0),(12,P(R__Arm=(-30,-12,0),R__ForeArm=(-55,0,25),Head=(0,4,0))),(24,P(R__Arm=(-35,-18,0),R__ForeArm=(-60,0,35),Head=(-3,-3,5))),
  (36,P(L__Arm=(-30,-12,0),L__ForeArm=(-55,0,25),Head=(0,-4,0))),(48,R0)],
'Look_Around':[(0,R0),(12,P(Head=(0,0,40),Spine2=(0,0,10))),(28,P(Head=(0,0,40),Spine2=(0,0,10))),(44,P(Head=(0,0,-40),Spine2=(0,0,-10))),(60,P(Head=(0,0,-40),Spine2=(0,0,-10))),(72,R0)],
'Sit_Chair':[(0,R0),(18,P(hips_loc=(0,0.06,-0.40),LR__UpLeg=(-88,0,0),LR__Leg=(88,0,0),LR__Arm=(-40,15,0),LR__ForeArm=(-35,0,-25))),
  (48,P(hips_loc=(0,0.06,-0.40),LR__UpLeg=(-88,0,0),LR__Leg=(88,0,0),LR__Arm=(-40,15,0),LR__ForeArm=(-35,0,-25),Head=(3,0,5)))],
}
# ---- talking body loops (pair with the Talking mouth in the panel) ----
_H=dict(LR__Arm=(-22,14,0),LR__ForeArm=(-50,0,-45))  # hands loosely together in front
ACTIONS.update({
'Talk_Idle':[(0,P(**_H)),(16,P(**_H,Head=(-4,4,6),Spine2=(-2,0,2))),(32,P(**_H,Head=(4,-3,-3))),(48,P(**_H,Head=(-3,-4,-7),Spine2=(-2,0,-2))),
  (64,P(**_H,Head=(3,3,3))),(80,P(**_H,Head=(-2,2,-4))),(96,P(**_H))],
'Talk_Both_Hands':[(0,P(**_H)),
  (10,P(LR__Arm=(-32,2,0),LR__ForeArm=(-60,0,0),Head=(-5,0,0),Spine2=(-3,0,0))),
  (20,P(LR__Arm=(-28,6,0),LR__ForeArm=(-55,0,-15),Head=(3,5,4))),
  (30,P(LR__Arm=(-35,-4,0),LR__ForeArm=(-62,0,8),Head=(-4,-4,-5))),
  (40,P(**_H,Head=(2,0,0))),(48,P(**_H))],
'Talk_Point_Explain':[(0,P(**_H)),
  (10,P(L__Arm=(-22,14,0),L__ForeArm=(-50,0,-45),R__Arm=(-45,-15,0),R__ForeArm=(-60,0,20),Head=(0,0,-6))),
  (18,P(L__Arm=(-22,14,0),L__ForeArm=(-50,0,-45),R__Arm=(-55,-15,0),R__ForeArm=(-40,0,20),Head=(4,0,-6))),
  (26,P(L__Arm=(-22,14,0),L__ForeArm=(-50,0,-45),R__Arm=(-45,-15,0),R__ForeArm=(-60,0,20),Head=(-2,0,-6))),
  (34,P(L__Arm=(-22,14,0),L__ForeArm=(-50,0,-45),R__Arm=(-55,-15,0),R__ForeArm=(-40,0,20),Head=(4,0,-6))),
  (48,P(**_H))],
'Talk_Angry':[(0,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(-6,0,0))),
  (8,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(8,0,0),Spine2=(5,0,0))),
  (16,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(-6,0,6))),
  (24,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(8,0,0),Spine2=(5,0,0))),
  (32,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(-6,0,-6))),
  (48,P(LR__Arm=(10,-45,0),LR__ForeArm=(15,115,0),Head=(-6,0,0)))],
'Talk_Shy':[(0,P(**_H,Head=(10,-8,-10))),(24,P(**_H,Head=(12,-12,-14),Spine=(0,0,-4))),(48,P(**_H,Head=(10,-8,-10)))],
'Talk_Sad':[(0,P(Spine2=(6,0,0),Head=(18,0,-4),LR__Arm=(0,20,0))),(24,P(Spine2=(7,0,0),Head=(14,0,4),LR__Arm=(0,20,0))),(48,P(Spine2=(6,0,0),Head=(18,0,-4),LR__Arm=(0,20,0)))],
})
