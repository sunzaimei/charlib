# charlib: rigged character GLB → character library

One command turns a rigged character (for example a Meshy export with a Mixamo skeleton) into a library ready for 漫剧 or games:

| Output | What it is |
|---|---|
| `<Name>_library.blend` | Rig and mesh with all actions, the expression and talking material, and a **Character** panel (3D Viewport → N → Character) |
| `<Name>_library.glb` | All animation clips with the neutral face, for Unity, Unreal or Godot |
| `textures/face_*.jpg` | One full texture per expression (neutral, smile, happy, sad, angry, surprised, shy, crying, eyes_closed, wink, mouth_A/I/U/E/O) |
| `textures/mouth_mask.png` | Limits the talking mouth layer to the mouth area |
| `previews/` | `face_layout.png` (landmark check), expression sheet, turnaround and action contact sheet |
| `build_report.json` | Detected landmarks, camera, bone count and actions, for debugging |

## Install

bpy ships wheels only for **Python 3.11**, so both paths pin that interpreter.

### With uv (recommended)

```bash
uv venv --python 3.11            # create .venv on Python 3.11
uv sync --extra dev              # install charlib + deps from pyproject.toml (incl. pytest)
#   uv pip install -r requirements.txt   # alternative: install from requirements.txt instead of the lockfile
```

### With venv + pip

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # bpy needs Python 3.11
pip install -r requirements.txt
```

## Build

With uv, prefix the command with `uv run` (no need to activate the venv); with venv+pip, activate first and drop the prefix.

```bash
uv run python -m charlib.build --glb path/to/character.glb --out out/MyChar --name MyChar
#   or, after `source .venv/bin/activate`:
#   python -m charlib.build --glb path/to/character.glb --out out/MyChar --name MyChar
#   or use the installed entry point:  charlib --glb ... --out ... --name ...
#
#   --config character.json   override landmarks, framing or expression list (see character.example.json)
#   --no-previews             skip the preview renders (faster)
#   --allow-no-face           blank face texture: build actions only
```

A full build takes about 2 minutes on a laptop CPU. The build runs these steps:

1. **Import and normalise:** rename the rig and mesh, remove the bone widget spheres, and check for a Mixamo-compatible skeleton (`mixamorig:*`, `mixamorig_*` or plain names).
2. **Face render and texel map:** render an orthographic front view of the face, and map every texel on the front of the face to a pixel in that render (depth-tested).
3. **Landmarks:** auto-detect the eyes and mouth, pairing the eyes by symmetry and mirroring when hair or an earring touches one eye. Check `previews/face_layout.png`. If the green circles are wrong, put `left_eye`, `right_eye` and `mouth` in the config.
4. **Expressions:** each expression is drawn in a fixed 1024 px reference face space, warped onto the real face with the landmark affine transform, then baked into the UV texture (with seam bleeding).
5. **Actions:** 22 hand-keyed clips, including 7 talking loops, written as world-axis rotations from a canonical A-pose. Arm bones are corrected first, so the clips work on both T-pose and A-pose rigs.
6. **Exports:** the GLB, the `.blend` with the panel, and the previews.

## Using the library in Blender

Open `<Name>_library.blend`, run the `charlib_panel.py` text once (Scripting → Run Script, or allow Auto Run), then press N → **Character**.

- **Expression:** dropdown, ◀ ▶ buttons, and **Key Expression Here** (adds an `expr:<name>` timeline marker).
- **Talking:** a mouth layer on top of any expression. Mouth 0–5 is keyframable. **Auto Talk** keys random mouth movement, and **Lip-sync Audio** keys the mouth to a wav or mp3 and adds the audio. **Remove Lip-sync & Voice** clears both.
- **Action:** dropdown, ◀ ▶ buttons, Play, and *Keep timeline length*.

## Repo layout

```
charlib/build.py           CLI + pipeline orchestration (bpy)
charlib/scene_io.py        import/normalise, face camera, renders, texel map (bpy)
charlib/face.py            landmark detection, expression painting, texture baking (numpy/PIL/OpenCV only)
charlib/animate.py         pose engine (world-axis poses, arm canonicalisation)
charlib/actions_def.py     the action library: edit or add clips here
charlib/panel_template.py  the in-Blender panel, embedded into every library .blend
tests/test_smoke.py        end-to-end build on a GLB (set CHARLIB_TEST_GLB)
```

## Adding things

- **New action:** add a `name: [(frame, P(...)), ...]` entry to `actions_def.py`. The conventions are in that file's docstring.
- **New expression:** add a branch to `recipe()` in `face.py`, drawing in reference coordinates (the eye and mouth constants are at the top of the file), then add the name to `Config.expressions`.

## Limitations

- **Face must be painted on the texture:** expressions repaint it. A blank face needs `--allow-no-face`, which builds actions only.
- **Character must face −Y:** the front, which is the glTF default after import, with one face material.
- **Hand-keyed motion:** the actions are stylised. For natural motion, add Mixamo or Meshy clips to the GLB before building and they are kept.
