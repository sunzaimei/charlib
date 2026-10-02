"""Build configuration. Everything has a sensible default; override with a JSON file (see character.example.json)."""
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    name: str = "Character"                 # used for object names, panel title, file names
    face_res: int = 1024                    # resolution of the front-view face image used for painting
    texture_quality: int = 92               # JPEG quality of the exported face textures
    # Face camera framing. None = derive from the head bone and the top of the mesh.
    face_center_z: float | None = None
    face_scale: float | None = None
    # Landmarks in face-image pixels [x, y]. None = auto-detect. Set these if detection picks the wrong spots.
    left_eye: list | None = None             # viewer's left eye centre
    right_eye: list | None = None
    mouth: list | None = None
    expressions: list = field(default_factory=lambda: [
        "neutral", "smile", "happy", "sad", "angry", "surprised", "shy", "crying", "eyes_closed", "wink",
        "mouth_A", "mouth_I", "mouth_U", "mouth_E", "mouth_O"])
    actions: list | None = None              # None = all actions in charlib.actions_def
    allow_no_face: bool = False             # build actions only if the face has no painted features
    previews: bool = True
    preview_res: int = 512

    @classmethod
    def load(cls, path=None, **overrides):
        data = {}
        if path:
            with open(path) as f:
                data = json.load(f)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_json(self):
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
