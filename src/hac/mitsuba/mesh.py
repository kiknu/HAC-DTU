from dataclasses import dataclass
from pathlib import Path

from hac.mesh import convert_stl_to_obj

@dataclass(frozen=True, slots=True)
class MitsubaMesh:
    obj_path: Path

    @classmethod
    def from_stl_path(cls, stl_path: Path) -> MitsubaMesh:
        obj_path = stl_path.with_suffix(".obj")
        convert_stl_to_obj(stl_path, obj_path)

        return cls(obj_path=obj_path)

    def to_filename(self) -> str:
        return str(self.obj_path)
