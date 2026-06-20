"""
Meshing utilities using `trimesh`.

We need conversion from STL to OBJ if we use Mitsuba.
"""


from pathlib import Path
import trimesh


def convert_stl_to_obj(stl_path: Path, obj_path: Path | None = None) -> Path:
    """
    Converts an STL file to an OBJ file.

    If `obj_path` is not provided, the OBJ file will be saved in the same directory as the STL file with the same name but with an `.obj` extension.
    """
    if obj_path is None:
        obj_path = stl_path.with_suffix(".obj")

    mesh = trimesh.load(stl_path, force="mesh")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text(trimesh.exchange.obj.export_obj(mesh))

    return obj_path
