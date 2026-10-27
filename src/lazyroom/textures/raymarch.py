"""Ray-marched 3D texture presets (ImageToolbox ``TextureFilterType.*3D``, backend ``raymarch``).

Port of ``RaymarchTextureGenerator.kt`` (ImageToolbox, Apache-2.0, T8RIN): a GLSL
sphere-tracing shader run through GPUImage.  The shader is ported to a numba
kernel in :mod:`._raymarch_impl`; uniforms, ranges and defaults follow
``RaymarchTextureParams`` / ``RaymarchParams.kt`` / ``TextureParams.withDefaultsFor``.
No randomness is involved, so the render ``seed`` has no effect.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.params import Color, File, Float, Int
from ._registry import TexturePreset, argb_hex, register

_OBJECT_COLOR = argb_hex(-4609051)  # TextureParams.withDefaultsFor(isRaymarch).foregroundColor

# (upstream enum name, id, title, kind, shows thickness, shows roundness)
_SHAPES = [
    ("Sphere3D", "sphere_3d", "3D Sphere", False, False),
    ("Cube3D", "cube_3d", "3D Cube", False, True),
    ("CubeStair3D", "cube_stair_3d", "Cube Stair", False, True),
    ("CubeSpheres3D", "cube_spheres_3d", "Cube Spheres", True, False),
    ("EmptiedCube3D", "emptied_cube_3d", "Emptied Cube", False, True),
    ("Torus3D", "torus_3d", "3D Torus", True, False),
    ("MobiusTorus3D", "mobius_torus_3d", "Möbius Torus", True, True),
    ("GyroRings3D", "gyro_rings_3d", "Gyro Rings", True, False),
    ("TorusGem3D", "torus_gem_3d", "Torus Gem", True, True),
    ("CubeOctahedron3D", "cube_octahedron_3d", "Cube Octahedron", False, True),
    ("InfiniteSpheroids3D", "infinite_spheroids_3d", "Infinite Spheroids", True, False),
    ("InfiniteStructure3D", "infinite_structure_3d", "Infinite Structure", True, True),
    ("InfiniteCylinders3D", "infinite_cylinders_3d", "Infinite Cylinders", True, False),
    ("TruchetRings3D", "truchet_rings_3d", "3D Truchet Rings", True, False),
    ("TruchetPipes3D", "truchet_pipes_3d", "3D Truchet Pipes", True, False),
]

# hidden uniforms keep the RaymarchTextureParams defaults
_DEFAULTS = {"size": 1.0, "thickness": 0.18, "roundness": 0.06, "count": 3, "inner_radius": 0.9}

_NOTES_COMMON = ("GLSL shader ported to numba in float64 (GPU: highp float32); the render seed has no effect "
                 "(no randomness upstream).")
_ENV_NOTE = (" Optional environment image: downscaled to fit 2048 px like ImageToolbox (coil size(2048)); "
             "smaller images are used as-is (coil may resample them), sampled bilinear/clamp-to-edge.")
_TRUCHET_NOTE = (" Per-cell orientation hash fract(sin(dot)) is evaluated in float32; GPU sin precision is "
                 "hardware-dependent, so tile orientations can differ from a given device.")


def _params(name: str, thickness: bool, roundness: bool) -> list:
    ps = []
    if name == "EmptiedCube3D":
        ps.append(Float("inner_radius", 0.9, title="Inner radius", min=0.5, max=1.2, step=0.01))
    ps += [
        Color("object_color", _OBJECT_COLOR, title="Object color"),
        Color("background_color", "#00000000", title="Background color"),
        Float("size", 1.0, title="Object size", min=0.25, max=2.5, step=0.01),
    ]
    if thickness:
        ps.append(Float("thickness", 0.18, title="Thickness", min=0.03, max=0.4, step=0.01))
    if roundness:
        ps.append(Float("roundness", 0.06, title="Roundness", min=0.0, max=0.2, step=0.01))
    ps += [
        Float("rotation_x", 20.0, title="Rotation X", min=-180.0, max=180.0, step=1, unit="deg"),
        Float("rotation_y", 30.0, title="Rotation Y", min=-180.0, max=180.0, step=1, unit="deg"),
        Float("rotation_z", 0.0, title="Rotation Z", min=-180.0, max=180.0, step=1, unit="deg"),
        Float("camera_distance", 3.5, title="Camera distance", min=1.5, max=8.0, step=0.01),
        Float("camera_yaw", 0.0, title="Camera azimuth", min=-180.0, max=180.0, step=1, unit="deg"),
        Float("camera_pitch", 0.0, title="Camera elevation", min=-80.0, max=80.0, step=1, unit="deg"),
        Float("field_of_view", 45.0, title="Field of view", min=20.0, max=90.0, step=1, unit="deg"),
        Float("metallic", 0.8, title="Metallic", min=0.0, max=1.0, step=0.01),
        Float("roughness", 0.25, title="Roughness", min=0.03, max=1.0, step=0.01),
        Float("transmission", 0.0, title="Transmission", min=0.0, max=1.0, step=0.01),
        Float("light_angle", -35.0, title="Light azimuth", min=-180.0, max=180.0, step=1, unit="deg"),
        Float("light_elevation", 50.0, title="Light elevation", min=-80.0, max=80.0, step=1, unit="deg"),
    ]
    if name == "MobiusTorus3D":
        ps.append(Int("count", 3, title="Count", min=1, max=12))
    ps.append(File("environment", "", title="Environment image", file_kind="image",
                   description="Optional equirectangular image used for reflections/refraction "
                               "(Use image for reflections)."))
    return ps


def _load_environment(path: str) -> np.ndarray:
    """Load an environment image as premultiplied RGBA float32 (GPU texture semantics)."""
    import cv2

    from ..core.io import load_image

    img, _ = load_image(path)
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[2] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], np.float32)])
    img = np.clip(img[..., :4], 0.0, 1.0)
    h, w = img.shape[:2]
    if max(w, h) > 2048:
        s = 2048.0 / max(w, h)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    img = img.copy()
    img[..., :3] *= img[..., 3:4]
    return np.ascontiguousarray(img, dtype=np.float32)


def _make_render(kind: int):
    def render_fn(width: int, height: int, seed: int, v: dict) -> np.ndarray:
        from . import _raymarch_impl as K

        def g(name):
            return float(v.get(name, _DEFAULTS.get(name, 0.0)))

        u = np.zeros(K.N_UNIFORMS, np.float64)
        u[K.U_SIZE] = g("size")
        u[K.U_THICKNESS] = g("thickness")
        u[K.U_ROUNDNESS] = g("roundness")
        u[K.U_COUNT] = g("count")
        u[K.U_INNER] = g("inner_radius")
        for (ci, si), ang in (((K.U_RXC, K.U_RXS), g("rotation_x")), ((K.U_RYC, K.U_RYS), g("rotation_y")),
                              ((K.U_RZC, K.U_RZS), g("rotation_z"))):
            a = math.radians(ang)
            u[ci], u[si] = math.cos(a), math.sin(a)
        cam = np.array([width / height, g("camera_distance"), g("camera_yaw"), g("camera_pitch"),
                        g("field_of_view"), g("metallic"), g("roughness"), g("transmission"),
                        g("light_angle"), g("light_elevation")], np.float64)
        obj = np.array(v["object_color"], np.float64)
        bg = np.array(v["background_color"], np.float64)
        env_path = v.get("environment") or ""
        if env_path:
            env = _load_environment(env_path)
            has_env = True
        else:
            env = np.zeros((1, 1, 4), np.float32)
            has_env = False
        return K.render(kind, int(width), int(height), u, cam, obj, bg, env, has_env)

    return render_fn


for _kind, (_name, _id, _title, _thick, _round) in enumerate(_SHAPES):
    register(TexturePreset(
        id=_id, title=_title, category="patterns", backend="raymarch", upstream=_name,
        params=_params(_name, _thick, _round), render_fn=_make_render(_kind), status="faithful",
        notes=_NOTES_COMMON + _ENV_NOTE + (_TRUCHET_NOTE if _name.startswith("Truchet") else ""),
        description=f"Ray-marched 3D {_title.lower()} with metallic/glass shading (ImageToolbox {_name}).",
    ))
