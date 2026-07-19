"""Background-override rendering for Roborock maps.

Everything in this module is synchronous and CPU/IO bound — callers must run
it via hass.async_add_executor_job.

The override composite re-parses the vacuum's raw map bytes with custom
palettes at the same scale the core roborock integration uses, so the canvas
dimensions (and therefore the calibration points exposed for the map card)
are identical to the unmodified render.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import logging
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageOps
from vacuum_map_parser_base.config.color import Color, ColorsPalette, SupportedColor
from vacuum_map_parser_base.config.drawable import Drawable
from vacuum_map_parser_base.config.image_config import ImageConfig
from vacuum_map_parser_base.config.size import Size, Sizes
from vacuum_map_parser_base.map_data import ImageDimensions, Point
from vacuum_map_parser_roborock.map_data_parser import RoborockMapDataParser

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import STORAGE_DIR

from .const import (
    DOMAIN,
    MAX_BG_DIMENSION,
    MAX_COMPOSITE_DIMENSION,
    MAX_SCALED_DIMENSION,
    MAX_UPLOAD_BYTES,
)

_LOGGER = logging.getLogger(__name__)

_TRANSPARENT: Color = (0, 0, 0, 0)
_OPAQUE_WHITE: Color = (255, 255, 255, 255)

_BASE_COLORS = (
    SupportedColor.MAP_OUTSIDE,
    SupportedColor.MAP_INSIDE,
    SupportedColor.MAP_WALL,
    SupportedColor.MAP_WALL_V2,
    SupportedColor.GREY_WALL,
    SupportedColor.SCAN,
    SupportedColor.CARPETS,
    SupportedColor.UNKNOWN,
    SupportedColor.NEW_DISCOVERED_AREA,
)

_WALL_COLORS = (
    SupportedColor.MAP_WALL,
    SupportedColor.MAP_WALL_V2,
    SupportedColor.GREY_WALL,
)

ALLOWED_FORMATS = {"png": "png", "jpeg": "jpg", "webp": "webp"}


class InvalidBackgroundImage(Exception):
    """Uploaded background failed validation; error_key maps to a flow error."""

    def __init__(self, error_key: str) -> None:
        super().__init__(error_key)
        self.error_key = error_key


def _transparent_room_colors() -> dict[str, Color]:
    return dict.fromkeys(ColorsPalette.ROOM_COLORS, _TRANSPARENT)


def _overlay_palette() -> ColorsPalette:
    """Palette rendering only drawables: all base pixels transparent."""
    return ColorsPalette(
        dict.fromkeys(_BASE_COLORS, _TRANSPARENT), _transparent_room_colors()
    )


def _mask_palette() -> ColorsPalette:
    """Palette rendering only physical walls (and the charger) opaque."""
    colors: dict[SupportedColor, Color] = dict.fromkeys(_BASE_COLORS, _TRANSPARENT)
    for wall_color in _WALL_COLORS:
        colors[wall_color] = _OPAQUE_WHITE
    colors[SupportedColor.CHARGER] = _OPAQUE_WHITE
    colors[SupportedColor.CHARGER_OUTLINE] = _OPAQUE_WHITE
    return ColorsPalette(colors, _transparent_room_colors())


def _create_parser(
    palette: ColorsPalette, drawables: list[Drawable], map_scale: int
) -> RoborockMapDataParser:
    return RoborockMapDataParser(
        palette,
        Sizes(
            {
                k: v * map_scale
                for k, v in Sizes.SIZES.items()
                if k != Size.MOP_PATH_WIDTH
            }
        ),
        drawables,
        ImageConfig(scale=map_scale),
        [],
    )


def build_overlay(raw: bytes, drawables: list[Drawable], map_scale: int) -> Image.Image:
    """Parse the raw map into the RGBA layer containing only the drawables.

    Raises ValueError/IndexError when the raw payload cannot be parsed
    (same failure modes as python-roborock's own MapParser).
    """
    overlay_map = _create_parser(_overlay_palette(), drawables, map_scale).parse(raw)
    if overlay_map.image is None:
        raise ValueError("Raw map data could not be rendered")
    return overlay_map.image.data.convert("RGBA")


def build_wall_mask(raw: bytes, map_scale: int) -> Image.Image:
    """Parse the raw map into a single-band alpha mask of walls + charger.

    Only needed while a tuning preview inverts the wall pixels, so callers
    build it lazily instead of paying a second full parse on every map update.
    """
    mask_map = _create_parser(_mask_palette(), [Drawable.CHARGER], map_scale).parse(raw)
    if mask_map.image is None:
        raise ValueError("Raw map data could not be rendered")
    return mask_map.image.data.convert("RGBA").getchannel("A")


@lru_cache(maxsize=4)
def _scaled_background(bg_bytes: bytes, scale_x: float, scale_y: float) -> Image.Image:
    """Decode and scale the background image (cached; treat result read-only).

    Each axis scale is capped so the scaled edge length never exceeds
    MAX_SCALED_DIMENSION, protecting against runaway PIL allocations.
    """
    bg = Image.open(io.BytesIO(bg_bytes))
    bg = ImageOps.exif_transpose(bg).convert("RGBA")
    scale_x = min(scale_x, MAX_SCALED_DIMENSION / bg.width)
    scale_y = min(scale_y, MAX_SCALED_DIMENSION / bg.height)
    if scale_x != 1 or scale_y != 1:
        bg = bg.resize(
            (max(1, round(bg.width * scale_x)), max(1, round(bg.height * scale_y))),
            Image.Resampling.LANCZOS,
        )
    return bg


def clear_scaled_backgrounds() -> None:
    """Drop cached decoded backgrounds (call on unload and when flows end)."""
    _scaled_background.cache_clear()


_ROTATE_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


@dataclass(frozen=True)
class MapTransform:
    """Affine map between vacuum-world mm and base (unrotated) render pixels.

    The forward direction delegates to the parser's own ImageDimensions.to_img
    (the transform MapData.calibration() uses), so override placement can never
    drift from the calibration points. Only the translation moves as the map's
    crop box changes, which is why an override anchored in vacuum coordinates
    survives the base map being re-cropped.
    """

    dims: ImageDimensions

    @classmethod
    def from_dimensions(cls, dims: ImageDimensions) -> MapTransform:
        return cls(dims)

    @property
    def base_size(self) -> tuple[int, int]:
        """Unrotated base render size in px (equals the parsed overlay size)."""
        return (
            int(self.dims.width * self.dims.scale),
            int(self.dims.height * self.dims.scale),
        )

    def vacuum_to_base(self, wx: float, wy: float) -> tuple[float, float]:
        """Vacuum-world mm -> base (unrotated) render pixel."""
        p = self.dims.to_img(Point(wx, wy))
        return (p.x, p.y)

    def base_to_vacuum(self, px: float, py: float) -> tuple[float, float]:
        """Base (unrotated) render pixel -> vacuum-world mm.

        Exact inverse of vacuum_to_base, assuming the Roborock parser's mm/50
        image transformation.
        """
        dims = self.dims
        return (
            (px / dims.scale + dims.left) * 50,
            (dims.top + dims.height - 1 - py / dims.scale) * 50,
        )


def rotate_point(
    x: float, y: float, w: int, h: int, rotation: int
) -> tuple[float, float]:
    """Unrotated base pixel (frame w x h) -> rotated frame. CCW, matching the
    PIL transpose used in compose_final and the plain-map rotation."""
    if rotation == 90:
        return (y, w - x)
    if rotation == 180:
        return (w - x, h - y)
    if rotation == 270:
        return (h - y, x)
    return (x, y)


def unrotate_point(
    x: float, y: float, w: int, h: int, rotation: int
) -> tuple[float, float]:
    """Inverse of rotate_point; (w, h) is the UNROTATED base size (not swapped)."""
    if rotation == 90:
        return (w - y, x)
    if rotation == 180:
        return (w - x, h - y)
    if rotation == 270:
        return (y, h - x)
    return (x, y)


def anchor_topleft(
    transform: MapTransform, anchor_vx: float, anchor_vy: float, rotation: int
) -> tuple[float, float]:
    """Rotated-frame top-left px of an override anchored in vacuum-world mm.

    Single source of the anchor geometry, shared by the image entity's render
    and calibration and by the options flow's form defaults, so they cannot
    disagree.
    """
    bx, by = transform.vacuum_to_base(anchor_vx, anchor_vy)
    w_px, h_px = transform.base_size
    return rotate_point(bx, by, w_px, h_px, rotation)


def canvas_origin(offset_x: float, offset_y: float) -> tuple[int, int]:
    """Top-left of the union canvas (<= 0 per axis). Depends only on the override
    offset, never the background size, so calibration can compute the identical
    origin without decoding the image. floor() so a fractional-negative offset
    does not clip the override's top/left edge."""
    return (math.floor(min(0.0, offset_x)), math.floor(min(0.0, offset_y)))


def union_canvas(
    rot_w: int,
    rot_h: int,
    offset_x: float,
    offset_y: float,
    ovr_w: int,
    ovr_h: int,
) -> tuple[int, int, int, int]:
    """Union bbox of the rotated base extent [0, rot] and the override
    [offset, offset + ovr]. Returns (canvas_w, canvas_h, origin_x, origin_y).

    The canvas dimensions are capped at MAX_COMPOSITE_DIMENSION while the
    origin is preserved, so content may be cropped at the right/bottom but the
    origin-based calibration shift stays valid for every rendered pixel.
    """
    origin_x, origin_y = canvas_origin(offset_x, offset_y)
    canvas_w = math.ceil(max(rot_w, offset_x + ovr_w)) - origin_x
    canvas_h = math.ceil(max(rot_h, offset_y + ovr_h)) - origin_y
    return (
        min(canvas_w, MAX_COMPOSITE_DIMENSION),
        min(canvas_h, MAX_COMPOSITE_DIMENSION),
        origin_x,
        origin_y,
    )


def compose_final(
    overlay: Image.Image,
    mask: Image.Image | None,
    bg_bytes: bytes,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
    rotation: int,
) -> bytes:
    """Composite the background under the drawables and encode as PNG.

    offset_x/offset_y are the scaled override's top-left in the ROTATED frame.
    The canvas is the union bounding box of the rotated base-map extent and the
    override extent; areas covered by neither are transparent. With a wall mask
    (adjust preview) the wall/dock pixels invert whatever lies beneath them,
    which also draws white outlines wherever they fall over transparent areas.
    """
    if rotation in _ROTATE_TRANSPOSE:
        overlay = overlay.transpose(_ROTATE_TRANSPOSE[rotation])
        if mask is not None:
            mask = mask.transpose(_ROTATE_TRANSPOSE[rotation])
    rot_w, rot_h = overlay.size

    bg = _scaled_background(bg_bytes, scale_x, scale_y)
    ovr_w, ovr_h = bg.size

    canvas_w, canvas_h, origin_x, origin_y = union_canvas(
        rot_w, rot_h, offset_x, offset_y, ovr_w, ovr_h
    )

    canvas = Image.new("RGBA", (canvas_w, canvas_h), _TRANSPARENT)
    canvas.paste(bg, (round(offset_x - origin_x), round(offset_y - origin_y)), bg)

    base_dest = (-origin_x, -origin_y)
    if base_dest[0] < canvas_w and base_dest[1] < canvas_h:
        canvas.alpha_composite(overlay, dest=base_dest)
        if mask is not None:
            full_mask = Image.new("L", (canvas_w, canvas_h), 0)
            full_mask.paste(mask, base_dest)
            inverted = ImageChops.invert(canvas.convert("RGB")).convert("RGBA")
            canvas = Image.composite(inverted, canvas, full_mask)

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def validate_upload(data: bytes) -> str:
    """Validate an uploaded background image and return its file extension."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidBackgroundImage("file_too_large")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").lower()
            width, height = image.size
            image.verify()
    except Image.DecompressionBombError as err:
        raise InvalidBackgroundImage("image_too_large") from err
    except Exception as err:
        raise InvalidBackgroundImage("invalid_image") from err
    if image_format not in ALLOWED_FORMATS:
        raise InvalidBackgroundImage("invalid_image")
    if width > MAX_BG_DIMENSION or height > MAX_BG_DIMENSION:
        raise InvalidBackgroundImage("image_too_large")
    return ALLOWED_FORMATS[image_format]


def override_dir(hass: HomeAssistant) -> Path:
    """Directory where uploaded background files are stored."""
    return Path(hass.config.path(STORAGE_DIR, DOMAIN))


def save_override_file(hass: HomeAssistant, filename: str, data: bytes) -> None:
    """Persist an uploaded background image."""
    directory = override_dir(hass)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_bytes(data)


def read_override_file(hass: HomeAssistant, filename: str) -> bytes:
    """Read a stored background image (raises OSError when missing)."""
    return (override_dir(hass) / filename).read_bytes()


def delete_override_files_for_key(hass: HomeAssistant, rotation_key: str) -> None:
    """Delete stored background files for a map (any extension)."""
    directory = override_dir(hass)
    if not directory.is_dir():
        return
    for path in directory.glob(f"{rotation_key}.*"):
        path.unlink(missing_ok=True)
