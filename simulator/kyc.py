"""Synthetic KYC document image rendering for the MuleNet simulator.

The images are deliberately crude: flat coloured rectangles with drawn
text and a large "SYNTHETIC SAMPLE" watermark. They exist so ``docai/``
has something to run Document AI against and so document *reuse* across
banks is detectable.

Hard rules (see CLAUDE.md and the data-engineer brief):

* No real people's likenesses. The "photo" is a flat grey rectangle with
  the words ``NO PHOTO`` on it.
* No real ID number formats and no valid checksums. Every rendered number
  uses the obviously-fake ``SYN-xxxx-xxxx-xxxx`` shape derived from an
  HMAC digest.
* No real names. Holders are ``SYNTHETIC HOLDER <hex>``.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ID_CARD_SIZE = (640, 400)
ADDRESS_PROOF_SIZE = (640, 520)

WATERMARK = "SYNTHETIC SAMPLE - NOT A REAL DOCUMENT"

_ACCENTS: tuple[tuple[int, int, int], ...] = (
    (36, 78, 140),
    (140, 42, 58),
    (32, 110, 88),
    (108, 64, 138),
    (168, 104, 24),
    (48, 96, 112),
)


def _font(size: int) -> ImageFont.ImageFont:
    """Pillow's built-in bitmap font, scaled when the runtime supports it."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1
        return ImageFont.load_default()


def accent_for(key: str) -> tuple[int, int, int]:
    """Pick a deterministic accent colour from a hashed key."""
    return _ACCENTS[int(key[:8], 16) % len(_ACCENTS)]


def display_id_number(hashed_id_number: str) -> str:
    """Render a hashed id as an obviously-fake ``SYN-xxxx-xxxx-xxxx`` string.

    This is *not* a real national-id format and carries no checksum; it is a
    human-readable projection of the HMAC digest so that two accounts sharing
    an identity also share a visible number on their document images.
    """
    h = hashed_id_number[:12].upper()
    return f"SYN-{h[0:4]}-{h[4:8]}-{h[8:12]}"


def holder_label(identity_key: str) -> str:
    """A clearly synthetic holder name derived from a hashed identity key."""
    return f"SYNTHETIC HOLDER {identity_key[:6].upper()}"


def _watermark(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    draw.rectangle([0, height - 34, width, height], fill=(235, 225, 120))
    draw.text((14, height - 25), WATERMARK, fill=(90, 70, 0), font=_font(15))


def render_id_card(
    path: Path,
    *,
    holder_name: str,
    id_number_display: str,
    address: str,
    issuer: str,
    accent: tuple[int, int, int],
) -> Path:
    """Draw a placeholder ID card and save it as PNG at ``path``."""
    img = Image.new("RGB", ID_CARD_SIZE, (247, 247, 250))
    draw = ImageDraw.Draw(img)
    width, _ = ID_CARD_SIZE

    draw.rectangle([0, 0, width, 70], fill=accent)
    draw.text((18, 14), "SPECIMEN IDENTITY CARD", fill=(255, 255, 255), font=_font(24))
    draw.text((18, 45), f"ISSUER: {issuer} (FICTIONAL)", fill=(230, 230, 240), font=_font(14))

    # Deliberately not a likeness: a flat grey block labelled NO PHOTO.
    draw.rectangle([18, 96, 178, 296], fill=(198, 198, 204), outline=accent, width=3)
    draw.text((52, 182), "NO PHOTO", fill=(88, 88, 96), font=_font(20))
    draw.text((40, 210), "SYNTHETIC RECORD", fill=(110, 110, 118), font=_font(12))

    lines = [
        ("NAME", holder_name),
        ("ID NUMBER", id_number_display),
        ("ADDRESS", address),
        ("TYPE", "NON-TRANSACTABLE SAMPLE"),
    ]
    y = 104
    for label, value in lines:
        draw.text((200, y), label, fill=(120, 120, 130), font=_font(13))
        draw.text((200, y + 18), value, fill=(20, 20, 28), font=_font(18))
        y += 50

    draw.rectangle([18, 96, width - 18, 316], outline=accent, width=2)
    _watermark(draw, ID_CARD_SIZE)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def render_address_proof(
    path: Path,
    *,
    holder_name: str,
    address: str,
    issuer: str,
    statement_period: str,
    accent: tuple[int, int, int],
) -> Path:
    """Draw a placeholder utility-style address proof and save it as PNG."""
    img = Image.new("RGB", ADDRESS_PROOF_SIZE, (252, 252, 252))
    draw = ImageDraw.Draw(img)
    width, _ = ADDRESS_PROOF_SIZE

    draw.rectangle([0, 0, width, 84], fill=accent)
    draw.text((18, 16), "FICTIONAL UTILITIES CO.", fill=(255, 255, 255), font=_font(24))
    draw.text((18, 52), "PROOF OF ADDRESS - SAMPLE ONLY", fill=(230, 230, 240), font=_font(14))

    rows = [
        ("ACCOUNT HOLDER", holder_name),
        ("SERVICE ADDRESS", address),
        ("STATEMENT PERIOD", statement_period),
        ("PRESENTED TO", f"{issuer} (FICTIONAL)"),
        ("AMOUNT DUE", "0.00 (SAMPLE)"),
    ]
    y = 112
    for label, value in rows:
        draw.text((24, y), label, fill=(120, 120, 130), font=_font(13))
        draw.text((24, y + 18), value, fill=(20, 20, 28), font=_font(18))
        draw.line([24, y + 44, width - 24, y + 44], fill=(222, 222, 228), width=1)
        y += 62

    draw.rectangle([24, 440, width - 24, 470], fill=(238, 240, 246))
    draw.text(
        (32, 448),
        "NO PERSONAL DATA - GENERATED FOR TESTING",
        fill=(90, 90, 104),
        font=_font(14),
    )
    _watermark(draw, ADDRESS_PROOF_SIZE)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path
