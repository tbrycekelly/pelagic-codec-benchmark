from .base import Codec, Treatment
from .extended import AvifCodec, Jpeg2000Codec, JpegLsCodec, JpegXlCodec, TiffCodec, WebpCodec
from .jpeg import JpegCodec
from .png import PngCodec


def codec_for(name: str) -> Codec:
    codecs = {
        "png": PngCodec(), "tiff": TiffCodec(), "jpeg": JpegCodec(),
        "jpeg2000": Jpeg2000Codec(), "jpegxl": JpegXlCodec(),
        "webp": WebpCodec(), "avif": AvifCodec(), "jpegls": JpegLsCodec(),
    }
    try:
        return codecs[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported codec {name!r}; supported codecs are png, tiff, jpeg, jpeg2000, jpegxl, webp, avif, and jpegls.") from exc


__all__ = ["Codec", "Treatment", "codec_for"]
