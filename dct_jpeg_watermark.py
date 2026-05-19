"""
Blind binary-image watermarking in the DCT domain with JPEG-like quantization.

This version embeds the watermark using parity/QIM on one DCT coefficient:

    block -> level shift -> FDCT -> parity coefficient edit -> quantization
    -> dequantization -> IDCT -> inverse level shift

Extraction is blind: it only needs the watermarked image and the watermark
shape. The original host image is not required.

Only numpy and OpenCV are used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


HOST_IMAGE_PATH = "input.jpg"
WATERMARK_IMAGE_PATH = "watermark.png"
OUTPUT_DIR = "img_output"
WATERMARKED_IMAGE_PATH = f"{OUTPUT_DIR}/watermarked.jpg"
EXTRACTED_WATERMARK_PATH = f"{OUTPUT_DIR}/extracted_watermark.png"

DEFAULT_EMBEDDING_QF = 90
DEFAULT_RECOMPRESS_QF = 30
DEFAULT_COEFFICIENT = (4, 4)


def ensure_parent_dir(output_path: str) -> None:
    """Create the output parent directory if the path has one."""
    parent = Path(output_path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class WatermarkConfig:
    """Configuration for DCT-domain binary image watermarking."""

    quality_factor: int = 90
    coefficient: Tuple[int, int] = DEFAULT_COEFFICIENT
    use_y_channel: bool = True


class DCTJPEGWatermarker:
    """Embed and extract a binary image watermark in JPEG-style DCT blocks."""

    # Standard JPEG luminance quantization matrix.
    #
    # Low-frequency entries are small because human vision is sensitive to slow
    # brightness changes. High-frequency entries are larger because fine detail
    # can be compressed more aggressively with less visible damage.
    STANDARD_LUMINANCE_Q = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=np.float32,
    )

    def __init__(self, config: WatermarkConfig | None = None) -> None:
        self.config = config or WatermarkConfig()
        self._validate_config()
        self.quantization_matrix = self.generate_luminance_quantization_matrix(
            self.config.quality_factor
        )

    def _validate_config(self) -> None:
        if not 1 <= self.config.quality_factor <= 100:
            raise ValueError("quality_factor must be in the range 1..100.")

        u, v = self.config.coefficient
        if not (0 <= u < 8 and 0 <= v < 8):
            raise ValueError("coefficient must be inside an 8x8 block.")

        if (u, v) == (0, 0):
            raise ValueError(
                "Do not embed in coefficient (0, 0); the DC term controls block brightness."
            )

    @classmethod
    def generate_luminance_quantization_matrix(cls, quality_factor: int) -> np.ndarray:
        """
        Generate the scaled JPEG luminance quantization matrix.

        The JPEG quality scaling formula is intentionally non-linear:
        - QF < 50  -> scale = 5000 / QF, so very low quality grows rapidly.
        - QF >= 50 -> scale = 200 - 2*QF, so quality 100 approaches no loss.

        Quantization divides each DCT coefficient by this matrix and rounds the
        result. Larger matrix values mean coarser bins and stronger loss.
        """
        if not 1 <= quality_factor <= 100:
            raise ValueError("quality_factor must be in the range 1..100.")

        if quality_factor < 50:
            scale = 5000 / quality_factor
        else:
            scale = 200 - (2 * quality_factor)

        scaled = np.floor((cls.STANDARD_LUMINANCE_Q * scale + 50) / 100)
        return np.clip(scaled, 1, 255).astype(np.float32)

    @staticmethod
    def _crop_to_8x8(image: np.ndarray) -> np.ndarray:
        """Crop dimensions so every block is exactly 8x8 pixels."""
        height, width = image.shape[:2]
        cropped_height = height - (height % 8)
        cropped_width = width - (width % 8)

        if cropped_height == 0 or cropped_width == 0:
            raise ValueError("Host image must be at least 8x8 pixels.")

        return image[:cropped_height, :cropped_width].copy()

    def _prepare_luminance(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Extract the grayscale/Y plane used for watermarking.

        For color images, the watermark is inserted into Y from YCrCb. JPEG also
        prioritizes luminance this way, and keeping Cr/Cb unchanged helps reduce
        visible color artifacts.
        """
        image = self._crop_to_8x8(image)

        if image.ndim == 2:
            return image.astype(np.uint8), None

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Host image must be grayscale or BGR color.")

        if not self.config.use_y_channel:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray.astype(np.uint8), None

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 0].astype(np.uint8), ycrcb[:, :, 1:].copy()

    @staticmethod
    def _restore_image(luminance: np.ndarray, chroma: np.ndarray | None) -> np.ndarray:
        """Restore grayscale output or combine edited Y with the original chroma."""
        if chroma is None:
            return luminance

        ycrcb = np.dstack((luminance, chroma)).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _prepare_binary_watermark(watermark: np.ndarray) -> np.ndarray:
        """
        Convert any grayscale/BGR watermark image into a 0/1 binary image.

        Otsu thresholding is used so black-white logos, text masks, and simple
        binary images are normalized consistently.
        """
        if watermark.ndim == 3:
            watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
        elif watermark.ndim != 2:
            raise ValueError("Watermark must be a grayscale or BGR image.")

        _, binary = cv2.threshold(
            watermark.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary.astype(np.uint8)

    @staticmethod
    def _capacity_bits(host_shape: tuple[int, int]) -> int:
        """One watermark bit is embedded in each non-overlapping 8x8 host block."""
        height, width = host_shape
        return (height // 8) * (width // 8)

    def _validate_watermark_capacity(
        self, luminance_shape: tuple[int, int], watermark_shape: tuple[int, int]
    ) -> None:
        capacity = self._capacity_bits(luminance_shape)
        required = watermark_shape[0] * watermark_shape[1]

        if required > capacity:
            max_side_hint = int(np.floor(np.sqrt(capacity)))
            raise ValueError(
                "Watermark image exceeds host capacity: "
                f"{required} bits required, {capacity} bits available. "
                f"Use a watermark around {max_side_hint}x{max_side_hint} or smaller."
            )

    def _fdct(self, block: np.ndarray) -> np.ndarray:
        """
        Level shift and apply forward DCT.

        JPEG subtracts 128 before DCT so unsigned pixel values [0,255] become
        centered around zero. This makes the DC term represent average brightness
        cleanly and packs most natural-image energy into low frequencies.
        """
        shifted = block.astype(np.float32) - 128.0
        return cv2.dct(shifted)

    def _quantize(self, dct_block: np.ndarray) -> np.ndarray:
        """Quantize DCT coefficients using the configured JPEG luminance matrix."""
        return np.round(dct_block / self.quantization_matrix)

    @staticmethod
    def _parity(value: int) -> int:
        """Return parity for positive or negative integers as 0/even or 1/odd."""
        return abs(value) % 2

    @staticmethod
    def _nearest_index_with_parity(current_index: int, target_bit: int) -> int:
        """
        Find the nearest quantized index whose parity represents target_bit.

        Blind embedding uses QIM/parity:
        - even quantized coefficient index -> bit 0
        - odd quantized coefficient index  -> bit 1

        If the current index already has the desired parity, it is kept. If not,
        the index is moved to the nearest neighboring bin. This makes extraction
        possible from the watermarked image alone because the bit is encoded in
        the coefficient itself, not in a difference from the original image.
        """
        if DCTJPEGWatermarker._parity(current_index) == target_bit:
            return current_index

        plus = current_index + 1
        minus = current_index - 1

        if abs(plus) <= abs(minus):
            return plus
        return minus

    def _embed_bit_by_parity(
        self, dct_block: np.ndarray, bit: int, coefficient: Tuple[int, int]
    ) -> np.ndarray:
        """
        Embed one bit by forcing the JPEG-quantized index parity at coefficient.

        The edited value is placed at the center of the desired JPEG quantization
        bin. After the following quantization step, the selected coefficient
        should round to an even index for bit 0 or an odd index for bit 1.
        """
        u, v = coefficient
        quant_step = self.quantization_matrix[u, v]
        current_index = int(np.round(dct_block[u, v] / quant_step))
        target_index = self._nearest_index_with_parity(current_index, int(bit))

        edited = dct_block.copy()
        edited[u, v] = target_index * quant_step
        return edited

    def _jpeg_reconstruct(self, dct_block: np.ndarray) -> np.ndarray:
        """
        Simulate JPEG quantization and decoding for one DCT block.

        The edited coefficient is quantized and dequantized before IDCT to
        simulate the relevant lossy JPEG stage.
        """
        quantized = self._quantize(dct_block)
        dequantized = quantized * self.quantization_matrix
        spatial = cv2.idct(dequantized.astype(np.float32)) + 128.0
        return np.clip(np.round(spatial), 0, 255).astype(np.uint8)

    def embed_binary_watermark(
        self, host_image: np.ndarray, watermark_image: np.ndarray
    ) -> np.ndarray:
        """
        Embed a binary image watermark into the host image.

        Coefficient (4,4) is a mid-frequency position. The bit is stored as
        parity of the JPEG-quantized coefficient index:
        - even index -> bit 0
        - odd index  -> bit 1

        This makes extraction blind because the original image is not needed.
        """
        luminance, chroma = self._prepare_luminance(host_image)
        watermark = self._prepare_binary_watermark(watermark_image)
        self._validate_watermark_capacity(luminance.shape, watermark.shape)

        watermarked_y = np.empty_like(luminance, dtype=np.uint8)
        watermark_bits = watermark.flatten()
        total_bits = watermark_bits.size
        bit_index = 0
        u, v = self.config.coefficient

        for row in range(0, luminance.shape[0], 8):
            for col in range(0, luminance.shape[1], 8):
                block = luminance[row : row + 8, col : col + 8]
                dct_block = self._fdct(block)

                if bit_index < total_bits:
                    dct_block = self._embed_bit_by_parity(
                        dct_block, int(watermark_bits[bit_index]), (u, v)
                    )
                    bit_index += 1

                watermarked_y[row : row + 8, col : col + 8] = self._jpeg_reconstruct(
                    dct_block
                )

        return self._restore_image(watermarked_y, chroma)

    def extract_binary_watermark(
        self,
        watermarked_image: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Extract a binary watermark image using blind extraction.

        Only the watermarked image is transformed and quantized. The bit is read
        from the parity of the selected quantized DCT coefficient:
        - even coefficient index -> bit 0
        - odd coefficient index  -> bit 1
        """
        watermarked_y, _ = self._prepare_luminance(watermarked_image)
        self._validate_watermark_capacity(watermarked_y.shape, watermark_shape)

        required_bits = watermark_shape[0] * watermark_shape[1]
        extracted_bits: list[int] = []
        u, v = self.config.coefficient

        for row in range(0, watermarked_y.shape[0], 8):
            for col in range(0, watermarked_y.shape[1], 8):
                watermarked_block = watermarked_y[row : row + 8, col : col + 8]

                watermarked_q = self._quantize(self._fdct(watermarked_block))
                coefficient_index = int(watermarked_q[u, v])

                extracted_bits.append(self._parity(coefficient_index))
                if len(extracted_bits) == required_bits:
                    extracted = np.array(extracted_bits, dtype=np.uint8)
                    return (extracted.reshape(watermark_shape) * 255).astype(np.uint8)

        raise RuntimeError("Extraction stopped unexpectedly before all bits were read.")

    def embed_file(
        self, host_path: str, watermark_path: str, output_path: str
    ) -> np.ndarray:
        """Load host + binary watermark images, embed, save, and return output."""
        host = cv2.imread(host_path, cv2.IMREAD_COLOR)
        if host is None:
            raise FileNotFoundError(f"Could not read host image: {host_path}")

        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        if watermark is None:
            raise FileNotFoundError(f"Could not read watermark image: {watermark_path}")

        watermarked = self.embed_binary_watermark(host, watermark)
        write_params = []
        if output_path.lower().endswith((".jpg", ".jpeg")):
            write_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.quality_factor]

        ensure_parent_dir(output_path)
        if not cv2.imwrite(output_path, watermarked, write_params):
            raise IOError(f"Could not write watermarked image: {output_path}")

        return watermarked

    def extract_file(
        self,
        watermarked_path: str,
        watermark_shape: Tuple[int, int],
        output_path: str,
    ) -> np.ndarray:
        """Extract a binary watermark from one watermarked file and save it."""
        watermarked = cv2.imread(watermarked_path, cv2.IMREAD_COLOR)
        if watermarked is None:
            raise FileNotFoundError(f"Could not read watermarked image: {watermarked_path}")

        extracted = self.extract_binary_watermark(
            watermarked, watermark_shape=watermark_shape
        )
        ensure_parent_dir(output_path)
        if not cv2.imwrite(output_path, extracted):
            raise IOError(f"Could not write extracted watermark image: {output_path}")

        return extracted

    def recompress_file(
        self, input_path: str, output_path: str, recompress_qf: int
    ) -> np.ndarray:
        """
        Recompress an existing watermarked image with a chosen JPEG recompress QF.

        This is separate from embed_file() so embedding and compression testing
        can be called independently during manual demos.
        """
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image to recompress: {input_path}")

        recompressed = self.jpeg_recompress_image(image, recompress_qf)
        ensure_parent_dir(output_path)
        if not cv2.imwrite(
            output_path,
            recompressed,
            [cv2.IMWRITE_JPEG_QUALITY, recompress_qf],
        ):
            raise IOError(f"Could not write recompressed image: {output_path}")

        return recompressed

    @staticmethod
    def jpeg_recompress_image(image: np.ndarray, quality_factor: int) -> np.ndarray:
        """
        Simulate external JPEG recompression.

        This QF is intentionally separate from the embedding QF. To demonstrate
        watermark failure, embed once at a stable QF, then recompress the saved
        watermarked image using lower and lower recompress QF values.
        """
        if not 1 <= quality_factor <= 100:
            raise ValueError("recompress quality_factor must be in the range 1..100.")

        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality_factor]
        )
        if not ok:
            raise IOError("Could not encode image during JPEG recompression.")

        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if recompressed is None:
            raise IOError("Could not decode image during JPEG recompression.")

        return recompressed

    @staticmethod
    def binary_watermark_accuracy(
        reference_watermark: np.ndarray, extracted_watermark: np.ndarray
    ) -> float:
        """Return pixel accuracy between the original binary watermark and extraction."""
        reference = DCTJPEGWatermarker._prepare_binary_watermark(reference_watermark) * 255
        extracted = DCTJPEGWatermarker._prepare_binary_watermark(extracted_watermark) * 255

        if reference.shape != extracted.shape:
            raise ValueError(
                f"Watermark shapes differ: {reference.shape} vs {extracted.shape}."
            )

        return float(np.mean(reference == extracted))

    def sweep_recompression_quality(
        self,
        watermarked_image: np.ndarray,
        reference_watermark: np.ndarray,
        min_recompress_qf: int = 1,
        max_recompress_qf: int = 100,
        failure_threshold: float = 0.75,
    ) -> list[tuple[int, float]]:
        """
        Test extraction after JPEG recompression across QF values.

        Returns a list of (recompress_qf, extraction_accuracy). A lower accuracy means
        more watermark bits changed. The first QF with accuracy below
        failure_threshold can be reported as the point where extraction fails.
        """
        if not 1 <= min_recompress_qf <= max_recompress_qf <= 100:
            raise ValueError("Recompress QF range must satisfy 1 <= min <= max <= 100.")

        binary_reference = self._prepare_binary_watermark(reference_watermark) * 255
        results: list[tuple[int, float]] = []

        for recompress_qf in range(max_recompress_qf, min_recompress_qf - 1, -1):
            recompressed = self.jpeg_recompress_image(watermarked_image, recompress_qf)
            extracted = self.extract_binary_watermark(recompressed, binary_reference.shape)
            accuracy = self.binary_watermark_accuracy(binary_reference, extracted)
            results.append((recompress_qf, accuracy))

        return results


def create_watermarker(embedding_qf: int) -> DCTJPEGWatermarker:
    """Create a watermarker configured with the embedding/extraction QF."""
    return DCTJPEGWatermarker(
        WatermarkConfig(
            quality_factor=embedding_qf,
            coefficient=DEFAULT_COEFFICIENT,
            use_y_channel=True,
        )
    )


def load_watermark_for_shape(watermark_image_path: str) -> np.ndarray:
    """Load the original watermark only to get shape and measure accuracy."""
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"Could not read watermark image: {watermark_image_path}")
    return watermark


def recompressed_output_paths(recompress_qf: int) -> tuple[str, str]:
    """Return filenames for a manually chosen recompression QF."""
    return (
        f"{OUTPUT_DIR}/watermarked_recompress_qf{recompress_qf}.jpg",
        f"{OUTPUT_DIR}/extracted_watermark_recompress_qf{recompress_qf}.png",
    )


def run_embed_watermark(
    host_image_path: str = HOST_IMAGE_PATH,
    watermark_image_path: str = WATERMARK_IMAGE_PATH,
    watermarked_image_path: str = WATERMARKED_IMAGE_PATH,
    embedding_qf: int = DEFAULT_EMBEDDING_QF,
) -> None:
    """Create only the watermarked image from input.jpg and watermark.png."""
    watermarker = create_watermarker(embedding_qf)
    watermarker.embed_file(host_image_path, watermark_image_path, watermarked_image_path)

    print("Embedding finished.")
    print(f"Embedding QF: {embedding_qf}")
    print(f"Watermarked image saved to: {watermarked_image_path}")


def run_compress_recompression(
    watermarked_image_path: str = WATERMARKED_IMAGE_PATH,
    watermark_image_path: str = WATERMARK_IMAGE_PATH,
    embedding_qf: int = DEFAULT_EMBEDDING_QF,
    recompress_qf: int = DEFAULT_RECOMPRESS_QF,
) -> None:
    """Recompress the default watermarked output, then extract the watermark."""
    watermarker = create_watermarker(embedding_qf)
    watermark = load_watermark_for_shape(watermark_image_path)
    recompressed_watermarked_path, recompressed_extracted_watermark_path = (
        recompressed_output_paths(recompress_qf)
    )

    recompressed = watermarker.recompress_file(
        watermarked_image_path, recompressed_watermarked_path, recompress_qf
    )
    recompressed_extracted = watermarker.extract_binary_watermark(
        recompressed, watermark.shape
    )
    accuracy = watermarker.binary_watermark_accuracy(watermark, recompressed_extracted)

    ensure_parent_dir(recompressed_extracted_watermark_path)
    if not cv2.imwrite(recompressed_extracted_watermark_path, recompressed_extracted):
        raise IOError(
            f"Could not write recompressed extracted watermark: "
            f"{recompressed_extracted_watermark_path}"
        )

    print("Recompression finished.")
    print(f"Embedding QF used by extractor: {embedding_qf}")
    print(f"Recompress QF: {recompress_qf}")
    print(f"Extraction accuracy: {accuracy:.4f}")
    print(f"Recompressed watermarked image saved to: {recompressed_watermarked_path}")
    print(
        "Recompressed extracted watermark saved to: "
        f"{recompressed_extracted_watermark_path}"
    )


def run_extract_watermark(
    watermarked_image_path: str = WATERMARKED_IMAGE_PATH,
    watermark_image_path: str = WATERMARK_IMAGE_PATH,
    extracted_watermark_path: str = EXTRACTED_WATERMARK_PATH,
    embedding_qf: int = DEFAULT_EMBEDDING_QF,
) -> None:
    """Run only blind extraction from an existing watermarked image."""
    watermarker = create_watermarker(embedding_qf)
    watermark = load_watermark_for_shape(watermark_image_path)

    extracted = watermarker.extract_file(
        watermarked_image_path,
        watermark_shape=watermark.shape,
        output_path=extracted_watermark_path,
    )
    accuracy = watermarker.binary_watermark_accuracy(watermark, extracted)

    print("Extraction finished.")
    print(f"Embedding QF used by extractor: {embedding_qf}")
    print(f"Extracted watermark saved to: {extracted_watermark_path}")
    print(f"Extraction accuracy: {accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line mode and file/QF options."""
    parser = argparse.ArgumentParser(
        description="Blind binary watermarking using DCT parity/QIM."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    embed_parser = subparsers.add_parser(
        "embed", help="embed watermark.png into input.jpg"
    )
    embed_parser.add_argument("--host", default=HOST_IMAGE_PATH)
    embed_parser.add_argument("--watermark", default=WATERMARK_IMAGE_PATH)
    embed_parser.add_argument("--output", default=WATERMARKED_IMAGE_PATH)
    embed_parser.add_argument("--embedding-qf", type=int, default=DEFAULT_EMBEDDING_QF)

    extract_parser = subparsers.add_parser(
        "extract", help="extract watermark from an existing watermarked image"
    )
    extract_parser.add_argument("--watermarked", default=WATERMARKED_IMAGE_PATH)
    extract_parser.add_argument("--watermark", default=WATERMARK_IMAGE_PATH)
    extract_parser.add_argument("--output", default=EXTRACTED_WATERMARK_PATH)
    extract_parser.add_argument("--embedding-qf", type=int, default=DEFAULT_EMBEDDING_QF)

    compress_parser = subparsers.add_parser(
        "compress", help="recompress watermarked image at one recompress QF and extract"
    )
    compress_parser.add_argument("--watermarked", default=WATERMARKED_IMAGE_PATH)
    compress_parser.add_argument("--watermark", default=WATERMARK_IMAGE_PATH)
    compress_parser.add_argument("--embedding-qf", type=int, default=DEFAULT_EMBEDDING_QF)
    compress_parser.add_argument(
        "--recompress-qf", type=int, default=DEFAULT_RECOMPRESS_QF
    )

    return parser.parse_args()


def main() -> None:
    """Dispatch the selected command-line mode."""
    args = parse_args()

    try:
        if args.mode == "embed":
            run_embed_watermark(
                host_image_path=args.host,
                watermark_image_path=args.watermark,
                watermarked_image_path=args.output,
                embedding_qf=args.embedding_qf,
            )
        elif args.mode == "extract":
            run_extract_watermark(
                watermarked_image_path=args.watermarked,
                watermark_image_path=args.watermark,
                extracted_watermark_path=args.output,
                embedding_qf=args.embedding_qf,
            )
        elif args.mode == "compress":
            run_compress_recompression(
                watermarked_image_path=args.watermarked,
                watermark_image_path=args.watermark,
                embedding_qf=args.embedding_qf,
                recompress_qf=args.recompress_qf,
            )
    except FileNotFoundError as exc:
        print(exc)
        print("Place input.jpg and watermark.png in this folder, then run again.")


if __name__ == "__main__":
    main()
