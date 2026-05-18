"""
Binary-image watermarking in the DCT domain with JPEG-like quantization.

This version embeds the watermark BEFORE quantization:

    block -> level shift -> FDCT -> watermark coefficient edit -> quantization
    -> dequantization -> IDCT -> inverse level shift

That ordering is important. If the quality factor (QF) is low, JPEG
quantization becomes coarse and can round away the watermark perturbation, so
the extracted watermark may become noisy or unreadable. This matches the
behavior requested for a fragile/quality-sensitive DCT watermark.

Only numpy and OpenCV are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class WatermarkConfig:
    """Configuration for DCT-domain binary image watermarking."""

    quality_factor: int = 90
    alpha: float = 20.0
    coefficient: Tuple[int, int] = (4, 4)
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

        if self.config.alpha <= 0:
            raise ValueError("alpha must be positive.")

    @classmethod
    def generate_luminance_quantization_matrix(cls, quality_factor: int) -> np.ndarray:
        """
        Generate the scaled JPEG luminance quantization matrix.

        The JPEG quality scaling formula is intentionally non-linear:
        - QF < 50  -> scale = 5000 / QF, so very low quality grows rapidly.
        - QF >= 50 -> scale = 200 - 2*QF, so quality 100 approaches no loss.

        Quantization divides each DCT coefficient by this matrix and rounds the
        result. Larger matrix values mean a larger coefficient perturbation is
        required to survive. Therefore, low QF can destroy this watermark.
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

    def _jpeg_reconstruct(self, dct_block: np.ndarray) -> np.ndarray:
        """
        Simulate JPEG quantization and decoding for one DCT block.

        Because the watermark is inserted before this function is called, a low
        QF can remove the additive watermark during rounding.
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

        Coefficient (4,4) is a mid-frequency position. It is less visually
        obvious than low frequencies, but more likely to survive than very high
        frequencies. Since embedding happens BEFORE quantization, survival still
        depends strongly on QF and alpha.
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
                    # Binary watermark mapping: white pixel/1 -> +alpha,
                    # black pixel/0 -> -alpha. This perturbation is still
                    # vulnerable to the following quantization step.
                    signal = 1.0 if watermark_bits[bit_index] == 1 else -1.0
                    dct_block[u, v] += self.config.alpha * signal
                    bit_index += 1

                watermarked_y[row : row + 8, col : col + 8] = self._jpeg_reconstruct(
                    dct_block
                )

        return self._restore_image(watermarked_y, chroma)

    def extract_binary_watermark(
        self,
        original_image: np.ndarray,
        watermarked_image: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Extract a binary watermark image using non-blind extraction.

        The original and watermarked images are both transformed and quantized.
        If the quantized mid-frequency coefficient increased, the bit is read as
        1; otherwise it is read as 0. Low QF can make many differences collapse
        to zero or flip, producing a damaged watermark.
        """
        original_y, _ = self._prepare_luminance(original_image)
        watermarked_y, _ = self._prepare_luminance(watermarked_image)

        min_height = min(original_y.shape[0], watermarked_y.shape[0])
        min_width = min(original_y.shape[1], watermarked_y.shape[1])
        original_y = original_y[:min_height, :min_width]
        watermarked_y = watermarked_y[:min_height, :min_width]

        self._validate_watermark_capacity(original_y.shape, watermark_shape)

        required_bits = watermark_shape[0] * watermark_shape[1]
        extracted_bits: list[int] = []
        u, v = self.config.coefficient

        for row in range(0, original_y.shape[0], 8):
            for col in range(0, original_y.shape[1], 8):
                original_block = original_y[row : row + 8, col : col + 8]
                watermarked_block = watermarked_y[row : row + 8, col : col + 8]

                original_q = self._quantize(self._fdct(original_block))
                watermarked_q = self._quantize(self._fdct(watermarked_block))
                difference = watermarked_q[u, v] - original_q[u, v]

                extracted_bits.append(1 if difference > 0 else 0)
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
        if not cv2.imwrite(output_path, watermarked):
            raise IOError(f"Could not write watermarked image: {output_path}")

        return watermarked

    def extract_file(
        self,
        original_path: str,
        watermarked_path: str,
        watermark_shape: Tuple[int, int],
        output_path: str,
    ) -> np.ndarray:
        """Extract a binary watermark from files and save it as a visible image."""
        original = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"Could not read original image: {original_path}")

        watermarked = cv2.imread(watermarked_path, cv2.IMREAD_COLOR)
        if watermarked is None:
            raise FileNotFoundError(f"Could not read watermarked image: {watermarked_path}")

        extracted = self.extract_binary_watermark(
            original, watermarked, watermark_shape=watermark_shape
        )
        if not cv2.imwrite(output_path, extracted):
            raise IOError(f"Could not write extracted watermark image: {output_path}")

        return extracted


if __name__ == "__main__":
    # Example usage:
    #
    # 1. Put a host image at "input.jpg".
    # 2. Put a black-white watermark image at "watermark.png".
    # 3. Set watermark_shape to the watermark image size, e.g. (32, 32).
    # 4. Run: python dct_jpeg_watermark.py
    #
    # Try quality_factor=90 first. Then lower it to 10 or 1: because embedding is
    # done before quantization, the extracted watermark should become much noisier.
    watermarker = DCTJPEGWatermarker(
        WatermarkConfig(
            quality_factor=71,
            alpha=20.0,
            coefficient=(4, 4),
            use_y_channel=True,
        )
    )

    host_image_path = "input.jpg"
    watermark_image_path = "watermark.png"
    watermarked_image_path = "watermarked.jpg"
    extracted_watermark_path = "extracted_watermark.png"

    try:
        watermark_for_shape = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
        if watermark_for_shape is None:
            raise FileNotFoundError(f"Could not read watermark image: {watermark_image_path}")

        watermarker.embed_file(
            host_image_path, watermark_image_path, watermarked_image_path
        )
        watermarker.extract_file(
            host_image_path,
            watermarked_image_path,
            watermark_shape=watermark_for_shape.shape,
            output_path=extracted_watermark_path,
        )
        print(f"Watermarked image saved to: {watermarked_image_path}")
        print(f"Extracted watermark saved to: {extracted_watermark_path}")
    except FileNotFoundError as exc:
        print(exc)
        print("Place input.jpg and watermark.png in this folder, then run again.")
