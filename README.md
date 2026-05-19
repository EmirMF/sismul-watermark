# DCT JPEG Binary Watermarking

[Dokumen laporan Watermarking](Watermarking_SistemMultimedia_18224066.pdf)

Implementasi watermarking citra biner pada domain DCT dengan quantization
bergaya JPEG. Watermark ditanam pada parity koefisien DCT `(4, 4)` di channel
Y/luminance, sehingga proses ekstraksi bersifat blind dan tidak membutuhkan
gambar asli.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## File Input

Letakkan file berikut di folder proyek:

```text
input.jpg       Gambar host/original
watermark.png   Watermark hitam-putih
```

Watermark akan dibinerisasi otomatis:

```text
hitam -> bit 0
putih -> bit 1
```

Kapasitas host adalah satu bit per blok `8x8`:

```text
kapasitas = (tinggi / 8) * (lebar / 8)
```

## Cara Menjalankan

Embedding:

```bash
python dct_jpeg_watermark.py embed
```

Output:

```text
img_output/watermarked.jpg
```

Extraction:

```bash
python dct_jpeg_watermark.py extract
```

Output:

```text
img_output/extracted_watermark.png
```

Recompression test:

```bash
python dct_jpeg_watermark.py compress --recompress-qf 30
```

Output:

```text
img_output/watermarked_recompress_qf30.jpg
img_output/extracted_watermark_recompress_qf30.png
```

## CLI Usage

```bash
python dct_jpeg_watermark.py <mode> [options]
```

Mode yang tersedia:

```text
embed      menyisipkan watermark ke input image
extract    mengekstrak watermark dari watermarked image
compress   recompress watermarked image lalu ekstrak watermark
```

Opsi `embed`:

```bash
python dct_jpeg_watermark.py embed \
  --host input.jpg \
  --watermark watermark.png \
  --output img_output/watermarked.jpg \
  --embedding-qf 90
```

Opsi `extract`:

```bash
python dct_jpeg_watermark.py extract \
  --watermarked img_output/watermarked.jpg \
  --watermark watermark.png \
  --output img_output/extracted_watermark.png \
  --embedding-qf 90
```

Opsi `compress`:

```bash
python dct_jpeg_watermark.py compress \
  --watermarked img_output/watermarked.jpg \
  --watermark watermark.png \
  --embedding-qf 90 \
  --recompress-qf 30
```

## Parameter

Default utama:

```text
embedding QF   = 90
recompress QF = 30
coefficient   = (4, 4)
```

Contoh menjalankan dengan QF lain:

```bash
python dct_jpeg_watermark.py embed --embedding-qf 100
python dct_jpeg_watermark.py extract --embedding-qf 100
python dct_jpeg_watermark.py compress --embedding-qf 100 --recompress-qf 75
```

Catatan: nilai `--embedding-qf` saat extraction dan compress harus sama dengan
nilai yang dipakai saat embedding, karena menentukan quantization matrix untuk
membaca parity koefisien.

## Ringkas Alur

- Host diproses pada channel Y dan dibagi menjadi blok `8x8`.
- Setiap blok di-level shift, lalu masuk DCT.
- Satu bit watermark ditanam pada parity indeks kuantisasi koefisien `(4, 4)`.
- Blok dikembalikan dengan dequantization, IDCT, inverse shift, lalu disimpan.
- Extraction membaca parity koefisien yang sama dari `img_output/watermarked.jpg`.
