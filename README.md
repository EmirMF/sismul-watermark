# DCT JPEG Binary Watermarking

Proyek ini berisi implementasi watermarking citra biner pada gambar JPEG-like
menggunakan Discrete Cosine Transform (DCT). Watermark disisipkan dengan metode
blind parity/QIM pada koefisien DCT, sehingga extraction cukup memakai
`watermarked.jpg` dan ukuran watermark.

## Fitur

- Watermark berupa citra biner `watermark.png`.
- Host image dapat berupa grayscale atau BGR/color.
- Pemrosesan dilakukan per blok `8x8`, seperti pipeline JPEG.
- Menggunakan level shifting, FDCT, quantization, dequantization, IDCT, dan inverse shifting.
- Quantization matrix mengikuti standar JPEG luminance matrix dengan scaling berdasarkan QF.
- Ekstraksi bersifat blind, sehingga gambar asli tidak dibutuhkan saat extraction.
- Hanya menggunakan `numpy` dan `opencv-python`.

## Setup

Buat virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

## Struktur File

```text
dct_jpeg_watermark.py   Script utama watermarking
requirements.txt        Dependency Python
README.md               Dokumentasi proyek
```

File input/output lokal yang digunakan saat eksperimen:

```text
input.jpg                Gambar host/original
watermark.png            Citra watermark biner
watermarked.jpg          Gambar hasil embedding
extracted_watermark.png  Hasil ekstraksi watermark
```

## Format Watermark

`watermark.png` sebaiknya berupa gambar hitam-putih sederhana.

Contoh yang cocok:

- Logo hitam-putih.
- Teks putih di background hitam.
- Ukuran kecil seperti `16x16`, `32x32`, atau `64x64`.
- Format PNG agar watermark tidak rusak karena kompresi.

Script akan mengubah watermark menjadi biner:

```text
hitam  -> bit 0
putih  -> bit 1
```

Kapasitas host:

```text
kapasitas bit = (tinggi_host / 8) * (lebar_host / 8)
```

Contoh:

```text
input.jpg 256x256
kapasitas = 32 * 32 = 1024 bit
watermark maksimum = 32x32 pixel
```

## Cara Menjalankan

Letakkan file berikut di folder proyek:

```text
input.jpg
watermark.png
```

Jalankan:

```bash
python dct_jpeg_watermark.py
```

Output yang dibuat:

```text
watermarked.jpg
extracted_watermark.png
```

## Konfigurasi

Pengaturan utama ada di bagian bawah `dct_jpeg_watermark.py`:

```python
watermarker = DCTJPEGWatermarker(
    WatermarkConfig(
        quality_factor=71,
        coefficient=(4, 4),
        use_y_channel=True,
    )
)
```

Parameter:

- `quality_factor`: kualitas simulasi JPEG, rentang `1..100`.
- `coefficient`: posisi koefisien DCT yang dipakai untuk embedding.
- `use_y_channel`: jika `True`, watermark ditanam pada channel luminance `Y`.

## Alur Embedding

1. Baca `input.jpg`.
2. Jika gambar berwarna, ambil channel `Y` dari YCrCb.
3. Crop ukuran gambar agar habis dibagi `8`.
4. Baca `watermark.png` lalu ubah menjadi citra biner.
5. Bagi host image menjadi blok non-overlap `8x8`.
6. Lakukan level shifting dengan mengurangi pixel sebesar `128`.
7. Terapkan Forward DCT pada setiap blok.
8. Sisipkan bit watermark pada parity indeks kuantisasi koefisien `(4,4)`.
9. Lakukan quantization menggunakan matrix JPEG luminance sesuai QF.
10. Lakukan dequantization, IDCT, inverse shifting, dan clipping ke `0..255`.
11. Simpan hasil sebagai `watermarked.jpg`.

## Alur Extraction

Ekstraksi hanya membutuhkan gambar watermarked dan ukuran watermark:

```text
watermarked.jpg  Gambar hasil embedding
watermark_shape   Ukuran watermark, misalnya (32, 32)
```

Langkah ekstraksi:

1. Ambil channel `Y` dari `watermarked.jpg`.
2. Bagi gambar menjadi blok `8x8`.
3. Lakukan level shifting, DCT, dan quantization.
4. Ambil indeks koefisien terkuantisasi pada posisi `(4,4)`.
5. Jika indeks genap, bit dibaca sebagai `0`.
6. Jika indeks ganjil, bit dibaca sebagai `1`.
7. Susun bit kembali menjadi citra watermark.
8. Simpan hasil sebagai `extracted_watermark.png`.

## Mengapa QF Rendah Merusak Watermark

Watermark disisipkan sebagai parity indeks kuantisasi:

```text
DCT -> paksa parity indeks koefisien -> quantization -> dequantization -> IDCT
```

Pada QF rendah, nilai quantization matrix menjadi besar dan JPEG compression
menjadi lebih kasar. Perubahan pada koefisien mid-frequency lebih mudah
terganggu oleh rounding, clipping, dan recompression JPEG, sehingga parity yang
dibaca saat extraction dapat berubah.

Itulah sebabnya:

```text
QF tinggi  -> watermark lebih mudah diekstrak
QF rendah  -> watermark lebih noisy atau gagal diekstrak
```

Jika watermark terlalu sulit terbaca, gunakan QF lebih tinggi atau pilih
koefisien yang sedikit lebih rendah frekuensinya. Jika watermark terlalu mudah
terbaca pada QF rendah, pilih koefisien dengan frekuensi lebih tinggi, misalnya
`(5,5)`.

## Catatan DCT

Pada blok DCT `8x8`, posisi kiri atas adalah frekuensi rendah:

```text
(0,0)      DC / rata-rata brightness
kanan      frekuensi horizontal makin tinggi
bawah      frekuensi vertikal makin tinggi
kanan bawah frekuensi paling tinggi
```

Koefisien `(4,4)` dipilih sebagai mid-frequency karena relatif tidak terlalu
terlihat oleh mata, tetapi masih lebih tahan dibanding area high-frequency
seperti `(7,7)`.
