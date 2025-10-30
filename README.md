# Visualisasi Iklim & Kesesuaian Lahan Kelapa Sawit

Aplikasi web interaktif untuk visualisasi data iklim dan analisis kesesuaian lahan untuk tanaman kelapa sawit. Bagian dari Aktualisasi CPNS Golongan IIIA tahun 2025, Direktorat Layanan Iklim Terapan BMKG.

## Fitur

1. **Visualisasi Spasial Curah Hujan**
   - Peta sebaran curah hujan tahunan
   - Tampilan interaktif dengan colorbar
   - Opsi untuk menambahkan batas wilayah

2. **Analisis Temporal**
   - Grafik curah hujan bulanan
   - Grafik temperatur bulanan
   - Analisis bulan kering

3. **Visualisasi Temperatur**
   - Peta sebaran rata-rata temperatur
   - Grafik temporal temperatur bulanan

4. **Analisis Kemiringan Lahan**
   - Peta sebaran kemiringan dalam persen
   - Pengaturan batas maksimum slope

5. **Peta Kesesuaian Lahan**
   - Visualisasi kesesuaian fuzzy dan crisp
   - Klasifikasi S1, S2, S3, dan N
   - Opsi download peta dalam format PNG

## Persyaratan Data Input

File input harus dalam format:
- `.parquet` atau `.csv`
- Memiliki kolom wajib:
  - `LON`: Longitude/Bujur
  - `LAT`: Latitude/Lintang
  - `CH_Jan` hingga `CH_Des`: Data curah hujan bulanan
  - `T2M_Jan` hingga `T2M_Des`: Data temperatur bulanan (opsional)
  - `slope_percent`: Data kemiringan lahan (opsional)
  - `kelas` dan `kelas_crisp`: Klasifikasi kesesuaian lahan

## Cara Penggunaan

1. Jalankan aplikasi dengan Streamlit:
   ```bash
   streamlit run app.py
   ```

2. Upload data:
   - Upload file data (.parquet atau .csv) melalui sidebar
   - Opsional: Upload file batas wilayah (.shp dalam .zip atau .geojson)

3. Atur parameter di sidebar:
   - Ambang batas bulan kering
   - Ukuran titik visualisasi
   - Batas maksimum slope
   - Tingkat zoom peta

4. Pilih jenis visualisasi yang diinginkan dari dropdown menu

5. Download hasil visualisasi dalam format PNG

## Dependensi

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- GeoPandas (opsional, untuk fitur batas wilayah)
- PyArrow atau FastParquet (untuk membaca file parquet)

## Catatan

- Aplikasi akan memberikan peringatan jika kolom yang diperlukan tidak lengkap
- Fitur batas wilayah bersifat opsional dan memerlukan GeoPandas
- Visualisasi dapat di-zoom dan memiliki opsi download
- Klasifikasi kesesuaian menggunakan skema warna:
  - S1: Hijau (#1b9e77)
  - S2: Biru (#1f77b4)
  - S3: Merah (#d62728)
  - N: Putih (#ffffff)
