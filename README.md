# Retrieval putusan ITE menggunakan Case-Based Reasoning

Proyek kelompok Penalaran Komputer membangun prototipe retrieval pada 50 rekaman putusan. Sistem mengubah teks menjadi TF-IDF, mengurutkan cosine similarity, dan menggunakan voting berbobot dari lima kasus teratas untuk label reuse. Amar diambil dari kasus peringkat pertama.

## Data dan pipeline

`data/processed/cases.csv` tersedia sebagai snapshot basis kasus yang digunakan kode lokal. Folder `src/` memuat preprocessing, ekstraksi metadata, retrieval, reuse, dan evaluasi. TF-IDF retrieval menggunakan unigram/bigram, maksimal 30.000 fitur, serta daftar stopword khusus. Classifier Multinomial Naive Bayes berfungsi sebagai pembanding label.

Nama folder dan README lama menyebut PN Surabaya. Laporan menyebut PT Surabaya; kolom metadata `pengadilan` menandai semua 50 rekaman sebagai Pengadilan Tinggi Surabaya. Namun, hasil ekstraksi masih perlu diperiksa terhadap dokumen asal. README portofolio menggunakan judul netral sampai cakupan korpus dipastikan. Pemeriksaan menemukan 48 nomor perkara unik yang tidak kosong dan satu baris dengan teks lengkap duplikat. Jumlah 50 mengacu pada rekaman, dengan jumlah putusan unik yang belum dipastikan.

## Dua konfigurasi evaluasi

| Sumber | Jumlah kueri | Hasil retrieval yang tercatat | Status |
| --- | ---: | --- | --- |
| Draft artikel dan output notebook `05_evaluation` | 5 | Ground truth berada di peringkat pertama pada semua kueri | Hasil historis; teks lengkap kelima kueri belum ditemukan dalam `queries.json` aktif |
| `data/eval/queries.json`, CSV aktif, dan notebook `00_full_cbr_workflow` | 2 | Ground truth tidak masuk top-5 pada kedua kueri | Konfigurasi aktif diperiksa ulang; hasil rinci di `results/` |

Keduanya merupakan eksperimen berbeda. Artikel mencatat Hit@5 = 1,00 dan Precision@5 = 0,20 karena hanya satu ground truth per kueri. Konfigurasi aktif mencatat Hit@5 = 0,00 dan Top-1 Accuracy = 0,00. Prediksi label pada dua kueri aktif benar, tetapi kedua label harapannya sama, yaitu pidana penjara. Hasil ini belum mengukur kemampuan membedakan kelas.

Bukti historis ada pada [draft artikel tugas kuliah](evidence/draft-artikel.docx) dan [notebook evaluasi lima kueri](evidence/05_evaluation.ipynb). Bukti konfigurasi aktif ada pada [workflow lengkap](evidence/00_full_cbr_workflow.ipynb), `data/eval/`, dan [hasil pemeriksaan ulang retrieval](results/verified-current-retrieval.csv).

## Menjalankan snapshot yang tersedia

Jalankan dari folder proyek `cbr-ite/`:

```powershell
python -m pip install -r requirements.txt
python -m src.pipeline --stage retrieval
python -m src.pipeline --stage evaluate --k 5
```

Perintah pertama membangun ulang model dari `cases.csv`, lalu perintah berikutnya menjalankan kueri aktif dalam `data/eval/queries.json`. Artifak model muncul pada `models/`, dan metrik evaluasi ditulis ke `data/eval/`. Salinan metrik pemeriksaan sesi ini tersimpan terpisah di `results/verified-current-*.csv`.

Contoh eksplorasi query:

```powershell
python -m src.pipeline --query "penyebaran informasi elektronik melalui media sosial" --k 5
```

Kueri contoh tidak memiliki ground truth dan tidak dipakai untuk klaim akurasi. Tahap `--all` memerlukan dokumen mentah; dokumen tersebut belum dimasukkan ke paket ini. Snapshot terstruktur cukup untuk mengulangi tahap retrieval dan evaluasi aktif, tetapi belum untuk mereplikasi ekstraksi dari PDF.

## Batasan

Sebagian kueri historis dekat dengan kasus dalam korpus. Relevansi baru direpresentasikan oleh satu target per kueri. Keberhasilan menemukan target belum mengukur kualitas seluruh daftar kandidat. Metadata, amar final, duplikasi, dan label tidak diketahui perlu diaudit dari sumber dokumen.

Teks retrieval dapat menyertakan amar dan bagian putusan lengkap. Untuk tugas mencari dokumen yang sudah terbit, pilihan ini perlu dijelaskan. Untuk mengevaluasi prediksi hasil perkara baru, fitur yang hanya diketahui setelah putusan harus dikeluarkan dan desain evaluasinya perlu diubah.

Langkah berikutnya adalah memulihkan lima kueri lengkap beserta ground truth asli, memisahkan dua eksperimen, memeriksa duplikasi dokumen, serta menambah penilaian relevansi manual. Prototipe membantu pencarian dokumen pembanding; keluaran reuse belum divalidasi sebagai prediksi putusan perkara baru.

Penulis artikel: Naufal Zainul Zaki dan Sabian Franstiawan. Kontribusi pribadi Naufal belum dikonfirmasi.
