# Laporan Proyek Machine Learning - Rebecca Olivia

## Domain Proyek
Domain yang dipilih untuk proyek _machine learning_ ini adalah **Kesehatan**, dengan judul **Predictive Analytics: Prediksi Kanker**.

### Latar Belakang
![dataset-cover](https://github.com/user-attachments/assets/609efb12-0a90-4358-9839-70f56b915838)

Beban penyakit kanker di Indonesia tergolong tinggi. Mengacu pada data _Global Burden of Cancer Study_ (GLOBOCAN) tahun 2020, dari total 19,3 juta kasus kanker secara global, Indonesia mencatatkan 396.914 kasus dan 234.511 kematian yang disebabkan oleh kanker [[1](https://ejournal.unib.ac.id/JurnalVokasiKeperawatan/article/view/22338/10237)]. Lima jenis kanker yang paling banyak ditemui baik pada laki-laki maupun perempuan Indonesia adalah kanker payudara, paru, serviks, kolorektal atau usus besar dan rektum, serta hati [[2](https://ugm.ac.id/id/berita/jumlah-penderita-kanker-terus-meningkat-kenali-gejala-awal-untuk-deteksi-dini/#:~:text=Dari%20sejumlah%20kasus%20yang%20ada,besar%20dan%20rektum%2C%20serta%20hati.)].

Salah satu tantangan terbesar dalam penanganan kanker adalah keterlambatan dalam proses diagnosis. Banyak pasien baru teridentifikasi mengidap kanker ketika penyakit sudah memasuki stadium lanjut, sehingga peluang kesembuhan menjadi sangat kecil. Untuk itu, deteksi dini memegang peranan penting dalam meningkatkan angka harapan hidup. Upaya ini dapat dilakukan melalui pemeriksaan kesehatan secara rutin serta pengenalan terhadap gejala awal. Selain itu, prediksi yang akurat juga menjadi kunci untuk memperbesar kemungkinan sembuh dan mengurangi angka kematian akibat kanker [[3](https://www.ejournal.itn.ac.id/index.php/jati/article/view/10752/6190)].

Penerapan *predictive analytics* dalam bidang kesehatan dapat membantu mendeteksi risiko kanker sejak dini menggunakan data rekam medis dan gaya hidup. Model prediksi ini dapat membantu:

- **Tenaga medis**: untuk menyaring pasien yang berisiko tinggi secara cepat dan efisien.
- **Pasien**: agar bisa mengambil langkah preventif lebih awal.
- **Pemerintah dan institusi kesehatan**: dalam menyusun kebijakan berbasis data dan efisiensi sumber daya.

Dalam proyek ini, digunakan pendekatan machine learning berbasis klasifikasi untuk memprediksi apakah seseorang berpotensi mengidap kanker atau tidak, berdasarkan fitur seperti usia, jenis kelamin, indeks massa tubuh (IMT), kebiasaan merokok, risiko genetik, tingkat aktivitas fisik, konsumsi alkohol mingguan, riwayat pribadi terhadap kanker, serta status diagnosis.

## Business Understanding
Pengembangan model prediksi kanker sangat penting dalam mendukung diagnosis dini dan penanganan penyakit secara lebih efektif. Dengan memanfaatkan data medis seperti data visual dan sensorik (misalnya ukuran benjolan, tekstur, atau hasil pemeriksaan laboratorium), model ini dapat membantu para tenaga medis dalam mengidentifikasi potensi kanker sejak dini. Selain itu, model ini juga dapat meningkatkan efisiensi layanan kesehatan dan mengurangi beban kerja dokter dalam proses diagnosis awal. Prediksi yang akurat akan berdampak besar terhadap kualitas hidup pasien melalui tindakan penanganan yang lebih cepat dan tepat sasaran.

### Problem Statements
Berdasarkan latar belakang di atas, maka permasalahan yang akan dijawab dalam proyek ini adalah:
1. Bagaimana membangun model machine learning yang dapat memprediksi kemungkinan seseorang mengidap kanker berdasarkan data fitur medis?
2. Model machine learning apa yang memiliki akurasi terbaik dalam memprediksi kasus kanker?
3. Bagaimana penerapan model ini dapat membantu meningkatkan efektivitas diagnosis dini dalam sistem layanan kesehatan?

### Goals
Proyek ini memiliki tujuan sebagai berikut:
1. Mengembangkan model machine learning untuk memprediksi kemungkinan seseorang menderita kanker berdasarkan data medis (fitur numerik dan kategorikal).
2. Membandingkan beberapa algoritma klasifikasi untuk menentukan model dengan performa terbaik.
3. Menyediakan solusi yang dapat mendukung diagnosis awal dan pengambilan keputusan medis yang lebih cepat dan akurat.

### Solution Statements
Untuk mencapai tujuan proyek ini, langkah-langkah yang dilakukan meliputi:

- **Eksplorasi dan Analisis Data**  
  Melakukan analisis univariat dan multivariat terhadap fitur-fitur dalam dataset untuk memahami distribusi data, mengidentifikasi korelasi antar fitur, serta mendeteksi outlier yang berpotensi memengaruhi kinerja model.

- **Preprocessing Data**  
  Melakukan pembersihan data (handling missing values dan duplikasi) dan normalisasi agar data siap digunakan dalam proses pelatihan model machine learning sehingga dapat menghasilkan prediksi yang optimal.

- **Pembangunan dan Evaluasi Model**  
  Membangun dan membandingkan beberapa algoritma machine learning untuk menentukan model terbaik dalam melakukan prediksi. Model yang digunakan antara lain:
  
  - **K-Nearest Neighbors (KNN)**: KNN merupakan algoritma klasifikasi yang bekerja dengan prinsip kesamaan jarak. Dalam proses prediksinya, algoritma ini akan mencari sejumlah _k_ tetangga terdekat dari data uji berdasarkan jarak tertentu, seperti Euclidean distance. Objek baru kemudian diklasifikasikan ke dalam kelas mayoritas dari tetangga terdekat tersebut. Pendekatan ini bersifat non-parametrik dan cocok digunakan ketika hubungan antar fitur bersifat non-linear [[4](https://journal.irpi.or.id/index.php/malcom/article/view/1078/519)].
  - **Random Forest**: Random Forest adalah algoritma ensemble yang terdiri dari banyak pohon keputusan (_decision trees_) yang dibangun dari subset acak data dan fitur. Setiap pohon membuat prediksi secara mandiri, kemudian hasil akhirnya ditentukan melalui voting mayoritas. Pendekatan ini meningkatkan akurasi dan kestabilan model serta mampu mengurangi risiko overfitting yang kerap terjadi pada model _decision tree_ tunggal [[5](https://jidt.org/jidt/article/view/393/205)].
  - **Support Vector Machine (SVM)**: SVM merupakan algoritma supervised learning yang berfungsi untuk memisahkan data ke dalam dua kelas atau lebih dengan mencari hyperplane terbaik yang memaksimalkan margin antar kelas. Metode ini dapat bekerja pada ruang data berdimensi tinggi dan juga mampu menangani klasifikasi non-linear menggunakan kernel trick, sehingga sangat andal untuk berbagai jenis permasalahan klasifikasi [[6](https://ejournal.poltekharber.ac.id/index.php/informatika/article/view/977/795)].
  - **Naive Bayes**: Naive Bayes adalah metode klasifikasi berbasis probabilitas yang menerapkan Teorema Bayes dengan asumsi independensi antar fitur. Meskipun asumsi ini jarang terpenuhi secara ketat dalam data nyata, algoritma ini terbukti efisien dan cukup akurat dalam berbagai kasus. Kelebihan utama Naive Bayes adalah kesederhanaannya dalam implementasi dan kecepatan dalam melakukan prediksi [[7](https://ojs.stikombanyuwangi.ac.id/index.php/jikom/article/view/280/147)].
  - **Extra Trees Classifier**: Extra Trees (Extremely Randomized Trees) adalah varian dari Random Forest yang membedakan diri melalui pemilihan split point secara acak saat membangun pohon keputusan. Teknik ini memperkenalkan lebih banyak variasi antar pohon dalam ensemble, sehingga dapat meningkatkan kemampuan generalisasi model dan efisiensi waktu pelatihan. Seperti Random Forest, keputusan akhir diambil berdasarkan hasil agregasi dari seluruh pohon [[8](https://www.ejournal.itn.ac.id/index.php/jati/article/view/8797/4781)].

## Data Understanding
### Gathering Data
Dataset yang digunakan dalam proyek ini adalah "The Cancer Data V2", yang berisi informasi mengenai karakteristik pasien serta status diagnosis kanker. File dataset diunggah dari penyimpanan lokal ke lingkungan kerja Google Colab dalam format ZIP, kemudian diekstrak dan dibaca menggunakan library pandas.

Proses pengumpulan data dilakukan melalui tiga langkah utama:
1. Mengunggah file ZIP (The_Cancer_data_1500_V2.csv.zip) dari komputer lokal ke Google Colab.
2. Mengekstrak isi file ZIP untuk mendapatkan file CSV.
3. Membaca file CSV ke dalam bentuk DataFrame agar dapat dianalisis lebih lanjut.

Dataset ini berisi 1.500 entri dan 9 fitur yang merepresentasikan karakteristik pasien, seperti usia, jenis kelamin, indeks massa tubuh (IMT), kebiasaan merokok, risiko genetik, tingkat aktivitas fisik, konsumsi alkohol mingguan, riwayat pribadi terhadap kanker, serta status diagnosis.

### Data Assesing and Data Cleaning
Informasi Datasets

| Jenis       | Keterangan                                                                                                                                      |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Title**   | 🩺📊 Cancer Prediction Dataset 🌟🔬                                                                                                               |
| **Source**  | [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)                                                              |
| **Maintainer** | [Rabie El Kharoua](https://www.kaggle.com/rabieelkharoua)                                                                                     |
| **License** | Attribution 4.0 International                                                                                                                   |
| **Visibility** | Publik                                                                                                                                       |
| **Tags**    | Earth and Nature, Cancer, Tabular, Health Conditions, Binary Classification                                                                     |
| **Usability** | 10.00                                                                                                                                         |

> Klik pada nama **Kaggle** untuk langsung menuju ke dataset.
> Klik pada nama **Rabie El Kharoua** untuk langsung menuju ke user pemilik datasey

**Dataset Info**
- Format: CSV (Comma-Separated Values)
- Jumlah data: **1.500 baris** dan **9 kolom**
- Semua kolom memiliki **1500 nilai non-null** (tidak ada missing values)
- Tipe data:
  - `int64` (6 kolom): untuk data diskrit (contoh: usia, biner)
  - `float64` (3 kolom): untuk data kontinu (contoh: BMI, aktivitas fisik)

Contoh data:
| Age | Gender | BMI       | Smoking | GeneticRisk | PhysicalActivity | AlcoholIntake | CancerHistory | Diagnosis |
|-----|--------|-----------|---------|--------------|------------------|----------------|----------------|------------|
| 58  | 1      | 16.08     | 0       | 1            | 8.15             | 4.15           | 1              | 1          |
| 71  | 0      | 30.83     | 0       | 1            | 9.36             | 3.52           | 0              | 0          |
| 48  | 1      | 38.78     | 0       | 2            | 5.14             | 4.73           | 0              | 1          |

---

**Feature Explanation**

| Feature Name       | Tipe Data | Deskripsi                                                                                  |
|--------------------|-----------|---------------------------------------------------------------------------------------------|
| `Age`              | Integer   | Usia pasien dalam tahun. Rentang antara 20 hingga 80 tahun.                                |
| `Gender`           | Integer   | Jenis kelamin pasien. `0` = laki-laki, `1` = perempuan.                                     |
| `BMI`              | Float     | Indeks massa tubuh pasien. Nilai berkisar antara 15 – 40.                                   |
| `Smoking`          | Integer   | Status merokok pasien. `0` = tidak merokok, `1` = merokok.                                  |
| `GeneticRisk`      | Integer   | Risiko genetik terhadap kanker. `0` = rendah, `1` = sedang, `2` = tinggi.                   |
| `PhysicalActivity` | Float     | Jam aktivitas fisik per minggu. Rentang 0 – 10 jam/minggu.                                 |
| `AlcoholIntake`    | Float     | Konsumsi alkohol per minggu. Rentang 0 – 5 satuan/minggu.                                   |
| `CancerHistory`    | Integer   | Riwayat pribadi terhadap kanker. `0` = tidak ada, `1` = ada.                                |
| `Diagnosis`        | Integer   | Label target diagnosis. `0` = tidak terdiagnosis kanker, `1` = terdiagnosis kanker.         |

### Exploratory Data Analysis (EDA)
Tahap *Exploratory Data Analysis (EDA)* dilakukan untuk memahami karakteristik data secara menyeluruh sebelum memasuki proses *data preprocessing* dan pemodelan. Proses ini bertujuan untuk:

- Mengetahui distribusi data dan proporsi target.
- Mengidentifikasi hubungan antar fitur.
- Mendeteksi nilai pencilan (*outliers*).
- Mengatasi ketidakseimbangan kelas pada label target.

---
1. Univariate Analysis  
<p align="center">
  <img src="https://github.com/user-attachments/assets/4383e96a-86e6-4101-9f3d-0f72583c02aa" alt="Gambar 1. Pie-Chart Cancer & Non-Cancer" width="500"/>
</p>

<p align="center"><strong>Gambar 1.</strong> Pie-Chart Kanker & Kanker</p>

Pie chart di atas menunjukkan proporsi pasien yang terdiagnosis kanker (`1`) dan non-kanker (`0`). Sebanyak <strong>557 pasien (37.1%)</strong> terdiagnosis kanker (warna pink), sedangkan <strong>943 pasien (62.9%)</strong> tidak terdiagnosis kanker (warna biru).  Informasi ini penting untuk mengetahui apakah kelas target seimbang atau tidak, karena ketidakseimbangan dapat menyebabkan bias dalam model klasifikasi.


<p align="center">
  <img src="https://github.com/user-attachments/assets/5cd2cb5e-f1c3-4049-81e8-afe16b3d869a" alt="Gambar 2. Histogram Fitur Numerik" width="500"/>
</p>

<p align="center"><strong>Gambar 2.</strong> Histogram Fitur Numerik</p>

Histogram digunakan untuk melihat distribusi dari seluruh fitur numerik. Sebagian besar fitur menunjukkan distribusi yang tidak simetris, dengan beberapa fitur memiliki penyebaran yang sempit dan lainnya menunjukkan kemungkinan adanya outlier. Analisis ini membantu menentukan perlunya transformasi data atau penanganan nilai ekstrim.

---
2. Multivariate Analysis
<p align="center">
  <img src="https://github.com/user-attachments/assets/0710c837-434c-44a0-af84-598276949b4a" alt="Gambar 3. Pairplot Diagnosis terhadap Fitur" width="500"/>
</p>

<p align="center"><strong>Gambar 3.</strong> Pairplot Diagnosis terhadap Fitur</p>

Pairplot memberikan gambaran hubungan antar fitur berdasarkan label diagnosis. Warna berbeda menunjukkan kategori kanker dan non-kanker. Beberapa fitur menunjukkan pemisahan yang cukup jelas antar kelas, yang menjadi indikasi baik bahwa fitur-fitur ini berpotensi kuat untuk pemodelan prediktif.


<p align="center">
  <img src="https://github.com/user-attachments/assets/6dfe8b5c-192b-47e0-9550-6ec230cc42eb" alt="Gambar 4. Heatmap Korelasi Antar Fitur" width="500"/>
</p>

<p align="center"><strong>Gambar 4.</strong> Heatmap Korelasi Antar Fitur</p>

Warna merah menunjukkan korelasi positif tinggi, sedangkan biru menunjukkan korelasi negatif. Terlihat bahwa fitur-fitur seperti `radius_mean`, `perimeter_mean`, dan `area_mean` sangat berkorelasi satu sama lain. Informasi ini berguna dalam pemilihan fitur atau teknik reduksi dimensi.

## Data Preparation
Pada tahap ini, dilakukan beberapa teknik persiapan data agar model machine learning dapat dilatih dengan optimal. Teknik yang digunakan mencakup **Data Splitting** dan **Standardization**, yang dijelaskan secara berurutan sesuai implementasi dalam notebook.

1. Outlier Detection and Handling
<p align="center">
  <img src="https://github.com/user-attachments/assets/39281b8e-2fa5-44df-858d-814c56551339" alt="Gambar 5. Boxplot Sebelum Penanganan Outlier" width="500"/>
</p>

<p align="center"><strong>Gambar 5.</strong> Boxplot Sebelum Penanganan Outlier</p>

Boxplot menunjukkan nilai ekstrim (outlier) pada sebagian besar fitur. Outlier dapat mempengaruhi performa model dan perlu ditangani secara hati-hati.

Setelah proses deteksi dan pembersihan menggunakan metode Interquartile Range (IQR), data menjadi lebih bersih:

<p align="center">
  <img src="https://github.com/user-attachments/assets/65f9ea0c-ebcf-41f7-afd0-5c406a09e5b3" alt="Gambar 6. Boxplot Setelah Penangan Outliers" width="500"/>
</p>

<p align="center"><strong>Gambar 6.</strong> Boxplot Setelah Penanganan Outlier</p>

Fitur-fitur setelah penanganan menunjukkan distribusi yang lebih stabil tanpa banyak nilai pencilan ekstrem.

---
2. Class Imbalance & SMOTE

Distribusi awal kelas target:
- `Kelas 0` (non-kanker): 907 data
- `Kelas 1` (kanker): 377 data
Distribusi ini menunjukkan ketimpangan kelas (class imbalance) yang signifikan. Jika tidak ditangani, model akan cenderung bias terhadap kelas mayoritas.

**Output Distribusi Kelas Setelah SMOTE**

Teknik SMOTE (Synthetic Minority Oversampling Technique) digunakan untuk menyeimbangkan kelas dengan menambahkan data sintetis pada kelas minoritas (kanker).

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ce69ebf-c968-4f30-98e4-653ba124b23c" alt="Gambar 7. Visualisasi Sebelum dan Sesudah SMOTE" width="500"/>
</p>

<p align="center"><strong>Gambar 7.</strong> Visualisasi Sebelum dan Sesudah SMOTE</p>

Gambar ini membandingkan jumlah data antar kelas:
- Sebelum SMOTE: kelas kanker (merah) jauh lebih sedikit dibanding non-kanker (hijau),
- Setelah SMOTE: distribusi menjadi seimbang (907 data per kelas), sehingga model dapat dilatih dengan adil dan tidak bias.


### Data Splitting
Tahap ini bertujuan untuk membagi data menjadi **data pelatihan (training set)** dan **data pengujian (testing set)**. Pemisahan ini penting agar model dapat diuji pada data yang belum pernah dilihat sebelumnya, guna mengevaluasi kemampuan generalisasinya.
- Teknik: `train_test_split` dari `sklearn.model_selection`
- Proporsi: 70% data untuk pelatihan, 30% untuk pengujian
- Alasan:
  - Menghindari overfitting karena model hanya dilatih pada subset data (training set).
  - Memungkinkan evaluasi performa model yang lebih objektif menggunakan testing set.

### Standardization
Standarisasi dilakukan untuk memastikan bahwa seluruh fitur numerik berada dalam skala yang sama. Ini sangat penting terutama untuk algoritma yang sensitif terhadap skala data seperti _K-Nearest Neighbors_, SVM, Logistic Regression, dan lain-lain.
- Teknik: `StandardScaler` dari `sklearn.preprocessing`
- Apa yang dilakukan: Mengubah fitur agar memiliki distribusi dengan **mean = 0** dan **standar deviasi = 1**.
- Alasan:
  - Mempercepat proses konvergensi pada algoritma pembelajaran.
  - Menghindari dominasi fitur dengan nilai besar terhadap model.

## Modeling
Pada tahap ini, dilakukan proses pengembangan model machine learning untuk menyelesaikan permasalahan klasifikasi diagnosis. Beberapa algoritma diuji dan dibandingkan, dimulai dari Extra Trees Classifier, Random Forest, KNN (K-Nearest Neighbor), Support Vector Classifier, dan Naive Bayes. Pemodelan dilakukan secara bertahap, dimulai dari baseline model (tanpa tuning) hingga proses hyperparameter tuning untuk meningkatkan performa model.

### Extra Trees Classifier
**Extra Trees Classifier** (Extremely Randomized Trees) merupakan algoritma *ensemble learning* berbasis pohon keputusan yang bekerja dengan membangun banyak *decision tree* dan menggabungkan hasilnya untuk menghasilkan prediksi yang lebih akurat. Algoritma ini sangat mirip dengan **Random Forest**, namun memiliki tingkat randomisasi yang lebih tinggi dalam proses pemisahan node [[9](https://dspace.uii.ac.id/bitstream/handle/123456789/48182/19522292.pdf?sequence=1&isAllowed=y)]. Perbedaannya terletak pada cara pemilihan *split point*, jika Random Forest mencari pemisahan terbaik berdasarkan kriteria tertentu, Extra Trees memilih titik pemisahan secara acak untuk setiap fitur, lalu memilih yang terbaik dari pilihan acak tersebut. Pendekatan ini membuat proses pelatihan menjadi lebih cepat dan mampu mengurangi *variance*, serta efektif dalam menangani fitur dengan skala berbeda tanpa memerlukan normalisasi data [[10](https://dspace.uii.ac.id/bitstream/handle/123456789/54959/20523191.pdf?sequence=1&isAllowed=y)].

**Tahapan Pemodelan**
1. Baseline Model

   Model awal dibuat menggunakan parameter default dengan sedikit penyesuaian:
   - `n_estimators = 50`: Jumlah pohon yang digunakan dalam ensemble.
   - `max_depth = 5`: Kedalaman maksimum dari setiap pohon.
   - `max_features = 'sqrt'`: Jumlah fitur yang dipertimbangkan saat membagi setiap node.

   Hasil evaluasi model:
   - Akurasi Training: 80.93%
   - Akurasi Testing: 78.90%
   Model ini kemudian disimpan sebagai baseline untuk dibandingkan setelah proses tuning

2. Hyperparameter Tuning

   Untuk meningkatkan performa model, dilakukan pencarian parameter terbaik menggunakan **GridSearchCV** dengan validasi silang 5-fold. Hasil evaluasi model setelah tuning:
   - Akurasi Training: 90.23%
   - Akurasi Testing: 80.00%

3. Perbandingan Kinerja
   | Model                  | Akurasi Training | Akurasi Testing |
   | ---------------------- | ---------------- | --------------- |
   | Extra Trees (Baseline) | 80.93%           | 78.90%          |
   | Extra Trees (Tuned)    | 90.23%           | 80.00%          |
    
   Meskipun akurasi training meningkat secara signifikan setelah tuning (dari 80.93% menjadi 90.23%), akurasi pada data testing hanya meningkat sedikit, dari 78.90% menjadi 80.00%. Hal ini menunjukkan bahwa model hasil tuning cenderung lebih cocok terhadap data pelatihan (potensi overfitting), tetapi tidak memberikan peningkatan besar dalam kemampuan generalisasi terhadap data baru.

4. Analisis Kelebihan & Kekurangan
   | Aspek          | Penjelasan                                                                                                                                                |
   | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | **Kelebihan**  | - Cepat dilatih pada dataset besar.<br>- Tahan terhadap overfitting dibanding pohon tunggal.<br>- Tidak sensitif terhadap outlier dan fitur tidak relevan. |
   | **Kekurangan** | - Interpretabilitas rendah.<br>- Performa bisa kurang stabil jika jumlah pohon terlalu kecil.<br>- Bisa overfitting jika tidak diatur kedalaman pohonnya.  |

5. Kesimpulan Sementara
   
   Model Extra Trees menunjukkan performa yang cukup baik dengan akurasi testing mendekati 80%. Meskipun tuning memberikan peningkatan besar pada akurasi training, dampaknya terhadap data testing tidak signifikan. Oleh karena itu, penting untuk melakukan perbandingan dengan model lain seperti Random Forest atau SVC sebelum memutuskan model terbaik untuk digunakan secara final.


### Random Forest
**Random Forest** adalah algoritma ensemble machine learning berbasis pohon keputusan yang digunakan untuk tugas klasifikasi maupun regresi. Algoritma ini bekerja dengan membangun sejumlah pohon keputusan (decision trees) selama proses pelatihan, lalu menggabungkan hasil prediksi dari setiap pohon. Dalam kasus klasifikasi, prediksi akhir ditentukan berdasarkan voting mayoritas dari seluruh pohon. Pendekatan ini secara efektif mengurangi risiko overfitting yang sering terjadi pada model pohon tunggal serta meningkatkan kemampuan generalisasi model. Dengan menyatukan banyak pohon yang relatif tidak berkorelasi, Random Forest mampu menghasilkan prediksi yang lebih stabil dan akurat [[11](https://hostjournals.com/jimat/article/view/473/289)].

**Tahapan Pemodelan**
1. Baseline Model
   
   Model awal dibuat menggunakan parameter default dengan sedikit penyesuaian:
   - `n_estimators = 50`: Jumlah pohon dalam hutan.
   - `max_depth = 5`: Kedalaman maksimum pohon.
   - `max_features = 'sqrt'`: Fitur yang dipilih secara acak untuk split pada setiap node.

   Hasil evaluasi model:
   - Akurasi Training: 86.76%
   - Akurasi Testing: 81.10%
   Model ini kemudian disimpan sebagai baseline untuk dibandingkan setelah proses tuning

2. Hyperparameter Tuning

   Untuk meningkatkan performa model, dilakukan pencarian parameter terbaik menggunakan **GridSearchCV** dengan validasi silang 5-fold. Hasil evaluasi model setelah tuning:
   - Akurasi Training: 93.46%
   - Akurasi Testing: 84.22%

3. Perbandingan Kinerja
   | Model                    | Akurasi Training | Akurasi Testing |
   | ------------------------ | ---------------- | --------------- |
   | Random Forest (Baseline) | 86.76%           | 81.10%          |
   | Random Forest (Tuned)    | 93.46%           | 84.22%          |

   Setelah dilakukan tuning, akurasi training Random Forest meningkat cukup signifikan dari 86.76% menjadi 93.46%. Akurasi testing juga meningkat dari 81.10% menjadi 84.22%, menandakan bahwa tuning berhasil meningkatkan kemampuan generalisasi model terhadap data baru.


4. Analisis Kelebihan & Kekurangan
   | Aspek          | Penjelasan                                                                                                                                                                                    |
   | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | **Kelebihan**  | - Mengurangi overfitting dibandingkan decision tree tunggal. <br> - Dapat menangani data dengan fitur yang banyak. <br> - Cukup stabil terhadap noise.                                        |
   | **Kekurangan** | - Proses pelatihan dan prediksi relatif lambat untuk dataset besar. <br> - Interpretasi hasil model lebih sulit. <br> - Rentan terhadap overfitting jika parameter tidak diatur dengan tepat. |

5. Kesimpulan Sementara
   
   Model Random Forest menunjukkan performa yang sangat baik dengan akurasi testing mencapai 84.22% setelah dilakukan tuning. Peningkatan akurasi training yang signifikan dari 86.76% menjadi 93.46% menunjukkan model lebih fit terhadap data pelatihan. Namun, peningkatan akurasi testing juga cukup berarti, sehingga tuning berhasil memperbaiki kemampuan generalisasi model. Oleh karena itu, Random Forest menjadi kandidat kuat sebagai model terbaik dibandingkan dengan algoritma lain yang diuji.


### KNN (K-Nearest Neighbor)
**K-Nearest Neighbor (KNN)** adalah algoritma klasifikasi non-parametrik yang bekerja dengan menentukan kelas dari data baru berdasarkan mayoritas label dari K tetangga terdekatnya. Termasuk *lazy learner*, KNN tidak membentuk model selama pelatihan, melainkan menggunakan seluruh data untuk prediksi. Meski sederhana, KNN cukup efektif dan dapat mengungguli model kompleks dalam beberapa kasus, terutama pada dataset kecil. Namun, performanya bisa menurun jika terdapat fitur tidak relevan atau pada data berdimensi tinggi [[12](https://dspace.uii.ac.id/bitstream/handle/123456789/29782/15611077%20Meila%20Ika%20Pradipta.pdf?sequence=1&isAllowed=y)].

**Tahapan Pemodelan**
1. Baseline Model
   
   Model awal dibuat menggunakan parameter default dengan sedikit penyesuaian:
   - `n_neighbors = 5`: Jumlah tetangga terdekat yang digunakan untuk voting kelas.

   Hasil evaluasi model:
   - Akurasi Training: 84.24%
   - Akurasi Testing: 70.64%
   Model ini kemudian disimpan sebagai baseline untuk dibandingkan setelah proses tuning

2. Hyperparameter Tuning

   Untuk meningkatkan performa model, dilakukan pencarian parameter terbaik menggunakan **GridSearchCV** dengan validasi silang 5-fold. Hasil evaluasi model setelah tuning:
   - Akurasi Training: 100.00%
   - Akurasi Testing: 78.72%
     
3. Perbandingan Kinerja
   | Model          | Akurasi Training | Akurasi Testing |
   | -------------- | ---------------- | --------------- |
   | KNN (Baseline) | 84.24%           | 70.64%          |
   | KNN (Tuned)    | 100.00%          | 78.72%          |

   Meskipun akurasi training setelah tuning mencapai 100%, akurasi testing meningkat dari 70.64% menjadi 78.72%. Hal ini menunjukkan bahwa tuning membuat model sangat fit terhadap data pelatihan (kemungkinan overfitting), namun ada peningkatan kemampuan generalisasi yang cukup signifikan pada data testing.

4. Analisis Kelebihan & Kekurangan
   | Aspek          | Penjelasan                                                                                                                                                                          |
   | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | **Kelebihan**  | - Implementasi sederhana. <br> - Tidak memerlukan pelatihan eksplisit. <br> - Cocok untuk data dengan distribusi kompleks.                                                          |
   | **Kekurangan** | - Sensitif terhadap skala data. <br> - Waktu prediksi lambat untuk dataset besar. <br> - Rentan terhadap overfitting, terutama jika `weights='distance'` dan data memiliki outlier. |

5. Kesimpulan Sementara
   
   Model KNN menunjukkan peningkatan akurasi testing yang cukup besar setelah tuning, walaupun model tampak mengalami overfitting dengan akurasi training sempurna. Perbaikan ini menandakan bahwa tuning parameter berhasil meningkatkan performa model pada data testing, namun perlu hati-hati terhadap potensi overfitting yang terjadi. Evaluasi lebih lanjut diperlukan untuk memastikan kestabilan model pada data baru.


### Support Vector Classifier
**Support Vector Classifier (SVC)** merupakan algoritma klasifikasi yang termasuk dalam keluarga Support Vector Machine (SVM). Algoritma ini bekerja dengan mencari hyperplane terbaik yang dapat memisahkan kelas-kelas dalam ruang berdimensi tinggi. SVC efektif digunakan untuk masalah klasifikasi linier maupun non-linier, dan dikenal mampu menghasilkan margin klasifikasi maksimum. Dengan kernel trick, SVC dapat menangani data non-linier dengan memetakan data ke ruang fitur berdimensi lebih tinggi [[13](https://scikit-learn.org/stable/modules/svm.html)].

**Tahapan Pemodelan**
1. Baseline Model
   
   Model awal dibuat menggunakan parameter default dengan sedikit penyesuaian:
   - `kernel='rbf'`: Menggunakan Radial Basis Function sebagai fungsi kernel.
   - `C=1.0`: Parameter regulasi default.
   - `random_state=42`: Untuk memastikan reprodusibilitas hasil.

   Hasil evaluasi model:
   - Akurasi Training: 68.24%
   - Akurasi Testing: 68.26%
   Model ini kemudian disimpan sebagai baseline untuk dibandingkan setelah proses tuning

2. Hyperparameter Tuning

   Untuk meningkatkan performa model, dilakukan pencarian parameter terbaik menggunakan **GridSearchCV** dengan validasi silang 5-fold. Hasil evaluasi model setelah tuning:
   - Akurasi Training: 100.00%
   - Akurasi Testing: 81.10%

3. Perbandingan Kinerja
   | Model          | Akurasi Training  | Akurasi Testing |
   | -------------- | ----------------- | --------------- |
   | SVC (Baseline) | 68.24%            | 68.26%          |
   | SVC (Tuned)    | 100.00%           | 81.10%          |

   Setelah tuning, akurasi training naik drastis hingga 100%, menandakan model sangat fit terhadap data pelatihan (overfitting). Namun, akurasi testing juga meningkat signifikan dari 68.26% menjadi 81.10%, menunjukkan bahwa tuning berhasil meningkatkan kemampuan generalisasi model.


4. Analisis Kelebihan & Kekurangan                                                                                                                                      
   | Aspek          | Penjelasan                                                                                                                                                          |
   | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | **Kelebihan**  | - Efektif untuk data berdimensi tinggi. <br> - Mampu menangani klasifikasi linier dan non-linier. <br> - Mendukung berbagai fungsi kernel.                          |
   | **Kekurangan** | - Sensitif terhadap pemilihan parameter `C`, `kernel`, dan `gamma`. <br> - Waktu komputasi tinggi pada dataset besar. <br> - Kurang transparan secara interpretasi. |

5. Kesimpulan Sementara
   
   Model SVC mengalami peningkatan performa yang paling besar setelah dilakukan tuning hyperparameter. Meskipun ada indikasi overfitting dengan akurasi training sempurna, peningkatan akurasi testing yang cukup signifikan menunjukkan model ini memiliki potensi terbaik untuk prediksi pada data baru. Evaluasi lebih lanjut disarankan untuk mengoptimalkan keseimbangan antara bias dan variance.


### Naive Bayes
**Naive Bayes** adalah algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi independensi antar fitur. Model ini menghitung probabilitas setiap kelas berdasarkan distribusi fitur dan memilih kelas dengan probabilitas tertinggi. Salah satu variannya, Gaussian Naive Bayes, digunakan ketika fitur mengikuti distribusi normal. Naive Bayes dikenal cepat dan efisien, terutama untuk data berdimensi tinggi dan masalah klasifikasi teks, meskipun performanya bisa menurun ketika asumsi independensi tidak terpenuhi [[14](https://scikit-learn.org/stable/modules/naive_bayes.html)].

**Tahapan Pemodelan**
1. Baseline Model
   
   Model awal dibangun dengan parameter default dari `GaussianNB`. Tidak ada parameter khusus yang disetel pada tahap awal.

   Hasil evaluasi model:
   - Akurasi Training: 78.01%
   - Akurasi Testing: 76.88%
   Model ini kemudian disimpan sebagai baseline untuk dibandingkan setelah proses tuning

2. Hyperparameter Tuning

   Untuk menyempurnakan performa, dilakukan pencarian nilai optimal dari parameter `var_smoothing`. Hasil evaluasi model setelah tuning:
   - Akurasi Training: 78.09%
   - Akurasi Testing: 77.25%

3. Perbandingan Kinerja
   | Model                  | Akurasi Training | Akurasi Testing |
   | ---------------------- | ---------------- | --------------- |
   | Naive Bayes (Baseline) | 78.01%           | 76.88%          |
   | Naive Bayes (Tuned)    | 78.09%           | 77.25%          |

   Setelah tuning, akurasi training dan testing hanya mengalami peningkatan yang sangat kecil, hampir tidak signifikan. Hal ini mengindikasikan bahwa parameter default model Naive Bayes sudah cukup optimal dan tuning tidak memberikan pengaruh besar terhadap performa model.


4. Analisis Kelebihan & Kekurangan                                                                                                                                      
   | Aspek          | Penjelasan                                                                                                                                             |
   | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | **Kelebihan**  | - Cepat dan efisien. <br> - Performa baik untuk data berdimensi tinggi. <br> - Cocok untuk klasifikasi teks dan data yang mendekati distribusi normal. |
   | **Kekurangan** | - Asumsi independensi fitur jarang terpenuhi dalam data nyata. <br> - Kurang fleksibel dalam menangani fitur yang berkorelasi.                         |

5. Kesimpulan Sementara
   
   Model Naive Bayes menunjukkan stabilitas performa dengan akurasi sekitar 77% pada data testing. Karena peningkatan setelah tuning sangat minimal, model ini mungkin sudah berada pada konfigurasi optimalnya dengan parameter default. Untuk peningkatan performa lebih lanjut, bisa dipertimbangkan pendekatan lain seperti fitur engineering atau penggunaan model berbeda.

  
## Evaluation
Proyek ini berfokus pada _predictive analytics_ untuk kasus klasifikasi, **metrik utama yang digunakan adalah akurasi**. Akurasi didefinisikan sebagai:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large&space;\text{Accuracy}&space;=&space;\frac{\text{Jumlah&space;prediksi&space;benar}}{\text{Total&space;seluruh&space;prediksi}}" title="Accuracy Formula" />
</p>

Akurasi merupakan metrik yang cocok digunakan ketika distribusi kelas cukup seimbang dan tujuan utamanya adalah mengukur seberapa sering model memberikan prediksi yang benar. Dalam konteks ini (prediksi penyakit kanker), distribusi kelas sudah diseimbangkan dengan metode oversampling (SMOTE), sehingga akurasi menjadi metrik yang relevan dan dapat diandalkan.

### Hasil Evaluasi Model
Evaluasi dilakukan terhadap lima model klasifikasi baik sebelum maupun sesudah proses hyperparameter tuning, dan hasilnya dirangkum dalam bentuk tabel dan grafik berikut:

**Tabel Perbandingan Akurasi**
| Model         | Accuracy Before Tuning | Accuracy After Tuning |
| ------------- | ---------------------- | --------------------- |
| Extra Trees   | 78.90%                 | 80.00%                |
| Random Forest | 81.10%                 | 84.22%                |
| KNN           | 70.64%                 | 78.72%                |
| SVC           | 68.26%                 | 81.10%                |
| Naive Bayes   | 76.88%                 | 77.25%                |

**Visualisasi Grafik Akurasi**

Grafik di bawah ini menunjukkan perbandingan akurasi dari setiap model sebelum dan sesudah tuning:

<p align="center">
  <img src="https://github.com/user-attachments/assets/55b702e2-7286-41a2-bf65-be6ee8e0c055" alt="Gambar 8. Model Accuracy Chart" width="500"/>
</p>

<p align="center"><strong>Gambar 8.</strong> Model Accuracy Chart</p>

Dari tabel akurasi dan grafik, terlihat bahwa hampir semua model mengalami peningkatan akurasi setelah tuning, meskipun tingkat peningkatannya bervariasi.

### Analisis dan Interpretasi
- Model dengan peningkatan terbesar setelah tuning:
  - **SVC (Support Vector Classifier)** meningkat dari 68.26% menjadi 81.10% (+12.84 poin)
  - **KNN** meningkat dari 70.64% menjadi 78.72% (+8.08 poin)

- Model dengan performa tertinggi setelah tuning:
  - **Random Forest** dengan akurasi tertinggi sebesar 84.22%
  - **SVC** sebagai alternatif kuat dengan akurasi 81.10%

- Model dengan peningkatan kecil setelah tuning:
  - **Extra Trees** meningkat dari 78.90% menjadi 80.00%, menunjukkan kestabilan dan performa yang baik.
  - **Naive Bayes** sedikit mengalami peningkatan dari 76.88% menjadi 77.25%, menunjukkan model yang relatif stabil dan tidak banyak terpengaruh tuning.

### Kesimpulan Akhir
Berdasarkan hasil evaluasi:
- **Random Forest** menjadi **model terbaik** dalam proyek ini dengan akurasi tertinggi 84.22% setelah tuning, menunjukkan kehandalan model ini dalam menangani dataset.
- **SVC** juga menunjukkan peningkatan signifikan dan performa yang baik, menjadikannya alternatif kuat kedua dengan akurasi **81.10%**.
- **KNN** memberikan peningkatan cukup besar dan bisa menjadi opsi tambahan dengan akurasi **78.72%**.
- **Extra Trees** dan **Naive Bayes** menunjukkan kestabilan dengan peningkatan yang relatif kecil, sehingga cocok untuk kasus di mana kestabilan dan kecepatan menjadi pertimbangan.

Secara keseluruhan, **Random Forest** direkomendasikan sebagai model klasifikasi utama karena akurasinya yang tertinggi dan peningkatan performa yang signifikan setelah tuning. Namun, **SVC** dan **KNN** juga dapat menjadi pilihan yang baik sesuai kebutuhan dan konteks penggunaan.

## Referensi

**[1]** L. Hafsah, “Gambaran tingkat kecemasan pada pasien kanker yang menjalani kemoterapi di RSUD Dr. M. Yunus Bengkulu,” *J. Vokasi Keperawatan (JVK)*, vol. 5, no. 1, pp. 21–28, 2022.  

**[2]** UGM, “Jumlah penderita kanker terus meningkat, kenali gejala awal untuk deteksi dini,” *Universitas Gadjah Mada*, 2023. [Online]. Available: https://ugm.ac.id/id/berita/jumlah-penderita-kanker-terus-meningkat-kenali-gejala-awal-untuk-deteksi-dini/. [Accessed: May 13, 2024].  

**[3]** M. T. T. B. Sirait, N. S. Fathonah, and M. N. Fauzan, “Pemanfaatan algoritma ADASYN dan support vector machine dalam meningkatkan akurasi prediksi kanker paru-paru,” *JATI (J. Mahasiswa Tek. Inform.)*, vol. 8, no. 5, pp. 8773–8778, 2024.  

**[4]** I. P. Putri, T. Terttiaavini, and N. Arminarahmah, "Comparative analysis of machine learning algorithms for predicting child stunting," *MALCOM Indones. J. Mach. Learn. Comput. Sci.*, vol. 4, pp. 257–265, 2024.

**[5]** D. P. Sinambela, H. Naparin, M. Zulfadhilah, and N. Hidayah, "Implementasi algoritma decision tree dan random forest dalam prediksi perdarahan pascasalin," *J. Inform. dan Teknol.*, pp. 58–64, 2023.

**[6]** H. Tuhuteru and A. Iriani, "Analisis sentimen Perusahaan Listrik Negara Cabang Ambon menggunakan metode support vector machine dan naive Bayes classifier," *J. Informatika: J. Pengembangan IT*, vol. 3, no. 3, pp. 394–401, 2018.

**[7]** M. Syawaludin and M. Khulaimi, "Perancangan sistem pakar prediksi diagnosis penyakit diabetes menggunakan algoritma naive Bayes berbasis web," *Jikom: J. Informatika dan Komputer*, vol. 15, no. 1, pp. 161–171, 2025.

**[8]** K. Handayani, E. Erni, B. Lailiah, and R. Sa'adah, "Klasifikasi kanker payudara menggunakan extra tree dengan SMOTE," *JATI (J. Mahasiswa Tek. Inform.)*, vol. 7, no. 6, pp. 3100–3105, 2023.

**[9]** M. E. M. Zees, *Implementasi Autogluon dalam efisiensi model prediktif machine learning pada dataset International Business Machines (IBM) Human Resource (HR) Analytics Employee Attrition*, Doctoral dissertation, Universitas Islam Indonesia, 2023.

**[10]** R. A. Effendy, *Eksplorasi algoritma tree based model untuk kasus tipe kepribadian dengan Myers Briggs Type Indicator (MBTI)*, Doctoral dissertation, Universitas Islam Indonesia, 2025.

**[11]** M. Fadhilla, R. Wandri, A. Hanafiah, P. R. Setiawan, Y. Arta, and S. Daulay, "Analisis performa algoritma machine learning untuk identifikasi depresi pada mahasiswa," *J. Inform. Manag. dan Inf. Technol.*, vol. 5, no. 1, pp. 40–47, 2025.

**[12]** A. Kesumawati, *Klasifikasi curah hujan menggunakan metode ensemble subset K-nearest neighbor (Studi kasus: Curah hujan Kota Bogor Tahun 2014–2018)*, 2020.

**[13]** Scikit-learn, "Support vector machines," *scikit-learn*, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/svm.html. [Accessed: May 13, 2024].

**[14]** Scikit-learn, "Naive Bayes," *scikit-learn*, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/naive_bayes.html. [Accessed: May 13, 2024].
