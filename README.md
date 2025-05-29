# Laporan Proyek Machine Learning - M Wildan Nurohman

## Domain Proyek
Domain yang dipilih untuk proyek _machine learning_ ini adalah **Ekonomi dan Bisnis**, dengan judul **Predictive Analytics: Prediksi Biaya Asuransi Kesehatan**.

### Latar Belakang
<img src="https://github.com/user-attachments/assets/93548ca5-34e4-4726-992b-97276e06e509" alt="Contoh Gambar" style="width:100%; height:auto;">

Kesehatan merupakan salah satu aspek paling krusial dalam kehidupan manusia modern. Dengan meningkatnya biaya perawatan medis dari tahun ke tahun, banyak individu dan keluarga berupaya melindungi diri mereka melalui asuransi kesehatan. Salah satu tantangan utama bagi perusahaan asuransi adalah bagaimana menentukan premi atau biaya asuransi yang adil dan akurat bagi tiap individu berdasarkan karakteristik pribadinya seperti usia, status merokok, indeks massa tubuh (BMI), dan lainnya[[1](https://www.researchgate.net/publication/383522368_Health_Insurance_Factor_Analysis)].

Prediksi biaya asuransi berbasis data dapat membantu perusahaan asuransi menyusun penawaran premi yang lebih tepat dan transparan, serta membantu calon nasabah memahami faktor apa saja yang memengaruhi besaran premi yang dibebankan. Analisis prediktif menggunakan teknik machine learning telah terbukti efektif dalam mengidentifikasi pola dan hubungan non-linear antar variabel dalam sistem kompleks seperti asuransi kesehatan.
Studi oleh Billa dan Nagpal (2024) menunjukkan bahwa penerapan teknik machine learning, termasuk regresi linier dan random forest, dapat secara efektif memprediksi premi asuransi kesehatan berdasarkan data demografis dan gaya hidup individu. Model-model ini tidak hanya meningkatkan akurasi prediksi tetapi juga membantu dalam pengambilan keputusan yang lebih baik oleh pemangku kepentingan di sektor kesehatan dan asuransi[[2](https://journal.esrgroups.org/jes/article/download/3962/3065/7574)].


## Business Understanding
Perusahaan asuransi kesehatan sering kali menghadapi kesulitan dalam menetapkan premi yang mencerminkan risiko individu secara akurat. Ketidakakuratan dalam penetapan premi dapat menyebabkan kerugian finansial bagi perusahaan atau ketidakadilan bagi pelanggan. Dengan memanfaatkan data historis dan teknik machine learning, perusahaan dapat mengembangkan model prediktif yang mempertimbangkan berbagai faktor individu untuk menentukan biaya asuransi yang lebih akurat dan adil.

### Problem Statements
Berdasarkan latar belakang di atas, maka permasalahan yang akan dijawab dalam proyek ini adalah:
1. Bagaimana memprediksi biaya asuransi kesehatan (charges) seseorang secara akurat berdasarkan informasi demografis dan gaya hidupnya seperti usia, indeks massa tubuh (BMI), status merokok, dan wilayah tempat tinggal?
2. Faktor mana yang paling signifikan dalam mempengaruhi besarnya biaya asuransi medis berdasarkan data historis?
3. Bagaimana pemodelan machine learning dapat membantu perusahaan asuransi dalam menetapkan premi yang lebih adil dan proporsional terhadap risiko individu?

### Goals
Proyek ini memiliki tujuan sebagai berikut:
1. Mengembangkan model prediksi biaya asuransi kesehatan berdasarkan fitur-fitur demografis dan gaya hidup seperti usia, BMI, status merokok, dan wilayah tempat tinggal..
2. Mengidentifikasi dan menganalisis faktor-faktor yang paling berpengaruh terhadap biaya asuransi medis melalui teknik eksplorasi data dan feature importance dari model machine learning.
3. Menyediakan wawasan berbasis data yang dapat membantu perusahaan asuransi dalam merancang struktur premi yang lebih adil dan proporsional terhadap risiko masing-masing individu.

### Solution Statements
Untuk mencapai tujuan proyek ini, langkah-langkah yang dilakukan meliputi:

- **Eksplorasi dan Analisis Data**  
  Melakukan analisis univariat dan multivariat terhadap fitur-fitur dalam dataset untuk memahami distribusi data, mengidentifikasi korelasi antar fitur, serta mendeteksi outlier yang berpotensi memengaruhi kinerja model.

- **Preprocessing Data**  
  Melakukan pembersihan data (handling missing values dan duplikasi) dan normalisasi agar data siap digunakan dalam proses pelatihan model machine learning sehingga dapat menghasilkan prediksi yang optimal.

- **Pembangunan dan Evaluasi Model**  
  Membangun dan membandingkan beberapa algoritma machine learning untuk menentukan model terbaik dalam melakukan prediksi. Model yang digunakan antara lain:
  
  
  - **Random Forest**: Random Forest adalah algoritma ensemble yang terdiri dari banyak pohon keputusan (_decision trees_) yang dibangun dari subset acak data dan fitur. Setiap pohon membuat prediksi secara mandiri, kemudian hasil akhirnya ditentukan melalui voting mayoritas. Pendekatan ini meningkatkan akurasi dan kestabilan model serta mampu mengurangi risiko overfitting yang kerap terjadi pada model _decision tree_ tunggal [[3](https://jidt.org/jidt/article/view/393/205)].
  - **Linear Regression**: Linear Regression adalah teknik analisis data yang memprediksi nilai data yang tidak diketahui dengan menggunakan nilai data lain yang terkait dan diketahui. Secara matematis memodelkan variabel yang tidak diketahui atau tergantung dan variabel yang dikenal atau independen sebagai persamaan linier. [[4](https://aws.amazon.com/id/what-is/linear-regression/)].
## Data Understanding
### Gathering Data
Dataset yang digunakan dalam proyek ini adalah "Medical Cost Personal Datasets", yang berisi informasi mengenai karakteristik pasien dan biaya yang dikeluarkan untuk pengobatan
Proses pengumpulan data dilakukan melalui tiga langkah utama:
1. Mendownload dataset dari yang masih berupa zip di kaggle
2. Mengekstrak isi file ZIP untuk mendapatkan file CSV secara lokal
3. Mengupload file CSV ke google colab
4. Membaca file CSV ke dalam bentuk DataFrame agar dapat dianalisis lebih lanjut.

Dataset ini berisi 1.338 baris dan 7 fitur yang merepresentasikan karakteristik pasien, seperti : umur pasien, jenis kelamin, kategori berat badan pasien, jumlah tanggungan anak, status kebiasaan merokok, wilayah tempat tinggal, dan niaya tanggungan asuransi
### Data Assesing and Data Cleaning
Informasi Datasets

| Jenis       | Keterangan                                                                                                                                      |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Title**   | 👨‍⚕️💉 Medical Cost Personal Datasets 💵💰                                                                                                               |
| **Source**  | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)                                                              |
| **Maintainer** | [Miri Choi](https://www.kaggle.com/mirichoi0218)                                                                                     |
| **License** | [Database Content License](https://opendatacommons.org/licenses/dbcl/1-0/)                                                                                                                   |
| **Visibility** | Publik                                                                                                                                       |
| **Tags**    | Education, Health, Insurance, Finance, Healthcare                                                                    |
| **Usability** | 8.82                                                                                                                                         |

> Klik pada nama **Kaggle** untuk langsung menuju ke dataset.
> Klik pada nama **Miri Choi** untuk langsung menuju ke user pemilik datasey

**Dataset Info**
- Format: CSV (Comma-Separated Values)
- Jumlah data: **1.338 baris** dan **7 kolom**
- Semua kolom memiliki **1338 nilai non-null** (tidak ada missing values)
- Tipe data:
  - `int64` (2 kolom): untuk data diskrit (contoh: usia dan jumlah tanggungan anak)
  - `float64` (2 kolom): untuk data kontinu (contoh: BMI dan jumlah uang yg dikeluarkan untuk biaya pengobatan)
  - `object` (3 kolom): untuk data yang berupa string (contoh: wilayah, status merokok, dan jenis kelamin)

Contoh data:
| age | sex | bmi       | children | smoker | region | charges |
|-----|--------|-----------|---------|--------------|------------------|----------------|
| 19  | female      | 27.900     | 1       | yes            | southwest             | 16884.92400           |
| 18  | male      | 33.770     | 0      | no            | southeast             | 1725.55230           |
| 28  | male      | 33.000     | 3       | yes            | northwest             | 4449.46200           |

---

**Feature Explanation**

| Fitur      | Tipe        | Deskripsi                                          |
| ---------- | ----------- | -------------------------------------------------- |
| `age`      | numerik     | Usia individu (dalam tahun)                        |
| `sex`      | kategorikal | Jenis kelamin: male / female                       |
| `bmi`      | numerik     | Body Mass Index                                    |
| `children` | numerik     | Jumlah anak tanggungan                             |
| `smoker`   | kategorikal | Status merokok: yes / no                           |
| `region`   | kategorikal | Wilayah tempat tinggal: southeast, northwest, dll. |
| `charges`  | numerik     | Biaya asuransi medis (target prediksi)             |

### Exploratory Data Analysis (EDA)
Tahap *Exploratory Data Analysis (EDA)* dilakukan untuk memahami karakteristik data secara menyeluruh sebelum memasuki proses *data preprocessing* dan pemodelan. Proses ini bertujuan untuk:

- Mengetahui distribusi data dan proporsi target.
- Mengidentifikasi hubungan antar fitur.
- Mendeteksi nilai pencilan (*outliers*).
- Mengatasi ketidakseimbangan kelas pada label target.

---
1. Univariate Analysis  
<p align="center">
  <img src="https://github.com/wildannrr/health-insurance_predictive-analytics/blob/main/assets/perokok%20vs%20non_countp.png" alt="Gambar 1. Perokok vs Non-Perokok" width="500"/>
</p>

<p align="center"><strong>Gambar 1.</strong> Perokok vs Non-perokok</p>

CountPlot di atas menunjukkan proporsi pasien yang memiliki kebiasaan merokok (`yes`) dan non-perokok (`no`). Sebanyak <strong> 274 pasien (20.5%)</strong> memiliki kebiasaan merokok, sedangkan <strong>1064 pasien (79.5)</strong> tidak 
memiliki kebiasaan merokok.

| Status Perokok | Jumlah Data |
| -------------- | ----------- |
| no             | 1064        |
| yes            | 274         |
| **Total**      | **1338**    |


<p align="center">
  <img src="https://github.com/wildannrr/health-insurance_predictive-analytics/blob/main/assets/univariate_hist.png" alt="Gambar 2. Histogram Fitur Numerik" width="500"/>
</p>

<p align="center"><strong>Gambar 2.</strong> Histogram Fitur Numerik</p>

Age : 
- Distribusi usia menunjukkan puncak yang signifikan di kisaran 18-20 tahun, diikuti oleh penurunan bertahap hingga usia 60-an. Ini menunjukkan bahwa dataset memiliki lebih banyak individu muda (18-30 tahun) dibandingkan kelompok usia yang lebih tua, dengan distribusi yang agak miring ke kiri.

BMI :
- Distribusi BMI mendekati normal, dengan puncak utama di kisaran 30-35. Ada ekor panjang ke arah nilai BMI yang lebih tinggi (>45), yang mengindikasikan adanya beberapa outlier atau individu dengan BMI ekstrem.

Children :
- Distribusi sangat miring ke kiri, dengan mayoritas individu (sekitar 500-600 orang) tidak memiliki anak (0). Jumlah individu dengan 1, 2, atau 3 anak menurun secara bertahap, dan sangat sedikit yang memiliki 4 atau 5 anak, menunjukkan bahwa fitur ini memiliki variasi terbatas.

<p align="center">
  <img src="https://github.com/wildannrr/health-insurance_predictive-analytics/blob/main/assets/target_hist.png" alt="Gambar 3. Analisis Distribusi Fitur Charges" width="500"/>
</p>

<p align="center"><strong>Gambar 3.</strong> Analisis Distribusi Fitur Charges</p>


Distribusi sangat miring ke kanan (right-skewed), dengan puncak utama di kisaran 0-10,000. Sebagian besar biaya asuransi berada di bawah $20,000, tetapi ada ekor panjang hingga $60,000, menunjukkan adanya nilai ekstrem (misalnya, biaya tinggi untuk perokok atau individu dengan kondisi kesehatan serius).  

---
2. Multivariate Analysis
<p align="center">
  <img src="https://github.com/wildannrr/health-insurance_predictive-analytics/blob/main/assets/catt-charges_boxp.png" alt="Gambar 4. Box Plot" width="500"/>
</p>

<p align="center"><strong>Gambar 4.</strong> Boxplot Analisis kategorikal terhadap fitur charges</p>

#### **Charges vs Smoker** :
- Perokok (yes) memiliki biaya asuransi yang jauh lebih tinggi dibandingkan non-perokok.

- Median charges untuk perokok berada di sekitar 35.000, sementara non-perokok hanya sekitar 8.000.

- Distribusi biaya untuk perokok juga lebih menyebar (varians besar), dengan nilai maksimum menyentuh lebih dari 60.000.

**_Insights_** :

Merokok adalah faktor risiko utama yang menyebabkan peningkatan drastis pada biaya asuransi kesehatan. Hal ini menunjukkan bahwa status merokok merupakan predictor yang sangat kuat dan penting untuk model prediksi.

#### **Charges vs Sex** :
- Median biaya antara pria dan wanita hampir sama (sekitar 10.000–12.000).
- Terdapat outlier pada kedua kelompok, tetapi distribusi nilai charges relatif seimbang.

**_Insights_** : 

Jenis kelamin tidak terlalu berpengaruh signifikan terhadap biaya asuransi. Artinya, variabel sex mungkin memiliki kontribusi rendah dalam model prediksi.

#### **Charges vs Region** :
- Median biaya asuransi hampir sama di semua wilayah, meskipun southeast sedikit lebih tinggi.
- Semua wilayah memiliki banyak outlier, tapi tidak ada pola perbedaan yang sangat jelas antar region. 

**_Insights_** : 

Wilayah geografis hanya memiliki pengaruh kecil terhadap perbedaan biaya asuransi. Namun, wilayah southeast menunjukkan sedikit kecenderungan untuk memiliki biaya lebih tinggi, mungkin karena faktor demografis atau pola hidup.


<p align="center">
  <img src="https://github.com/wildannrr/health-insurance_predictive-analytics/blob/main/assets/corr_heatmap.png" alt="Gambar 5. Heatmap Korelasi Antar Fitur" width="500"/>
</p>

<p align="center"><strong>Gambar 5.</strong> Heatmap Korelasi Antar Fitur</p>

Variabel age, bmi, dan children memiliki korelasi positif terhadap charges, tapi age dan bmi lebih dominan. Ini artinya, makin tua dan makin tinggi BMI seseorang, makin mahal potensi biaya asuransinya.

## Data Preparation
Pada proses Data Preparation dilakukan beberapa teknik persiapan data agar model machine learning dapat dilatih dengan optimal. Teknik yang digunakan mencakup Data Encoding Splitting dan Standardization, yang dijelaskan secara berurutan sesuai implementasi dalam notebook. Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
  - Duplicate data (data yang serupa dengan data lainnya) = Terdapat kolom yang memiliki duplikasi data.
  - Missing value (data atau informasi yang "hilang" atau tidak tersedia) = Tidak ada Missing Value
  - Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada) = Dipertahankan karena biaya asuransi mahal bisa wajar, terutama pada perokok dan usia lanjut



1. Cek data duplikat
### 🔍 Cek dan Hapus Data Duplikat

```python
# Cek jumlah duplikat
duplicate_rows = df[df.duplicated()]
print(f"Jumlah duplikat: {duplicate_rows.shape[0]}")

# Tampilkan baris duplikat
print(duplicate_rows)

# Hapus duplikat
df = df.drop_duplicates()

# Verifikasi kembali
print(f"Jumlah data setelah menghapus duplikat: {df.shape[0]}")
```
Output : 

![image](https://github.com/user-attachments/assets/48aa9215-bd91-49fb-9940-8ae5bf72456b)

2. Cek Missing Value

![image](https://github.com/user-attachments/assets/53a81080-f8ed-41a4-bcf0-50b4f3849d5f)

Tidak terdapat missing value (NaN/NULL) pada dataset ini.

### Encoding Data Kategorikal

```python
# Encoding fitur kategorikal
encoded_df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Menampilkan beberapa kolom awal
print(encoded_df.head())
```
Penjelasan :
- drop_first=True menghindari dummy variable trap.
- Kolom sex akan jadi sex_male (0: female, 1: male)
- Kolom smoker akan jadi smoker_yes (0: false, 1: true)
- Kolom region akan jadi region_northwest, region_southeast, region_southwest (tanpa northeast karena di-drop)

### Standardization
Standarisasi dilakukan untuk memastikan bahwa seluruh fitur numerik berada dalam skala yang sama (age, bmi, children).
```python
# Fitur numerik
num_features = ['age', 'bmi', 'children']

# Inisialisasi scaler
scaler = StandardScaler()

# Simpan hasil scaling di DataFrame baru
encoded_df[num_features] = scaler.fit_transform(encoded_df[num_features])
```

- Teknik: `StandardScaler` dari `sklearn.preprocessing`
- Apa yang dilakukan: Mengubah fitur agar memiliki distribusi dengan **mean = 0** dan **standar deviasi = 1**.
- Alasan:
  - Mempercepat proses konvergensi pada algoritma pembelajaran.
  - Menghindari dominasi fitur dengan nilai besar terhadap model.


### Data Splitting
Tahap ini bertujuan untuk membagi data menjadi **data pelatihan (training set)** dan **data pengujian (testing set)**. Pemisahan ini penting agar model dapat diuji pada data yang belum pernah dilihat sebelumnya, guna mengevaluasi kemampuan generalisasinya.
```python
# Split Dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Teknik: `train_test_split` dari `sklearn.model_selection`
- Proporsi: 80% data untuk pelatihan, 20% untuk pengujian
- Alasan:
  - Menghindari overfitting karena model hanya dilatih pada subset data (training set).
  - Memungkinkan evaluasi performa model yang lebih objektif menggunakan testing set.



## Modeling
Bagian ini bertujuan untuk membangun model machine learning menggunakan algoritma yang sesuai dan meningkatkan kinerjanya melalui proses tuning. Model dikembangkan, dievaluasi, dan disempurnakan dengan memilih kombinasi parameter terbaik menggunakan teknik seperti Grid Search.

### Linear Regression
Linear Regression adalah algoritma yang menyediakan hubungan linier antara variabel independen dan variabel dependen untuk memprediksi hasil kejadian di masa mendatang. Ini adalah metode statistik yang digunakan dalam ilmu data dan pembelajaran mesin untuk analisis prediktif.
Variabel bebas juga merupakan variabel prediktor atau penjelas yang tetap tidak berubah karena perubahan variabel lain. Namun, variabel dependen berubah seiring dengan fluktuasi variabel bebas. Model regresi memprediksi nilai variabel dependen, yang merupakan variabel respons atau hasil yang dianalisis atau dipelajari.[[4](https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-linear-regression/)]

**Tahapan Pemodelan**
  - `lr_model = LinearRegression()`: Inisialisasi model
  - `lr_model.fit(X_train, y_train)`: Melatih model dengan data pelatihan (X_train, y_train) agar model belajar hubungan antara fitur dan target 
  - `y_pred = lr_model.predict(X_test)`: Prediksi model
    
```
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

- MAE (Mean Absolute Error): Rata-rata selisih absolut antara nilai aktual dan prediksi.
- RMSE (Root Mean Squared Error): Akar dari rata-rata kuadrat selisih prediksi.
- R² (R-squared): Mengukur seberapa baik model menjelaskan variasi data. Semakin mendekati 1, semakin baik model.

Performance : 
- MAE  : 4177.05
- RMSE : 5956.34
- R²   : 0.8069

Model ini menunjukkan hasil yang cukup baik di hubungan antar fitur bersifat linier. Namun, performanya bisa menurun bila ada non-linearitas atau interaksi kompleks antar fitur (pengaruh besar dari perokok terhadap biaya asuransi).

### Random Forest
**Random Forest** adalah algoritma ensemble machine learning berbasis pohon keputusan yang digunakan untuk tugas klasifikasi maupun regresi. Algoritma ini bekerja dengan membangun sejumlah pohon keputusan (decision trees) selama proses pelatihan, lalu menggabungkan hasil prediksi dari setiap pohon. Dalam kasus klasifikasi, prediksi akhir ditentukan berdasarkan voting mayoritas dari seluruh pohon. Pendekatan ini secara efektif mengurangi risiko overfitting yang sering terjadi pada model pohon tunggal serta meningkatkan kemampuan generalisasi model. Dengan menyatukan banyak pohon yang relatif tidak berkorelasi, Random Forest mampu menghasilkan prediksi yang lebih stabil dan akurat [[11](https://hostjournals.com/jimat/article/view/473/289)].

**Tahapan Pemodelan**
1. Baseline Model
   
   Model awal dibuat menggunakan parameter default dengan sedikit penyesuaian:
   - `n_estimators = 100`: Jumlah pohon.
   - `random_state = 42`: Untuk reproducibilitty.
   - `n_jobs = -1`: Menggunakan semua core CPU

  Latih model dengan data training dan prediksi menggunakan data test
  ```
  rf_model.fit(X_train, y_train)
  y_pred_rf = rf_model.predict(X_test)
  ```
2. Hyperparameter Tuning

  ### 🔧 Hyperparameter Tuning dengan GridSearchCV

```python
# 5. Hyperparameter Tuning
print("\n=== Hyperparameter Tuning ===")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search dengan Cross Validation
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Melatih model dengan data training
rf_grid.fit(X_train, y_train)

# Menyimpan model terbaik dari hasil tuning
best_rf_model = rf_grid.best_estimator_ 
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV R² score: {rf_grid.best_score_:.4f}")
```
Output: 
![image](https://github.com/user-attachments/assets/1b1d637c-9725-49d7-a2b0-d0153f64370a)


Penjelasan :

- param_grid: kombinasi hyperparameter yang akan diuji.
- n_estimators: jumlah pohon dalam Random Forest.
- max_depth: kedalaman maksimum pohon. None artinya tak terbatas.
- min_samples_split: minimal jumlah sampel untuk membagi node.
- min_samples_leaf: minimal jumlah sampel di leaf node.

Total kombinasi 3 x 3 x 3 x 3 = 81 Kombinasi


- GridSearchCV: digunakan untuk menemukan kombinasi parameter terbaik.
- cv=5: menggunakan 5-fold cross-validation.
- scoring='r2': evaluasi berdasarkan R² Score.
- n_jobs=-1: gunakan seluruh core CPU untuk paralelisasi.
- verbose=1: mencetak progress pencarian grid.



  
## Evaluation

Akurasi merupakan metrik yang cocok digunakan ketika distribusi kelas cukup seimbang dan tujuan utamanya adalah mengukur seberapa sering model memberikan prediksi yang benar. Dalam konteks ini (prediksi penyakit kanker), distribusi kelas sudah diseimbangkan dengan metode oversampling (SMOTE), sehingga akurasi menjadi metrik yang relevan dan dapat diandalkan.

### Hasil Evaluasi
| Model             | MAE     | RMSE    | R² Score |
| ----------------- | ------- | ------- | -------- |
| Linear Regression | 4177.05 | 5956.34 | 0.8069   |
| Random Forest     | 2461.34 | 4362.76 | 0.8964   |

- Random Forest unggul dalam semua metrik, terutama R² Score yang meningkat lebih dari 11% dibanding Linear Regression.
- Hal ini menandakan bahwa Random Forest mampu menangkap kompleksitas data dan hubungan non-linier lebih baik.
  
**Visualisasi Features Importance**

Grafik di bawah ini menunjukkan features mana yang paling berperan penting terhadap biaya pengobatan (charges)
![image](https://github.com/user-attachments/assets/771f5df5-87f7-41f6-8eb4-28b06092b4b3)

Dari table tersebut, menjelaskan bahwa : 
- Menunjukan bahwa fitur yang paling berpengaruh adalah smoker (perokok) dengan lebih dari 60%
- Fitur sex penting (sekitar 19%)
- Fitur age juga cukup berpengaruh
- Fitur BMI sedikit berpengaruh (kurang dari 10%)
- Fitur lainnya seperti children, bmi_category, region, bmi_smoker memiliki pengaruh yg sangat kecil

Cross Validation untuk validasi tambahan
```python

cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='r2')
print(f"\n=== Cross Validation Results ===")
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

```
Output : 

![image](https://github.com/user-attachments/assets/4627cfe2-36ec-43ef-a967-eefde27b8eaf)

**Visualisasi model Random Forest : Actual vs Predicted**
![image](https://github.com/user-attachments/assets/a294d94d-7b47-44ec-98f3-06f3b2f44899)

Penjelasan :

**Sumbu X** : Menampilkan nilai prediksi (Predicted Values) dari model Random Forest terhadap biaya asuransi.
**Sumbu Y** : Menampilkan residual (selisih antara nilai aktual dan nilai prediksi)

**Garis merah putus putus :** Merupakan garis referensi residu nol. Titik-titik yang dekat dengan garis ini menandakan prediksi yang sangat akurat.

**_insights_** : 
- Banyak titik residu berada dekat dengan garis nol, menunjukkan bahwa model memprediksi dengan akurat untuk sebagian besar data.
- Tidak terlihat pola sistematik pada distribusi residu, ini berarti model tidak menunjukan bias serius (overfitting atau underfitting)



### Kesimpulan
Berdasarkan hasil evaluasi:
**Problem 1 : Prediksi Biaya Asuransi Kesehatan** 
-  Model seperti Linear Regression dan Random Forest digunakan untuk memprediksi biaya asuransi berdasarkan fitur demografis dan gaya hidup.
- Hasil evaluasi menunjukkan bahwa Random Forest memiliki performa lebih baik:
    - MAE: 2461.34 (lebih kecil dari Linear Regression)
    - RMSE: 4362.76
    - R²: 0.8964 (vs 0.8069 pada Linear Regression)
- Ini menunjukkan bahwa model machine learning dapat memberikan prediksi biaya yang cukup akurat.

**Problem 2:** Faktor yang paling signifikan
| Fitur      | Importance                      |
| ---------- | ------------------------------- |
| **Smoker** | Sangat dominan (lebih dari 60%) |
| **Sex**    | Penting (sekitar 19%)           |
| **Age**    | Cukup berpengaruh               |
| **BMI**    | Sedikit berpengaruh             |
| Lainnya    | Pengaruh sangat kecil           |

- Berdasarkan visualisasi feature importance dari Random Forest, fitur smoker (status merokok) adalah faktor paling signifikan yang memengaruhi biaya asuransi.
- Faktor penting lain adalah sex, age, dan bmi.
- Informasi ini penting bagi perusahaan untuk memahami variabel yang paling memengaruhi biaya premi.

**Problem 3:** Keadilan dan Proporsionalitas Premi
- Dengan menggunakan model machine learning, perusahaan dapat:
    - Menilai risiko individu lebih objektif berdasarkan data.
    - Menyesuaikan premi asuransi secara proporsional terhadap variabel risiko nyata seperti usia dan status merokok.
    - Menghindari penetapan premi yang bersifat generalisasi atau diskriminatif.

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
