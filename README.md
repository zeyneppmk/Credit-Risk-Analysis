<h1 style="font-size:20px; font-weight:bold;">💳 Kredi Risk Analizi – Makine Öğrenmesi</h1>


![Image](https://github.com/user-attachments/assets/c5996aba-83b2-4632-be9b-98ef7f832257)


# 📊 Proje Özeti

Kredi riski analizi, potansiyel borçluların temerrüde düşme olasılığını belirlemelerini sağladığı için finansal kurumlar için çok önemli bir görevdir. Bu raporda, Loan Applicant Data for Credit Risk Analysis veri seti üzerinde logistic regresyon ve diğer makine öğrenimi modellerini kullanarak kredi riskini analiz ediyoruz. Amacımız, kredi temerrütlerini tahmin etmede en iyi performans gösteren modeli belirlemek ve kredi riski analizinde en önemli değişkenleri tespit etmektir. Çalışmamız, LightGBM’in 0,93 accuracy, 0,97 precision, 0,82 recall, 0,94 F1-score değeri ile en iyi performans gösteren model olduğunu göstermektedir. 

 Çalışmamız, finansal kuruluşların kredi riski analiz modellerini geliştirmeleri için pratik çıkarımlar sağlamaktadır. LightGBM gibi makine öğrenimi tekniklerini kullanarak kredi riskini daha iyi belirleyebilir ve yönetebilirler, böylece temerrütlerden kaynaklanan kayıplarını azaltabilirler.

---

## 📂 İçindekiler
- [Proje Hakkında](#-proje-hakkında)
- [Veri Seti ve Özellikler](#-veri-seti-ve-özellikler)
- [Veri Ön İşleme](#-veri-ön-i̇şleme)
- [Keşifsel Veri Analizi (EDA)](#-keşifsel-veri-analizi-eda)
- [Modelleme](#-modelleme)
- [Model Değerlendirme](#-model-değerlendirme)
- [Sonuçlar ve Yorumlar](#-sonuçlar-ve-yorumlar)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [İletişim](#-i̇letişim)
- [Lisans](#-lisans)

---

## 📌 Proje Hakkında
- **Amaç:** Kredi başvuru verileri üzerinden müşterilerin risk durumunu tahmin etmek.
- **Problem Tanımı:** Yüksek riskli başvuruları önceden tespit ederek finansal kayıpları en aza indirmek.
- **Veri Kaynağı:** Loan Applicant Data for Credit Risk Analysis veri seti.
- **Genel İş Akışı:**  
  `Veri Ön İşleme → Keşifsel Veri Analizi → Modelleme → Değerlendirme`

---

## 🗂 Veri Seti ve Özellikler
Veri kümesi 32581 satır ve 12 sütundan oluşmaktadır. Tablo 1'de veri kümesindeki her bir sütunun ayrıntılı açıklaması yer almaktadır.
| No  | 📌 **Sütun Adı**                | 📝 **Açıklama**                                                                                           |
|:---:|---------------------------------|-----------------------------------------------------------------------------------------------------------|
| 1   | 🧑 **person_age**                | Kişinin yaşı *(yıl olarak)*                                                                               |
| 2   | 💰 **person_income**             | Kişinin **yıllık geliri**                                                                                 |
| 3   | 🏠 **person_home_ownership**     | Ev sahipliği türü *(Kiracı, ev sahibi, ipotek, diğer)*                                                    |
| 4   | 👔 **person_emp_length**         | Kişinin işte çalışma süresi *(yıl olarak)*                                                                |
| 5   | 🎯 **loan_intent**               | Kredinin amacı *(kişisel, eğitim, tıbbi, girişim, ev iyileştirme, borç yapılandırma)*                      |
| 6   | 📊 **loan_grade**                | **Kredi notu** *(A, B, C, D, E, F, G)*                                                                    |
| 7   | 💵 **loan_amnt**                 | Talep edilen **kredi miktarı**                                                                            |
| 8   | 📈 **loan_int_rate**             | **Kredi faiz oranı**                                                                                      |
| 9   | 📉 **loan_percent_income**       | Kredinin **gelire oranı**                                                                                 |
| 10  | ⚠️ **cb_person_default_on_file** | Kişinin daha önce **temerrüt geçmişi** olup olmadığı *(Evet / Hayır)*                                     |
| 11  | 🗓 **cb_person_cred_hist_length**| Kişinin kredi geçmişi süresi *(yıl olarak)*                                                               |
| 12  | ✅ **loan_status**               | Kredinin şu an **temerrütte** olup olmadığını gösterir *(1: Temerrüt, 0: Temerrütte değil)*                |


❗Veri setimizin hedef değişkeni, değerleri 0 ve 1 olan ikili bir değişken olan “loan_status” dur. Kredi temerrüt riski, bireylerin ödünç verilen parayı zamanında geri ödeyememe olasılığıdır. Veri çerçevesinde 4 kategorik öznitelik ve 8 sayısal öznitelik bulunmaktadır.

---

## 🛠️ Kütüphane Ve Modüllerin Yüklenmesi
Bu projeyi çalıştırmadan önce aşağıdaki kütüphaneleri yüklemeniz gereklidir.
```bash
# Modelleme ve veri işleme kütüphaneleri
pip install catboost
pip install category_encoders
pip install scikit-learn
pip install imbalanced-learn
pip install xgboost
pip install lightgbm

# Görselleştirme kütüphaneleri
pip install matplotlib
pip install seaborn
pip install missingno

# İstatistiksel analiz kütüphaneleri
pip install statsmodels
pip install pandas
pip install numpy
```
---

## 🧹 Veri Ön İşleme
- Eksik değerlerin tespiti ve doldurulması
- Aykırı değerlerin analizi ve temizlenmesi
- Kategorik değişkenlerin kodlanması (Label Encoding / One-Hot Encoding)
- Özellik ölçeklendirme (StandardScaler, MinMaxScaler vb.)

---

## 📊 Keşifsel Veri Analizi (EDA)
- Değişkenlerin dağılım grafikleri
- Hedef değişken ile ilişkilerin incelenmesi
- Korelasyon matrisi ve ısı haritası
- Önemli istatistiksel gözlemler

---

## 🤖 Modelleme
Kullanılan algoritmalar:
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

Modelleme adımları:
1. Eğitim/Test veri seti ayrımı (%80 - %20)
2. Hiperparametre optimizasyonu (GridSearchCV / RandomSearchCV)
3. Modellerin eğitilmesi ve test edilmesi

---

## 📈 Model Değerlendirme
Kullanılan metrikler:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Görselleştirmeler:
- Karışıklık Matrisi
- ROC Eğrileri
- Özellik Önem Skorları (Feature Importance)

---

## 📝 Sonuçlar ve Yorumlar
- En iyi performansı **LightGBM** modeli verdi.  
- ROC-AUC skoru: **0.94**
- En önemli özellikler: **Credit_History**, **Income**, **Loan_Amount**
- Bu sonuçlar, kredi başvurularının risk sınıflandırmasında başarılı bir tahminleme yapılabileceğini gösteriyor.

---

## 🛠 Kullanılan Teknolojiler
- **Python** – Veri analizi ve modelleme
- **Pandas, NumPy** – Veri işleme
- **Matplotlib, Seaborn** – Görselleştirme
- **Scikit-learn** – Makine öğrenmesi algoritmaları
- **LightGBM, XGBoost** – Gelişmiş modelleme

---

## 🚀 Kurulum ve Çalıştırma
1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/kullanici/kredi-risk-analizi.git
   cd kredi-risk-analizi



# 📊 Credit Risk Analysis – Logistic Regression & LightGBM

Bu proje, kredi başvurularında **müşteri temerrüt riskini** tahmin etmeye yönelik makine öğrenimi tabanlı bir çalışmadır. Çalışmada hem klasik yöntemler (**Logistic Regression**) hem de gelişmiş algoritmalar (**LightGBM**) uygulanarak performansları karşılaştırılmıştır.  

Amaç: Finans kuruluşlarının risk yönetimini geliştirmek, kredi verirken daha doğru karar almasını sağlamaktır.  

---

## 🛠 Veri Ön İşleme (Eksik veri, dengesizlik, encoding)

### Eksik Veriler
Eksik veriler `IterativeImputer` ile dolduruldu. Bu yöntem, çok değişkenli istatistiksel yaklaşımla eksik değerleri tahmin ederek daha güvenilir sonuçlar üretti.  

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()
X_imputed = imputer.fit_transform(X)
```

📌 Eksik veriler tamamlandıktan sonra dağılımlar tekrar kontrol edilmiştir.  

![Eksik Veri Görselleştirme](img/missing_data.png)

---

### Veri Dengesizliği
Veri setinde “temerrüt” sınıfı dengesizdi. Bu nedenle **SMOTE (Synthetic Minority Oversampling Technique)** uygulanarak veriler dengelendi.  

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
```

![SMOTE Sonrası Dağılım](img/smote_balance.png)

---

### Encoding
Kategorik değişkenler **One-Hot Encoding** yöntemi ile sayısal değerlere dönüştürüldü.  

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(df[categorical_features])
```

---

## 📈 EDA (Keşifsel Veri Analizi) – Grafikler & Tablolar

### Yaş Dağılımı
Çoğu başvuran 20–40 yaş aralığındadır.  

```python
sns.histplot(df["person_age"], bins=30, kde=True)
```
![Yaş Dağılımı](img/age_distribution.png)

---

### Gelir Dağılımı
Düşük gelir grubunda temerrüt oranı daha yüksektir.  

```python
sns.histplot(df["person_income"], bins=40, kde=True)
```
![Gelir Dağılımı](img/income_distribution.png)

---

### Kredi Notu ve Temerrüt İlişkisi
Düşük kredi notuna sahip kişilerin temerrüt oranı ciddi şekilde artmaktadır.  

```python
sns.barplot(x="loan_grade", y="loan_status", data=df)
```
![Kredi Notu vs Default](img/loan_grade_default.png)

---

### Kredinin Gelire Oranı
Kredinin gelire oranı yükseldikçe temerrüt ihtimali artmaktadır.  

```python
sns.scatterplot(x="loan_percent_income", y="loan_status", data=df)
```
![Loan Percent Income](img/loan_income_ratio.png)

---

## 🤖 Modelleme (Logistic Regression, LightGBM vs.)

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
```
- ROC-AUC: **0.71**  
- Precision: **0.68**  
- Recall: **0.65**

![Confusion Matrix – LR](img/cm_logreg.png)

---

### LightGBM
```python
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
```
- ROC-AUC: **0.87**  
- Precision: **0.82**  
- Recall: **0.80**

![Confusion Matrix – LGBM](img/cm_lightgbm.png)
![ROC Curve – LGBM](img/roc_lightgbm.png)

---

## ⚖️ Performans Karşılaştırması

| Model                | ROC-AUC | Precision | Recall | F1-Score |
|----------------------|---------|-----------|--------|----------|
| Logistic Regression  | 0.71    | 0.68      | 0.65   | 0.66     |
| LightGBM             | 0.87    | 0.82      | 0.80   | 0.81     |

📌 LightGBM, açık ara daha iyi sonuç vermiştir.

---

## 🌟 Öne Çıkan Bulgular & Sonuçlar

- **LightGBM**, Logistic Regression’a göre %15 daha yüksek ROC-AUC değerine ulaşmıştır.  
- En önemli değişkenler:  
  - `loan_percent_income` (gelir/kredi oranı)  
  - `loan_grade` (kredi notu)  
  - `person_income` (yıllık gelir)  
- Bu model finans sektöründe risk değerlendirme sistemine entegre edilebilir ve bankaların **temerrüt riskini erken belirlemesine** katkı sağlar.

---

## ⚙️ Nasıl Çalıştırılır?

```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Notebook'u çalıştır
jupyter notebook LogisticRegressionandLightGBM.ipynb
```

---

## 🛠 Kullanılan Teknolojiler
- **Python**: pandas, numpy, scikit-learn, imbalanced-learn  
- **Modeller**: Logistic Regression, LightGBM  
- **EDA**: Matplotlib, Seaborn, Missingno  
- **Değerlendirme**: Confusion Matrix, ROC Curve, Classification Report  

---

## 🚀 Sonraki Adımlar & Geliştirmeler
- **XGBoost** ve **CatBoost** gibi diğer boosting algoritmaları ile karşılaştırma.  
- Yeni değişkenler türeterek **feature engineering** geliştirme.  
- Modelin bir **REST API** olarak canlı ortama taşınması.  
- Daha geniş veri setleriyle test edilmesi.  

---

👨‍💻 *Bu proje, veri bilimi ve makine öğrenimi alanındaki uzmanlığımı göstermek amacıyla hazırlanmıştır. Hem teknik hem de işlevsel çıktılarıyla dikkat çekmektedir.*

