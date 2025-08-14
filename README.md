<h1 align="center">💳 <strong>Kredi Risk Analizi – Makine Öğrenmesi</strong></h1>
![Image](https://github.com/user-attachments/assets/78385db6-2e32-409f-871e-72fc26afae8c)


# 📊 Kredi Riski Analizi – Makine Öğrenmesi

Bu proje, kredi başvuru verilerini analiz ederek başvuru sahibinin **kredi riskini tahmin etmeyi** amaçlamaktadır.  
Finansal kurumların, kredi başvurularını değerlendirirken daha hızlı ve doğru kararlar almasına yardımcı olur.

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
- **Toplam Gözlem:** 2000+
- **Önemli Değişkenler:**
  - `Income` – Aylık gelir
  - `Age` – Yaş
  - `Credit_History` – Kredi geçmişi puanı
  - `Loan_Amount` – Talep edilen kredi tutarı
  - `Employment_Type` – Çalışma durumu
- **Hedef Değişken:** `Risk_Flag` (0 = Düşük Risk, 1 = Yüksek Risk)

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
