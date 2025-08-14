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
