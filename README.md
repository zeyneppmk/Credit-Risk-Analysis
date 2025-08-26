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
- Verinin genel bilgilerini alma
- Eksik değerlerin tespiti ve ortalama değerleri ile doldurulması
- Duplicate(yinelenen) satırların tespiti ve silinmesi
- Aykırı değerlerin analizi ve temizlenmesi
- Kategorik değişkenlerin kodlanması (Label Encoding / Binary Encoding /One-Hot Encoding)

### 1- Verinin Genel Bilgilerini Alma

  📌 Veri setinin anlaşılması için öncelikle incelenmesi gerekmektedir.
```python
df = pd.read_csv('loan_data.csv')
#tum sutunları gozlemlemek icin
pd.set_option('display.max_columns', None)
df.head(20)
```
<img width="1709" height="372" alt="Image" src="https://github.com/user-attachments/assets/664916ed-9c65-4c4c-b845-778f030cf401" />

---

📌 Verinin kaç satır ve sütundan oluştuğunun gözlemlenmesi yaptığımız işlemlerde öncesinin ve sonrasının daha iyi anlaşılması için önemlidir.

```python
df.shape[0],df.shape[1]
```
<img width="1825" height="47" alt="image" src="https://github.com/user-attachments/assets/6a2ad51d-e64f-49b3-831f-d6a41132cbee" />

---

📌 Veri yapısının genel bilgisini görmek için `df.info()` kullanılır

```python
#veri genel bilgilerini alma
df.info()
```
<img width="1844" height="438" alt="image" src="https://github.com/user-attachments/assets/c5cf1690-bc0b-48ce-9603-b7466ea6cec9" />

---

📌 Sayısal sütunların istatistiksel özetini görmek için `df.describe()` kullanılır

```python
#veri genel bilgilerini alma
df.describe()
```
<img width="1485" height="383" alt="image" src="https://github.com/user-attachments/assets/3cadaca0-309d-4cda-b6b7-9e768b0727ce" />


---


### 2- Eksik Veriler ve İşlem Yöntemleri

📌 Eksik veriler(Missing Values) belirlednikten sonra veri setinin durumuna göre nasıl bir yol izleneceği belirlenmelidir. Aşağıda bazı yöntemler açıklanmıştır ⬇️

| ✅ Yöntem | 📝 Açıklama | 📌 Ne Zaman Kullanılır? |
|-----------|------------|--------------------------|
| **Satır Silme** (`dropna`) | Eksik değer içeren satırları tamamen siler | Eksik oranı düşükse (< %5), veri kaybı kritik değilse |
| **Sütun Silme** | Eksik değer oranı çok yüksek olan sütunu siler | Eksik oranı çok büyükse (> %40) ve sütun kritik değilse |
| **Sabit Değer ile Doldurma** (`fillna("Unknown", 0)`) | Eksikleri belirli bir sabit değerle doldurur | Kategorik verilerde “Unknown” gibi, sayısalda 0 gibi nötr değer gerekiyorsa |
| **Ortalama / Medyan / Mod** | Sayısal veriler için mean/median, kategorik için mod ile doldurur | Eksik oranı orta düzeydeyse (%5–30), dağılım dengeliyse |
| **İleri / Geri Doldurma** (`ffill`, `bfill`) | Eksik değerleri bir önceki veya sonraki değerle doldurur | Zaman serilerinde (ör. günlük fiyat, sensör verisi) |
| **Tahmine Dayalı Yöntemler** (`KNNImputer`, `IterativeImputer`) | Diğer sütunları kullanarak eksikleri tahmin eder | Eksik oranı yüksekse veya basit doldurma yöntemleri işe yaramıyorsa |
| **Eksiklik Bayrağı Oluşturma** | Eksik değer var mı yok mu bilgisini binary sütun olarak ekler | Eksikliğin kendisi anlamlı bir bilgi taşıyorsa (örn. gelir bilgisi boş = riskli müşteri) |

📌 Bu projede `person_emp_length` ve `loan_int_rate` sütunlarında eksik değerler kaydedilmiştir. Bu verilerin projedeki önemi göz önüne alınarak farklı yöntemler kullanılmıştır. 

```python
##none veya nan degerlerin sayisini belirtir
print("Eksik Veri Kontrolü ve toplamda kaç adet eksik veri içerdiği")
print(df.isnull().sum())
```
<img width="1328" height="319" alt="image" src="https://github.com/user-attachments/assets/7198644a-4f91-49ec-ac8b-2b0ed4562222" />

 ---

📌 `person_emp_length`sütunundaki eksik değerlerin ortadan kalkması için ortalama değeri bulunup eksik olan satırlara yerleştirilmiştir. 

```python
# 'person_emp_length' sütunundaki ortalamayı hesaplayın
mean_emp_length = df['person_emp_length'].mean()

# Eksik (NaN) değerleri ortalama ile doldurun
df['person_emp_length'].fillna(mean_emp_length, inplace=True)

```
📌 `loan_int_rate` sütunundaki eksik değerlerin olduğu satırlar silindi.

```python
# 'loan_int_rate' sütunundaki eksik değerleri silmek
df = df[df['loan_int_rate'].notna()]

# Güncellenmiş DataFrame'i kontrol etmek için
df.head()
```
---

### 3- Duplicate (yinelenen) Satırların Tespiti
📌Veri toplama sürecindeki hatalar , veri birleştirirken yapılan hatalar ve benzer sebeplerden dolayı veri setlerinde duplicate(tekrarlayan) veriler ile karşılaşılmaktadır. Duplicate veriler istatikssel analizi bozar ve modeli yanıltır, hesaplama maliyetini arttırır. Duplicate veriler kontrol edilerek veri setinden silinmelidir.

```python
## Checking for Duplicates
dups = df.duplicated()
dups.value_counts() 
```

<img width="1100" height="200" alt="image" src="https://github.com/user-attachments/assets/09e02441-258a-496e-998b-3490c978b802" />


```python
print(f"duplicate(yinelenen) satirlari kaldirmadan once verinin sekli: {df.shape[0]},{df.shape[1]} \n")
df.drop_duplicates(inplace=True)
print(f"duplicate(yinelenen) satirlari kaldirdiktan sonra verinin sekli: {df.shape[0]},{df.shape[1]}")
```
<img width="1418" height="114" alt="image" src="https://github.com/user-attachments/assets/a44da34b-1bf2-4366-8ae9-d20c0b4c046c" />

---

### 4- Aykırı Değerlerin Analizi
📌İstatistikte aykırı değer, diğer gözlemlerden önemli ölçüde farklı olan bir veri noktasıdır.Veri girişi hataları, farklı ölçüm birimleri yada gerçek ama nadir durumlardan dolayı gözlemlenebilir.

📌Bu projede aykırı değerler aşağıdaki yöntemlerle kontrol edilmiştir ⬇️

#### 🔎 Frekans Analizi ile Aykırı Değer Tespiti 

- 📌 Sayısal değişkenlerde (ör. `person_age`, `person_emp_length`) **value_counts()** ile dağılım incelenmiştir.  
- 🔍 Böylece yaş veya çalışma süresi gibi değişkenlerde **beklenmeyen uç değerler** kolayca fark edilmiştir.  
- 🏠 Kategorik değişkenlerde (ör. `person_home_ownership`) **value_counts()** nadir kategorilerin belirlenmesi için kullanılmıştır.  
- 💰 Sürekli değişkenlerde (ör. `loan_int_rate`) oranların mantıklı aralıkta olup olmadığı kontrol edilmiştir.  

```python
df['person_age'].value_counts().sort_index()
df['person_emp_length'].value_counts()
df['person_home_ownership'].value_counts()
df['loan_int_rate'].value_counts()
```

---


### 5- Kategorik Değişkenlerin Kodlanması (Label Encoding / Binary Encoding)
📌 Veri setinde bazı sütunlar string türündedir(örn. "Ev Sahibi", "Kiracı").Öncelikle veri setinde bu sütunların tespit edilmesi gerekmedktedir.

```python
ccol = df.select_dtypes(include = ["object"]).columns
ncol = df.select_dtypes(include = ["int","float"]).columns

print(f"Categorical Column: {ccol} \n")
print(f"Numerical Column: {ncol}")

print(f"\nCategorical Column Count: {len(ccol)} \n")
print(f"Numerical Column Count: {len(ncol)}")
```
<img width="1439" height="284" alt="image" src="https://github.com/user-attachments/assets/c5673fec-bc3f-4adf-925f-3d96cda56ee5" />

---

📌 Makine öğrenmesi algoritmaları yalnızca **sayısal verilerle** çalışır.  
Bu nedenle kategorik (string) veriler uygun yöntemlerle **sayılara dönüştürülmelidir**.  
Aşağıda en sık kullanılan iki yöntem açıklanmıştır:

#### 1️⃣ Label Encoding
Verilerimizi birebir sayısallaştırmak için kullanılan fonksiyondur. Yani kategorik her veriye sayısal bir değer (etiket numarası) atar. Örneğin 4 farklı meyvemiz olduğunu varsayalım, Label Encoding fonksiyonu sıfırdan başlayıp her bir meyve için etiket numarası verecektir.

📌 Bu projede doğrudan `dsklearn.LabelEncoder` kullanmak yerine bazı özel tanımlı Label Encoding fonksiyonları yazılmıştır.
- **SC_LabelEncoder1** : Burada kredi notları (loan_grade) harflerden sayılara çevriliyor.

“A” en yüksek puanı temsil ettiği için 7, “G” en düşük puanı temsil ettiği için 1 atanmış. Eğer başka bir değer varsa → 0 döndürülüyor. Böylece kredi notunu sıralı (ordinal) hale getirilmiş olur.

- **SC_LabelEncoder2** : Bu, kişinin daha önce temerrüde düşüp düşmediğini gösteren (cb_person_default_on_file) sütun için.

Y (Yes) → 0, N (No) → 1 yapılmış.

- **SC_LabelEncoder3** : Ev sahipliği (person_home_ownership) kategorisi sayılara çevriliyor:

Rent (kiracı) → 1

Mortgage (ipotekli ev) → 2

Own (ev sahibi) → 3

Diğer tüm durumlar → 0


```python
def SC_LabelEncoder1(text):
    if text == "G":
        return 1
    elif text == "F":
        return 2
    elif text == "E":
        return 3
    elif text == "D":
        return 4
    elif text == "C":
        return 5
    elif text == "B":
        return 6
    elif text == "A":
        return 7
    else:
        return 0
def SC_LabelEncoder2(text):
    if text == "Y":
        return 0
    elif text == "N":
        return 1
def SC_LabelEncoder3(text):
    if text == "RENT":
        return 1
    elif text == "MORTGAGE":
        return 2
    elif text == "OWN":
        return 3
    else:
        return 0

df["loan_grade"] = df["loan_grade"].apply(SC_LabelEncoder1)
df["cb_person_default_on_file"] = df["cb_person_default_on_file"].apply(SC_LabelEncoder2)
df["person_home_ownership"] = df["person_home_ownership"].apply(SC_LabelEncoder3)
```

<img width="1844" height="507" alt="image" src="https://github.com/user-attachments/assets/0bbd339c-1210-40de-9ea6-1e630baa95f6" />

---

#### 2️⃣ Binary Encoding
Kategoriler önce sayılara, ardından ikilik (binary) sisteme çevrilir. Çok kategorili (high cardinality) değişkenlerde kullanılarak sütun patlamasını önler.

📌 `loan_intent` sütunu için EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT olmak üzere toplamda altı adet farklı değer bulunmaktadır. 

Bu sebepten binary encoding ile 3 basamaklı kodlanmış olur . 2 bit max 4 farklı kombinasyon yaparken 3 bit 8 farklı kombinasyon yapabilmektedir.

```python
# Binary Encoding işlemi
encoder = ce.BinaryEncoder(cols=['loan_intent'])
df_encoded = encoder.fit_transform(df)
df = df_encoded
# Sonuçları kontrol etme
df.head()
```
<img width="1865" height="267" alt="image" src="https://github.com/user-attachments/assets/b9af59fe-0663-4d61-b8a7-4f762729c8cf" />


---

## 🔍 Keşifsel Veri Analizi (Exploratory Data Analysis - EDA)

📌 **EDA (Exploratory Data Analysis)**, veri biliminde bir veri setini ilk defa incelediğimizde yaptığımız temel adımdır.  
Amacı, veriyi **daha iyi anlamak**, **örüntüleri görmek**, **anormallikleri tespit etmek** ve sonraki adımlar için uygun modelleri seçmeye zemin hazırlamaktır.

### 🛠️ EDA’da Kullanılan Yaygın Yöntemler
- **Tanımlayıcı İstatistikler:** Ortalama, medyan, standart sapma gibi özet bilgiler (`df.describe()`)  
- **Veri Yapısı İncelemesi:** Değişken türleri, eksik değerler, duplicate kayıtlar (`df.info()`, `df.isnull().sum()`)  
- **Görselleştirmeler:** Histogram, boxplot, dağılım grafikleri, korelasyon ısı haritaları  
- **Korelasyon Analizi:** Değişkenler arasındaki doğrusal/non-doğrusal ilişkileri anlamak

📌 Bu projede kullanılan bazı yöntemler aşağıdaki gibidir

- Korelasyon Matrisinin Hesaplanması

```python
# Korelasyon matrisi
corr_matrix = df.corr()

# Isı haritası
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```
<img width="1279" height="950" alt="image" src="https://github.com/user-attachments/assets/50f7795b-3633-4bab-b8a5-7682173e71bd" />


**cb_person_cred_hist_length**: Kredi geçmişi uzunluğu, doğrudan kişinin kredi itibarını gösterir. Daha uzun bir kredi geçmişi genellikle daha iyi kredi puanına işaret eder ve kredi risk analizinde kritik bir faktör olarak kabul edilir.

**person_age**, dolaylı bir etkiye sahiptir. Yaş tek başına kredi riski açısından yeterli bilgi sunmayabilir; ancak kredi geçmişi uzunluğu bireyin ödeme geçmişi hakkında doğrudan bilgi sağlar.

---
```python
# Korelasyon matrisini hesaplama
correlation_matrix = df.corr()

# loan_status ile olan korelasyonu görmek için
loan_status_correlation = correlation_matrix['loan_status'].sort_values(ascending=False)

print(loan_status_correlation)
```
<img width="1241" height="310" alt="image" src="https://github.com/user-attachments/assets/20fb1393-6741-4c63-8212-478aae65493c" />

Bu çıktıyı yorumlarken, her bir özellik ile **loan_status** (kredi durumu) arasındaki ilişkiyi inceleyebiliriz. Korelasyon değeri, -1 ile +1 arasında değişir ve aşağıdaki şekilde yorumlanır:

- **1.0**: Mükemmel pozitif ilişki
- **-1.0**: Mükemmel negatif ilişki
- **0.0**: Hiçbir ilişki yok

##### **loan_status ile diğer değişkenler arasındaki korelasyonlar:**

1. **loan_status: 1.000000**
   - **loan_status** ile kendi arasında mükemmel bir pozitif korelasyon vardır, çünkü bu değişken kendisini ifade eder.

2. **loan_percent_income: 0.379359**
   - **loan_percent_income** (kredi miktarının gelirle oranı) ile **loan_status** arasında orta düzeyde pozitif bir korelasyon vardır. Yani, gelirine oranla daha fazla kredi talep eden kişilerin temerrüte düşme olasılığı daha yüksek olabilir.

3. **loan_int_rate: 0.335788**
   - **loan_int_rate** (kredi faiz oranı) ile **loan_status** arasında da pozitif bir korelasyon vardır. Yüksek faiz oranlarına sahip kredilerin temerrüde düşme olasılığı daha yüksek olabilir.

4. **loan_amnt: 0.106885**
   - **loan_amnt** (kredi tutarı) ile **loan_status** arasında düşük düzeyde pozitif bir korelasyon vardır. Yani, kredi tutarı arttıkça temerrüte düşme olasılığı biraz daha artabilir, ancak bu ilişki çok güçlü değildir.

5. **loan_intent_0: 0.060206**, **loan_intent_1: 0.036874**, **loan_intent_2: -0.082012**
   - **loan_intent** kategorileri (kredi niyeti) ile **loan_status** arasında zayıf ilişkiler vardır. Kredi niyetinin temerrüt durumu üzerindeki etkisi çok belirgin değildir.

6. **cb_person_cred_hist_length: -0.014571**
   - **cb_person_cred_hist_length** (kredi geçmişi uzunluğu) ile **loan_status** arasında negatif bir ilişki vardır, ancak bu ilişki çok zayıftır. Kredi geçmişi uzun olan kişilerin temerrüte düşme olasılığı çok belirgin şekilde düşük değildir.

7. **person_emp_length: -0.085013**
   - **person_emp_length** (çalışma süresi) ile **loan_status** arasında negatif bir korelasyon vardır. Yani, daha uzun süre çalışan kişilerin temerrüte düşme olasılığı biraz daha düşük olabilir.

8. **person_income: -0.172207**
   - **person_income** (kişinin yıllık geliri) ile **loan_status** arasında negatif bir ilişki vardır. Yüksek geliri olan kişilerin temerrüte düşme olasılığı daha düşük olabilir.

9. **cb_person_default_on_file: -0.180412**
   - **cb_person_default_on_file** (kredi geçmişinde temerrüt olup olmadığı) ile **loan_status** arasında negatif bir ilişki vardır. Yani, kredi geçmişinde temerrüt bulunan kişilerin, kredi temerrüt durumunda olmama olasılığı daha yüksek olabilir.

10. **person_home_ownership: -0.232697**
    - **person_home_ownership** (ev sahipliği durumu) ile **loan_status** arasında orta düzeyde negatif bir korelasyon vardır. Ev sahipliği durumu, temerrüt durumuyla negatif bir ilişki gösteriyor, yani ev sahibi olan kişilerin temerrüde düşme olasılığı daha düşük olabilir.

11. **loan_grade: -0.376282**
    - **loan_grade** (kredi notu) ile **loan_status** arasında orta düzeyde negatif bir korelasyon vardır. Kredi notu arttıkça, temerrüde düşme olasılığı azalmaktadır. Bu, genellikle yüksek kredi notuna sahip kişilerin daha iyi ödeme geçmişine sahip olmaları ile ilgilidir.

---


---
```python
# Bağımsız ve bağımlı değişkenler
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Modeli tanımlama
logreg_model = LogisticRegression(random_state=42, max_iter=1000)

# Modeli eğitme
logreg_model.fit(X, y)

# Özelliklerin katsayılarını alma
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(logreg_model.coef_[0])  # Katsayıların mutlak değerini alıyoruz
})

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance (Logistic Regression)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
```
<img width="1193" height="588" alt="image" src="https://github.com/user-attachments/assets/334c6650-8368-46c8-95b0-3f029e43bb6f" />











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
🧩
