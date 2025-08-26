<h1 style="font-size:20px; font-weight:bold;">💳 Kredi Risk Analizi – Makine Öğrenmesi</h1>


![Image](https://github.com/user-attachments/assets/c5996aba-83b2-4632-be9b-98ef7f832257)



# 📊 Proje Özeti

Kredi riski analizi, potansiyel borçluların temerrüde düşme olasılığını belirlemelerini sağladığı için finansal kurumlar için çok önemli bir görevdir. Bu raporda, Loan Applicant Data for Credit Risk Analysis veri seti üzerinde logistic regresyon ve diğer makine öğrenimi modellerini kullanarak kredi riskini analiz ediyoruz. Amacımız, kredi temerrütlerini tahmin etmede en iyi performans gösteren modeli belirlemek ve kredi riski analizinde en önemli değişkenleri tespit etmektir. Çalışmamız, LightGBM’in 0,93 accuracy, 0,97 precision, 0,82 recall, 0,94 F1-score değeri ile en iyi performans gösteren model olduğunu göstermektedir. 

 Çalışmamız, finansal kuruluşların kredi riski analiz modellerini geliştirmeleri için pratik çıkarımlar sağlamaktadır. LightGBM gibi makine öğrenimi tekniklerini kullanarak kredi riskini daha iyi belirleyebilir ve yönetebilirler, böylece temerrütlerden kaynaklanan kayıplarını azaltabilirler.

---

## 📂 İçindekiler
- [Proje Hakkında](#-proje-hakkında)
- [Veri Seti ve Özellikler](#-veri-seti-ve-özellikler)
- [Kütüphane Ve Modüller](#-kütüphane-ve-modüller)
- [Veri Ön İşleme](#-veri-ön-i̇şleme)
- [Keşifsel Veri Analizi (EDA)](#-keşifsel-veri-analizi-eda)
- [Modelleme](#-modelleme)
- [Model Değerlendirme](#-model-değerlendirme)
- [Sonuçlar ve Yorumlar](#-sonuçlar-ve-yorumlar)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Sonraki Adımlar & Geliştirmeler](#-sonraki-adımlar-ve-geliştirmeler)
- [Soru / İletişim](#soru-i̇letişim)

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

🔗 Bu çıktıyı yorumlarken, her bir özellik ile **loan_status** (kredi durumu) arasındaki ilişkiyi inceleyebiliriz. Korelasyon değeri, -1 ile +1 arasında değişir ve aşağıdaki şekilde yorumlanır:

- ✅ **1.0**: Mükemmel pozitif ilişki
- ❌ **-1.0**: Mükemmel negatif ilişki
- ⚪ **0.0**: Hiçbir ilişki yok

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

```python
# loan_status ile aralarındaki ilişki güçlü olan sütunlar
fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(data=df2, x='loan_status', hue="loan_percent_income", ax=ax[0]).set_title("Grafik 1");
sns.countplot(data=df2, x='loan_status', hue='loan_int_rate', ax=ax[1]).set_title("Grafik 2");
```
<img width="1407" height="507" alt="image" src="https://github.com/user-attachments/assets/85f8e5ef-01fb-457e-bcb2-8c525a1aab18" />

📌 Grafik 1 – loan_status & loan_percent_income : Bu grafik, kredi durumunun (loan_status) farklı gelir oranları (loan_percent_income) ile nasıl dağıldığını gösteriyor.

Amaç: Gelirinin ne kadarını kredi ödemesine ayıran kişilerin, krediyi geri ödeyip ödememe durumunu gözlemlemek.

📌Grafik 2 – loan_status & loan_int_rate : Bu grafik, kredi durumunun (loan_status) farklı faiz oranları (loan_int_rate) ile ilişkisini gösteriyor.

Amaç: Faiz oranı yükseldikçe krediyi ödeyememe ihtimali artıyor mu sorusuna cevap aramak.

---

```python
plt.figure(figsize=(12,6))
sns.boxplot(x='loan_intent', y='loan_amnt', data=df2)
plt.xticks(rotation=30)
plt.title('Boxplot 2')
```
<img width="1169" height="692" alt="image" src="https://github.com/user-attachments/assets/52ef40e8-ce18-4f40-83e6-3c5e8a23d730" />

📌Boxplot 2 – loan_intent & loan_amnt : Bu boxplot, farklı kredi amaçları (loan_intent) için kullanılan kredi miktarlarının (loan_amnt) dağılımını gösteriyor.

Amaç: Hangi kredi türlerinde daha yüksek tutarlar çekildiğini ve aykırı değerleri (outlier) gözlemlemek.

---

```python
fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(data=df2, x='loan_status', hue='person_home_ownership', ax=ax[0]).set_title("Grafik 3");
sns.countplot(data=df2, x='loan_status', hue='loan_grade', ax=ax[1]).set_title("Grafik 4");
```
<img width="1419" height="507" alt="image" src="https://github.com/user-attachments/assets/653c9853-38c3-4610-bfa0-5cd0fdaa491b" />


📌Grafik 3 – loan_status & person_home_ownership : Bu grafik, kredi durumunu (loan_status) kişilerin ev sahipliği durumuna (person_home_ownership) göre karşılaştırıyor.

Amaç: Ev sahibi, kiracı veya ipotekli ev sahibi olmanın kredi ödeme davranışına etkisini görmek.

📌 Grafik 4 – loan_status & loan_grade : Bu grafik, kredi durumunu (loan_status) verilen kredi derecesine (loan_grade) göre gösteriyor.

Amaç: Düşük kredi derecesine sahip kişilerin ödemede daha çok sorun yaşayıp yaşamadığını incelemek.
---



## 🤖 Modelleme

Kredi temerrüt tahmini için iki farklı model kullanıldı:

- 📌 Logistic Regression: Basit ve yorumlanabilir bir doğrusal sınıflandırma modeli. Özellikle ikili sınıflandırma (loan_status) problemlerinde temel bir karşılaştırma noktası sunar.

- 📌 LightGBM (Light Gradient Boosting Machine): Ağaç tabanlı, hızlı ve güçlü bir boosting algoritması. Büyük veri setlerinde yüksek doğruluk ve hız sağlar.

Modelleme adımları:
1. Eğitim/Test veri seti ayrımı (%80 - %20)
2. Hiperparametre optimizasyonu (GridSearchCV / RandomSearchCV)
3. Modellerin eğitilmesi ve test edilmesi

---

### 1️⃣ Veri Bölme
- 🎯 loan_status hedef değişken, diğer sütunlar özellik olarak alındı.

- 📊 Veri %80 eğitim – %20 test olacak şekilde ayrıldı.

- ⚖️ stratify=y → Sınıflar (temerrüt / temerrüt değil) eğitim ve test setinde aynı dağılımı korudu.
- 
```python
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y,
                                                    shuffle=True)

```

---

### 2️⃣ Modellerin Tanımlanması

- Logistic Regression ve LightGBM modelleri tanımlandı.

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=0)
}
```

---

### 3️⃣ Cross-Validation (StratifiedKFold)

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

```

### 4️⃣ Eğitim ve Değerlendirme

Her model için:

✅ CV Accuracy Mean: 5 katlı doğrulama ortalama doğruluğu

✅ Test Accuracy: Test setinde genel doğruluk

✅ Precision: Doğru tahmin edilen pozitiflerin oranı

✅ Recall: Gerçek pozitifleri yakalama oranı

✅ F1-Score: Precision ve Recall’un dengesi

✅ ROC AUC: Modelin sınıfları ayırma gücü

📌 Sonuçlar hem test seti hem de çapraz doğrulama üzerinden raporlandı.

```python
# 4. Modelleri eğitme ve değerlendirme
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")

    # Cross-validation ile doğruluk skoru hesaplama
    cv_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_val_cv)

        # Cross-validation doğruluk
        cv_scores.append(accuracy_score(y_val_cv, y_pred_cv))

    mean_cv_accuracy = np.mean(cv_scores)

    # Modeli test setiyle değerlendirme
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Test seti için doğruluk metriklerini hesaplama
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # ROC AUC score

    # Sonuçları saklama
    model_results[model_name] = {
        'CV Accuracy Mean': mean_cv_accuracy,
        'Test Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }

```

---

## 📈 Model Değerlendirme
✅Kullanılan metrikler:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

🧩Görselleştirmeler:
- (Confusion Matrix) Karışıklık Matrisi
- ROC Eğrileri
- Özellik Önem Skorları (Feature Importance)

### Logistic Regresyon için Performans Metrikleri (Test Seti)

Çapraz doğrulama aşamasında elde edilen metrikler, modelin genelleme performansını ortaya koymaktadır. Ortalama sonuçlar şu şekildedir:

- **Doğruluk Skoru (Accuracy): 0.8420**
Accuracy, modelin doğru sınıflandırdığı örneklerin toplam örnek sayısına oranıdır. Modelin doğru tahmin yapma oranı oldukça yüksek olup, genel performansın tatmin edici olduğunu göstermektedir.

- **Kesinlik (Precision): 0.7217**  
Precision, modelin pozitif olarak tahmin ettiği örneklerin gerçekten pozitif olma oranıdır. Model, pozitif sınıf (1) için yaptığı tahminlerde orta düzeyde bir isabet oranına sahiptir.

- **Duyarlılık (Recall): 0.4445**  
Duyarlılık metriği, modelin pozitif sınıfları tespit etmede bazı zorluklarla karşılaştığını göstermektedir. Yanlış negatif tahminlerin oranı bu metriği olumsuz etkileyen bir faktör olarak değerlendirilebilir.

- **F1 Skoru: 0.5502**
F1 skoru, precision ve recall arasındaki dengeyi yansıtmaktadır. Bu metrik, modelin genel başarımını dengeli bir şekilde değerlendirmektedir. Fakat bu modelde düşük çıkmıştır.

---

### LightGBM Classification için Performans Metrikleri (Test Seti)

Çapraz doğrulama aşamasında elde edilen metrikler, modelin genelleme performansını ortaya koymaktadır. Ortalama sonuçlar şu şekildedir:

- **Doğruluk Skoru (Accuracy): 0.9319**
  
Accuracy, modelin doğru sınıflandırdığı örneklerin toplam örnek sayısına oranıdır. Modelin doğru tahmin yapma oranı oldukça yüksek olup, genel performansın tatmin edici olduğunu göstermektedir. 

-**Kesinlik (Precision): 0.9665**

Precision, modelin pozitif olarak tahmin ettiği örneklerin gerçekten pozitif olma oranıdır. Model, pozitif sınıf (1) için yaptığı tahminlerde yüksek bir isabet oranına sahiptir. Yanlış pozitif tahminlerin oranının düşük olması, modelin güvenilirliğini artırmaktadır. 

-**Duyarlılık (Recall): 0.8221**

Duyarlılık metriği, modelin pozitif sınıfı doğru tespit etme oranını ifade etmektedir. Ve bu modelde yüksek çıkmıştır.

-**F1 Skoru: 0.9429**

F1 skoru, precision ve recall arasındaki dengeyi yansıtmaktadır. Bu metrik, modelin genel başarımını dengeli bir şekilde değerlendirmektedir.


---
### ROC Eğrisi ve AUC Analizi

**LightGBM (AUC = 0.94):**

LightGBM modeli, eğrisiyle daha geniş bir alan kapladığı için daha yüksek bir AUC değerine sahiptir.
Bu, modelin sınıflandırma performansının oldukça iyi olduğunu ve pozitif sınıfı negatif sınıftan ayırt etmede başarılı olduğunu gösterir.

-**Logistic Regression (AUC = 0.84):**

Lojistik regresyonun AUC değeri LightGBM'e göre daha düşüktür.
Bu model, pozitif ve negatif sınıfları ayırt etmekte LightGBM kadar etkili değildir ancak yine de iyi bir performans sergilemektedir.

<img width="780" height="588" alt="image" src="https://github.com/user-attachments/assets/ec0d66a1-a34f-4bff-ab02-e99ee0cd3511" />

---
### 📌Logistic Regresyon için Karışıklık Matrisi(Confusion Matrix) Analizi

- **TP (Doğru Pozitif):** Modelin "temerrüt" olarak tahmin ettiği ve gerçekte de temerrüt olan örnekler. 573 temerrüt değeri bu modeldin performans düşüklüğünü göstermektedir.
- **TN (Doğru Negatif):** Modelin "ödenmiş" olarak tahmin ettiği ve gerçekte de ödenmiş olan örnekler. Modelde 4352 değerine karşılık gelmektedir. 
- **FP (Yanlış Pozitif):** Modelin "temerrüt" olarak tahmin ettiği, ancak gerçekte ödenmiş olan örnekler. Bu modelde 221 gibi çok az bir değere karşılık gelmektedir. 
- **FN (Yanlış Negatif):** Modelin "ödenmiş" olarak tahmin ettiği, ancak gerçekte temerrüt olan örnekler. Modelde 716 ya denk gelmektedir. Bu değer test verisine oranladığımızda yüksek çıkmıştır bu da modelin iyi çalışmadığını gösterir.

<img width="615" height="434" alt="image" src="https://github.com/user-attachments/assets/d79c6644-8e45-403f-a2bd-d28ee884d23c" />

---

### 📌LightGBM Classification için Karışıklık Matrisi(Confusion Matrix) Analizi

- **TP (Doğru Pozitif):** Modelin "temerrüt" olarak tahmin ettiği ve gerçekte de temerrüt olan örnekler. 922 Temerrüt değeri veride az olduğu için bu şekilde çıkmıştır yani aslında yüksek bir performans göstermektedir.
- **TN (Doğru Negatif):** Modelin "ödenmiş" olarak tahmin ettiği ve gerçekte de ödenmiş olan örnekler. Modelde 4541 değerine karşılık gelmektedir. Toplam test verisine oranladığımızda yüksek bir performans elde edildiği gözlemlenir.
- **FP (Yanlış Pozitif):** Modelin "temerrüt" olarak tahmin ettiği, ancak gerçekte ödenmiş olan örnekler. Bu modelde 32 gibi çok az bir değere karşılık gelmektedir. Bu modelin performansının iyi olduğunu gösterir.
- **FN (Yanlış Negatif):** Modelin "ödenmiş" olarak tahmin ettiği, ancak gerçekte temerrüt olan örnekler. Modelde 367 ye denk gelmektedir.

<img width="511" height="424" alt="image" src="https://github.com/user-attachments/assets/e74bd41e-bca4-4c74-8c28-3845e01644cd" />

---

## 📝 Sonuçlar ve Yorumlar

- **LightGBM:** En yüksek doğruluk (Accuracy: %93.19), precision (%96.65), ve F1-Score (%94.29) değerlerine ulaşmıştır. Aynı zamanda modelin ROC eğrisi altında kalan alan (AUC: 0.94) oldukça yüksektir, bu da sınıflandırma başarısının güçlü olduğunu göstermektedir.

- **Logistic Regression:** Diğer modellere kıyasla en düşük performansı sergilemiştir. Özellikle recall (%44.45) ve F1-Score (%55.02) değerlerinin düşük olması, bu modelin sınıf dengesizliğinden etkilenme potansiyelini ortaya koymaktadır.

💡 LightGBM modelinin Logistic Regression’a kıyasla daha yüksek doğruluk, daha yüksek AUC skoru ve daha düşük hata oranına sahip olduğu, bu nedenle kredi risk tahmini için daha uygun olduğu sonucuna varılmıştır

---

## 🛠 Kullanılan Teknolojiler
- **Python** – Veri analizi ve modelleme
- **Pandas, NumPy** – Veri işleme
- **Matplotlib, Seaborn** – Görselleştirme
- **Scikit-learn** – Makine öğrenmesi algoritmaları
- **LightGBM, XGBoost** – Gelişmiş modelleme

---

## 🚀 Sonraki Adımlar & Geliştirmeler
- **XGBoost** ve **CatBoost** gibi diğer boosting algoritmaları ile karşılaştırma.  
- Yeni değişkenler türeterek **feature engineering** geliştirme.  
- Modelin bir **REST API** olarak canlı ortama taşınması.  
- Daha geniş veri setleriyle test edilmesi.  

## ❓ Soru / İletişim

Herhangi bir sorunuz varsa lütfen [GitHub Issues](https://github.com/zeyneppmk/Credit-Risk-Analysis/issues) bölümünden yeni bir issue açabilirsiniz.

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
