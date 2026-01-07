Image Classification Framework

1. Projenin Amacı

Bu proje, modern derin öğrenme tabanlı görüntü sınıflandırma modellerinin tek bir eğitim altyapısı altında, karşılaştırılabilir ve tekrar üretilebilir biçimde eğitilmesini ve değerlendirilmesini amaçlamaktadır.

Framework; Vision Transformer (ViT), Swin Transformer, ConvNeXt ve EfficientNet gibi güncel mimarileri destekleyecek şekilde tasarlanmıştır. Tüm deneyler konfigürasyon dosyaları (TOML) üzerinden yönetilmekte, böylece kod değişikliği yapılmadan farklı deney senaryoları çalıştırılabilmektedir.

Proje olanakları;Deney tekrar edilebilirliğini sağlamak. Eğitim ve test süreçlerini standartlaştırmak. Model, optimizer ve scheduler.eçimlerini soyutlamak.

Deney çıktılarının sistematik biçimde kaydedilmesi

2. Proje Mimarisi

Proje, modüler bir yapı izlemektedir. Her bileşen tek bir sorumluluğa sahiptir.

framework/
│
├── train.py
│   Ana giriş noktasıdır. CLI üzerinden config dosyası alır ve
│   tüm eğitim sürecini başlatır.
│
├── pipeline.py
│   Eğitim, doğrulama ve test döngülerini yöneten ana akıştır.
│
├── configs/
│   Model ve deney ayarlarının bulunduğu TOML dosyaları.
│
├── model/
│   Model üretiminden sorumludur (factory yaklaşımı).
│
├── training/
│   Loss, optimizer, scheduler ve config doğrulama bileşenleri.
│
├── data_loading.py
│   Dataset, DataLoader ve Data örnek görsel oluşturma tanımlarını içerir. 
│
├── utils/
│   Konfigürasyon yükleme ve deney klasörü yönetimi.
│
├── runs/
│   Deney çıktılarının otomatik kaydedildiği dizin.Otomatşk oluşturulur.
│
├── requirements.txt/
│   Projede kullanılan tüm bağımlılıkları içerir.
│
├── setup.ps1/
│   Windows power shell dosyası ihtiyaç duyulan sanal ortam ve kütüphane kurulumu gerçekleştirir.



Bu yapı sayesinde yeni bir model veya eğitim stratejisi eklemek mevcut kodu bozmadan mümkündür.

3. Kurulum Ortamı

Proje Windows işletim sistemi ve Python 3.11 üzerinde test edilmiştir.

3.1 Manuel Kurulum

3.1.1 Python Sürümü
python --version


Python 3.11 önerilir. Farklı sürümlerle çalışabilir ancak garanti edilmez. Eğer farklı süürüm ile çalışacaksanız "utils/config.py" dosyasından python sürümünüze göre toml kütüphanesini uygun şekilde güncelleyiniz.

3.1.2 Sanal Ortam
python -m venv .venv
.venv\Scripts\activate

3.1.3 Gerekli Kütüphaneler
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt


3.2 Otomatik Kurulum

Windows varsayılan olarak .ps1 çalıştırmayı engeller.Bu hata değil, güvenlik politikasıdır.(script çalıştırma izni ver tek seferliktir)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
PowerShell oturumu için izin verir. Sistemi kalıcı olarak değiştirmez.
.\setup.ps1 => setup.ps1 çalıştırma



4. Konfigürasyon Dosyaları (TOML)

Tüm deney ayarları .toml dosyaları üzerinden yapılır. Kod içerisinde hard-coded hiperparametre bulunmamaktadır.

Örnek yapı:

[system]
device = "cpu"

[data]
batch_size = 4
image_size = 224
visualize = true
num_sample_image_show = 2

[model]
name = "Swin"
train_type = 1


Bu yaklaşım:

Deneylerin izlenebilirliğini artırır. Hiperparametre karmaşasını azaltır. Raporlamada kolaylık sağlr.


5. Model Yönetimi

Model seçimi model_factory.py üzerinden yapılmaktadır. Bu dosya, factory pattern kullanarak seçilen modele göre uygun yapıyı üretir.

Desteklenen mimariler:

Vision Transformer (ViT)

Swin Transformer

ConvNeXt

EfficientNet

Yeni bir model eklemek için:

Model tanımını eklemek

Factory fonksiyonuna bağlamak

Schema dosyasına model ismini ekleme

Yeni bir .toml dosyası oluşturmak yeterlidir

6. Eğitim ve Test Süreci

Eğitim akışı Pipeline sınıfı tarafından yönetilir. Bu sınıf:

Eğitim döngüsü

Doğrulama değerlendirmesi

Learning rate scheduler

Early stopping

Test aşaması

işlemlerini tek bir yapı altında toplar.

Bu sayede eğitim kodu dağınık hale gelmez ve deney süreci okunabilir kalır.

7. Değerlendirme ve Çıktılar

Test aşamasında aşağıdaki çıktılar otomatik üretilir:

Accuracy ve loss değerleri 

%10’luk güven aralığı inceleme

Normalize edilmiş confusion matrix

O anki çalıştırma da en iyi performans gösterdiği iterasyondaki  model ağırlıkları

Tüm çıktılar deney bazlı bir klasöre kaydedilir, bu klasör otomatik oluşturulur(çalıştırılma-zamanı-tarih-saat_model-adi):

runs/
└── 20240118_213045_Swin/
    ├── config.toml
    ├── confusion_matrix.png
    └── confidence.txt


Bu yapı, farklı deneylerin karşılaştırılmasını kolaylaştırır. Aynı zamanda deney karmaşasını önler.

8. Görselleştirme

Veri artırma (augmentation) sonrası örnek görseller, konfigürasyon üzerinden opsiyonel olarak gösterilebilir.


9. Sonuç


Bu framework, deneysel derin öğrenme çalışmalarında:
Kod tekrarını azaltmayı, deney yönetimini standartlaştırmayı amaçlayan bir altyapı sunar.