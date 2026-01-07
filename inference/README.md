* Çalışma Prensibi

Konfigürasyon bilgileri .toml dosyasından okunur

İlgili model mimarisi seçilir

Kayıtlı model.pth dosyası yüklenir

Verilen görüntü üzerinden tahmin ve güven skoru hesaplanır

* Kullanım

Sanal ortam aktifken;

python run.py --config test_base.toml

* Konfigürasyon 

test_base.toml dosyasında aşağıdaki alanlar bulunmalıdır:

pred_model : Kullanılacak model türü

weights_path : Eğitilmiş model dosyasının yolu

image_path : Tahmin yapılacak görüntü

image_size : Girdi boyutu

num_class : Sınıf sayısı