# ⚡ HIZLI BAŞLANGIÇ

## 📦 Gereksinimler

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow scipy
```

## 🚀 Kullanım

### 1. Temel Kullanım

```bash
python weather_prediction_improved.py
```

Program size tarih soracak:
```
Tahmin edilecek tarihi girin (GG.AA.YYYY): 29.10.2024
```

### 2. Parametreleri Özelleştirme

Kod dosyasını açın ve `Config` sınıfını düzenleyin:

```python
class Config:
    # Model kaç kere eğitilsin?
    ENSEMBLE_SIZE = 3  # Varsayılan: 3, Daha iyi sonuç için: 5

    # Kaç epoch eğitim?
    EPOCHS = 50  # Varsayılan: 50, Daha uzun eğitim: 100

    # Veri temizleme ne kadar sıkı?
    OUTLIER_Z_THRESHOLD = 3.0  # Daha sıkı: 2.5, Daha gevşek: 3.5

    # Soğuk hava eşikleri (°C)
    EXTREME_COLD = -3.0  # Ekstrem soğuk
    VERY_COLD = 1.0      # Çok soğuk
    FROST = -5.0         # Don
```

## 📊 Çıktı

### Konsol Çıktısı

```
🧹 Veri temizleniyor...
  NaN temizleme: 50000 → 49800
✅ Temizleme tamamlandı: 50000 → 48500 (3.0% kayıp)

🔧 Özellikler oluşturuluyor...

📊 GEÇMİŞ 29 EKİM VERİLERİ
════════════════════════════════════════════════════════════
2014: Min=5.2°C, Max=12.8°C, Ort=9.1°C
2015: Min=4.8°C, Max=13.2°C, Ort=8.9°C
...
📈 SON 5 YIL: Min=4.5°C, Max=14.2°C, Ort=9.3°C
════════════════════════════════════════════════════════════

🤖 3 model eğitiliyor...
  Model 1/3...
  Model 2/3...
  Model 3/3...
✅ Model eğitimi tamamlandı!

🔍 Geçmiş performans analizi...
✅ 24 saat için bias hesaplandı

🎯 Tahminler yapılıyor...
🧠 Derin öğrenme modeli çalışıyor...

🤖 Model - Gece: 6.2°C, Gündüz: 11.5°C
📋 Strateji: standard

🔄 Adaptive post-processing...

════════════════════════════════════════════════════════════
🌡️  TAHMİN SONUÇLARI
════════════════════════════════════════════════════════════
00:20 → 7.2°C (bias: +0.15°C)
00:50 → 7.0°C (bias: +0.15°C)
01:20 → 6.8°C (bias: +0.12°C)
...
23:50 → 8.1°C (bias: +0.18°C)
════════════════════════════════════════════════════════════

📊 PERFORMANS METRİKLERİ
   MAE:  1.85°C
   RMSE: 2.32°C
   2°C altı tahminler: 38/48 (79.2%)
   Max hata: 4.12°C
   Median hata: 1.65°C

✅ Tahmin tamamlandı!
```

### Görselleştirme

Program otomatik olarak 4 grafik gösterir:
1. **Gerçek vs Tahmin**: Zaman serisinde karşılaştırma
2. **Scatter Plot**: Korelasyon analizi
3. **Hata Dağılımı**: Histogram
4. **Zamana Göre Hata**: Hangi saatlerde daha iyi/kötü

## 🔧 Yaygın Sorunlar

### ❌ CSV dosyası bulunamadı

```
FileNotFoundError: [Errno 2] No such file or directory: 'weatherdataaax2.csv'
```

**Çözüm**: CSV dosyalarını script ile aynı klasöre koyun:
```
senem/
├── weather_prediction_improved.py
├── weatherdataaax2.csv
└── weatherdatax.csv
```

### ❌ Yeterli veri yok

```
❌ Yeterli veri bulunamadı!
```

**Çözüm**: Config'de `MIN_RECORDS` değerini azaltın:
```python
MIN_RECORDS = 200  # 500'den 200'e düşür
```

### ❌ Çok fazla veri kayboldu

```
✅ Temizleme tamamlandı: 50000 → 15000 (70% kayıp)
```

**Çözüm**: Outlier eşiklerini gevşetin:
```python
OUTLIER_Z_THRESHOLD = 4.0  # 3.0'dan 4.0'a
IQR_MULTIPLIER = 2.5       # 1.8'den 2.5'e
```

### ❌ Model çok yavaş eğitiliyor

**Çözüm**: Parametreleri azaltın:
```python
ENSEMBLE_SIZE = 1  # 3'ten 1'e
EPOCHS = 30        # 50'den 30'a
K_FOLDS = 3        # 5'ten 3'e
```

### ❌ Tahminler çok kötü

**Çözüm 1**: Daha fazla model eğitin:
```python
ENSEMBLE_SIZE = 5
EPOCHS = 100
```

**Çözüm 2**: Bias hesaplamayı güçlendirin:
```python
BIAS_HISTORY_DAYS = 365  # 180'den 365'e
```

**Çözüm 3**: Post-processing stratejilerini ayarlayın:
```python
WEIGHT_TIME = 0.5      # 0.4'ten 0.5'e (zamana daha çok önem ver)
WEIGHT_WEATHER = 0.3   # 0.2'den 0.3'e (hava benzerliğine önem ver)
```

## 🎯 En İyi Sonuçlar İçin

### Soğuk Hava (Kış)
```python
EXTREME_COLD = -5.0    # Daha düşük eşik
BIAS_HISTORY_DAYS = 365  # Daha fazla geçmiş
WEIGHT_YEAR = 0.4      # Son yıllara daha çok önem
```

### Sıcak Hava (Yaz)
```python
TEMP_TOLERANCE_NORMAL = 5.0  # Daha geniş tolerans
WEIGHT_WEATHER = 0.3         # Hava benzerliği önemli
```

### Geçiş Mevsimleri (İlkbahar/Sonbahar)
```python
BIAS_HISTORY_DAYS = 90   # Son 3 ay yeterli
WEIGHT_TIME = 0.5        # Zamana öncelik
```

## 💡 İpuçları

1. **İlk çalıştırma yavaştır**: Model eğitimi 5-10 dakika sürebilir
2. **Veri kalitesi önemli**: Temiz veri = İyi tahmin
3. **Geçmiş veriye bakın**: Konsol çıktısında geçmiş yıl istatistiklerini kontrol edin
4. **Stratejiye dikkat**: Program hangi stratejiyi kullandığını gösterir
5. **Bias değerlerini izleyin**: Pozitif bias = Model düşük tahmin ediyor

## 📞 Destek

**Detaylı değişiklikler**: `DEGISIKLIKLER.md` dosyasına bakın

**Kod yapısı**: Dosya başındaki "İçindekiler" bölümüne bakın

**Parametre açıklamaları**: Config sınıfındaki yorumlara bakın
