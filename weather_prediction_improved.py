# -*- coding: utf-8 -*-
"""
Geliştirilmiş Hava Durumu Tahmin Sistemi
Adaptive Learning ile Sıcaklık Tahmini

@author: Senem
@improved: Claude Code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# SABİT DEĞERLER VE KONFIGÜRASYON
# =============================================================================

class Config:
    """Tüm sistem parametreleri"""

    # Dosya yolları
    RAW_DATA_PATH = 'weatherdataaax2.csv'
    VALIDATION_DATA_PATH = 'weatherdatax.csv'

    # Veri temizleme
    OUTLIER_Z_THRESHOLD = 3.0
    IQR_MULTIPLIER = 1.8
    TEMP_ANOMALY_WINDOW = 48
    TEMP_ANOMALY_STD = 2.8
    SMOOTH_TEMP_THRESHOLD = 12.0

    # Model
    TIME_STEPS = 48
    ENSEMBLE_SIZE = 3
    K_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 10
    LEARNING_RATE = 0.0005

    # Adaptive Learning
    BIAS_HISTORY_DAYS = 180
    BIAS_MIN_SAMPLES = 5
    BIAS_MIN_SEQUENCE = 5

    # Ağırlıklandırma
    WEIGHT_TIME = 0.4
    WEIGHT_YEAR = 0.3
    WEIGHT_WEATHER = 0.2
    WEIGHT_WEEKEND = 0.1

    # Post-processing
    TEMP_MIN_LIMIT = -20.0
    TEMP_MAX_LIMIT = 50.0
    MAX_HOURLY_CHANGE = 2.0
    SMOOTHING_WINDOW = 3

    # Sıcaklık eşikleri
    EXTREME_COLD = -3.0
    VERY_COLD = 1.0
    FROST = -5.0

    # Filtreleme
    DAYS_BEFORE = 4
    DAYS_AFTER = 1
    MIN_RECORDS = 500

    # Özellikler
    BASIC_FEATURES = [
        'Dew Point', 'Humidity', 'Wind Speed', 'Pressure', 'Temperature',
        'Temp_Change_1H', 'Temp_Change_6H', 'Temp_Change_24H', 'Temp_Change_Avg',
        'Hour_Temp_Deviation', 'Temperature_EMA_7', 'Temperature_EMA_14', 'Temperature_EMA_30'
    ]


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def fahrenheit_to_celsius(f):
    """Fahrenheit'i Celsius'a çevir"""
    return (f - 32) * 5.0 / 9.0


def get_season(month):
    """Aydan mevsim belirle (1:kış, 2:ilkbahar, 3:yaz, 4:sonbahar)"""
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4


def create_cyclical_features(df, col_name, period, start_num=0):
    """Döngüsel özellikleri sin/cos ile kodla"""
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * (df[col_name] - start_num) / period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    return df


def create_sequences(X, y, time_steps):
    """Zaman serisi sekansları oluştur"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# =============================================================================
# VERİ YÜKLEME VE TEMİZLEME
# =============================================================================

def load_and_convert_data(file_path):
    """Ham veriyi yükle ve temel dönüşümleri yap"""
    df = pd.read_csv(file_path)

    # Sütun dönüşümleri
    df['Temperature'] = df['Temperature'].str.replace(' °F', '', regex=False).astype(float)
    df['Temperature'] = df['Temperature'].apply(fahrenheit_to_celsius).round(1)

    if 'Dew Point' in df.columns:
        df['Dew Point'] = df['Dew Point'].str.replace(' °F', '', regex=False).astype(float)
    if 'Humidity' in df.columns:
        df['Humidity'] = df['Humidity'].str.replace(' %', '', regex=False).astype(float)
    if 'Wind Speed' in df.columns:
        df['Wind Speed'] = df['Wind Speed'].str.replace(' mph', '', regex=False).astype(float)
    if 'Pressure' in df.columns:
        df['Pressure'] = df['Pressure'].str.replace(' in', '', regex=False).astype(float)

    # Datetime oluştur
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df.drop(columns=['Date', 'Time'], inplace=True)

    return df


def smooth_temperature_anomalies(data, temp_col='Temperature', threshold=None):
    """Ani sıcaklık değişimlerini yumuşat"""
    if threshold is None:
        threshold = Config.SMOOTH_TEMP_THRESHOLD

    df = data.copy().sort_values('Datetime').reset_index(drop=True)
    temp_diff = df[temp_col].diff().abs()

    for idx in df.index[1:]:
        if temp_diff.loc[idx] > threshold:
            prev_val = df.loc[idx - 1, temp_col]
            next_idx = idx + 1

            if next_idx in df.index:
                next_val = df.loc[next_idx, temp_col]
                df.loc[idx, temp_col] = (prev_val + next_val) / 2
            else:
                df.loc[idx, temp_col] = prev_val

    return df


def detect_outliers_zscore(data, column, threshold=None):
    """Z-score ile outlier tespit et"""
    if threshold is None:
        threshold = Config.OUTLIER_Z_THRESHOLD

    df_clean = data.copy()
    z_scores = np.abs(stats.zscore(df_clean[column]))
    df_clean = df_clean[z_scores < threshold]
    return df_clean


def remove_outliers_iqr(df, columns, multiplier=None):
    """IQR yöntemiyle outlier temizle"""
    if multiplier is None:
        multiplier = Config.IQR_MULTIPLIER

    df_clean = df.copy()
    outlier_indices = set()

    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outliers = df_clean[
                (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            ].index
            outlier_indices.update(outliers)

    df_clean = df_clean.drop(list(outlier_indices)).reset_index(drop=True)
    return df_clean


def detect_temperature_anomalies(df, window_size=None, std_multiplier=None):
    """Rolling window ile anomali tespit et"""
    if window_size is None:
        window_size = Config.TEMP_ANOMALY_WINDOW
    if std_multiplier is None:
        std_multiplier = Config.TEMP_ANOMALY_STD

    df_clean = df.copy()

    df_clean['temp_rolling_mean'] = df_clean['Temperature'].rolling(
        window=window_size, center=True, min_periods=1
    ).mean()

    df_clean['temp_rolling_std'] = df_clean['Temperature'].rolling(
        window=window_size, center=True, min_periods=1
    ).std()

    df_clean['temp_deviation'] = abs(df_clean['Temperature'] - df_clean['temp_rolling_mean'])
    threshold = df_clean['temp_rolling_std'] * std_multiplier

    anomaly_indices = df_clean[df_clean['temp_deviation'] > threshold].index
    df_clean = df_clean.drop(anomaly_indices).reset_index(drop=True)
    df_clean = df_clean.drop(columns=['temp_rolling_mean', 'temp_rolling_std', 'temp_deviation'])

    return df_clean


def clean_data_pipeline(df):
    """Tam veri temizleme pipeline'ı"""
    print("🧹 Veri temizleniyor...")
    initial_count = len(df)

    # 1. NaN temizle
    df = df.dropna().reset_index(drop=True)
    print(f"  NaN temizleme: {initial_count} → {len(df)}")

    # 2. Ani değişimleri yumuşat
    df = smooth_temperature_anomalies(df)

    # 3. Outlier temizleme
    numeric_cols = ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        df = detect_outliers_zscore(df, col)

    df = remove_outliers_iqr(df, numeric_cols)
    df = detect_temperature_anomalies(df)

    print(f"✅ Temizleme tamamlandı: {initial_count} → {len(df)} "
          f"({100 * (1 - len(df) / initial_count):.1f}% kayıp)\n")

    return df


# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ
# =============================================================================

def add_time_features(df):
    """Zaman tabanlı özellikler ekle"""
    df = df.copy()

    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    df['IsDayTime'] = ((df['Hour'] >= 6) & (df['Hour'] <= 18)).astype(int)
    df['Season'] = df['Month'].apply(get_season)

    # Günün zamanı
    df['TimeOfDay'] = 0
    df.loc[(df['Hour'] >= 6) & (df['Hour'] < 12), 'TimeOfDay'] = 1
    df.loc[(df['Hour'] >= 12) & (df['Hour'] < 18), 'TimeOfDay'] = 2
    df.loc[(df['Hour'] >= 18) & (df['Hour'] < 24), 'TimeOfDay'] = 3

    # Yıl ağırlığı
    year_min = df['Datetime'].dt.year.min()
    year_max = df['Datetime'].dt.year.max()
    if year_max > year_min:
        df['YearWeight'] = (df['Datetime'].dt.year - year_min) / (year_max - year_min)
    else:
        df['YearWeight'] = 0.5

    return df


def add_temperature_features(df):
    """Sıcaklık tabanlı özellikler ekle"""
    df = df.sort_values('Datetime').copy()

    # Sıcaklık değişimleri
    df['Temp_Change_1H'] = df['Temperature'].diff(2).fillna(0)
    df['Temp_Change_6H'] = df['Temperature'].diff(12).fillna(0)
    df['Temp_Change_24H'] = df['Temperature'].diff(48).fillna(0)
    df['Temp_Change_Avg'] = df[['Temp_Change_1H', 'Temp_Change_6H', 'Temp_Change_24H']].mean(axis=1)

    # EMA
    df['Temperature_EMA_7'] = df.groupby(['Month', 'Day'])['Temperature'].transform(
        lambda x: x.ewm(span=120, adjust=False).mean()
    )
    df['Temperature_EMA_14'] = df.groupby(['Month', 'Day'])['Temperature'].transform(
        lambda x: x.ewm(span=360, adjust=False).mean()
    )
    df['Temperature_EMA_30'] = df.groupby(['Month', 'Day'])['Temperature'].transform(
        lambda x: x.ewm(span=480, adjust=False).mean()
    )

    # Saatlik sapma
    hourly_avg = df.groupby('Hour')['Temperature'].mean()
    df['Hour_Temp_Deviation'] = df.apply(
        lambda x: x['Temperature'] - hourly_avg[x['Hour']], axis=1
    )

    return df


def add_cyclical_features(df):
    """Döngüsel özellikleri ekle"""
    df = create_cyclical_features(df, 'Month', 12, 1)
    df = create_cyclical_features(df, 'Hour', 24, 0)
    df = create_cyclical_features(df, 'DayOfYear', 365, 1)
    df = create_cyclical_features(df, 'TimeOfDay', 3, 1)
    return df


# =============================================================================
# VERİ FİLTRELEME
# =============================================================================

def get_previous_day_temps(target_date):
    """Önceki günün min/max sıcaklıklarını al"""
    yesterday = target_date - timedelta(days=1)

    try:
        raw_df = load_and_convert_data(Config.RAW_DATA_PATH)
        yesterday_data = raw_df[raw_df['Datetime'].dt.date == yesterday.date()]

        if len(yesterday_data) > 0:
            return yesterday_data['Temperature'].min(), yesterday_data['Temperature'].max()
    except:
        pass

    return None, None


def smart_filter_data(df, target_datetime):
    """
    Akıllı veri filtreleme:
    1. Önce tarihe özel geçmiş verileri al
    2. Yeterli değilse benzer hava koşullarına bak
    """
    target_month = target_datetime.month
    target_day = target_datetime.day

    # 1. Aynı tarihin geçmiş yılları (±2 gün)
    historical = df[
        (df['Month'] == target_month) &
        (abs(df['Day'] - target_day) <= 2)
    ]

    print(f"📅 Tarih bazlı filtre: {len(historical)} kayıt")

    if len(historical) >= Config.MIN_RECORDS:
        print("✅ Tarih bazlı veri yeterli")
        return historical

    # 2. Benzer hava koşulları
    try:
        min_prev, max_prev = get_previous_day_temps(target_datetime)

        if min_prev is not None:
            # Sıcaklık toleransı belirle
            if min_prev < 0:
                tolerance = 1.5
            elif min_prev < 5:
                tolerance = 2.0
            else:
                tolerance = 3.0

            print(f"📍 Önceki gün: Min {min_prev:.1f}°C → Tolerans ±{tolerance}°C")

            # Benzer günleri bul (son 5 yıl + aynı ay)
            raw_df = load_and_convert_data(Config.RAW_DATA_PATH)
            raw_df = add_time_features(raw_df)

            similar_evenings = raw_df[
                (raw_df['Hour'] == 20) &
                (raw_df['Month'] == target_month) &
                (raw_df['Datetime'].dt.year >= 2019) &
                (abs(raw_df['Temperature'] - min_prev) <= tolerance)
            ]

            next_dates = [(d + timedelta(days=1)).date() for d in similar_evenings['Datetime']]
            filtered = df[df['Datetime'].dt.date.isin(next_dates)]

            print(f"📊 Benzer günler: {len(set(next_dates))} gün, {len(filtered)} kayıt")

            if len(filtered) > 100:
                return filtered
    except Exception as e:
        print(f"⚠️ Benzer gün arama hatası: {e}")

    # 3. Klasik yöntem
    print("⚠️ Klasik filtrelemeye dönüldü")
    year_ranges = list(range(2014, 2025))
    past_data = []

    for year in year_ranges:
        try:
            start = datetime(year, target_month, target_day) - timedelta(days=Config.DAYS_BEFORE)
            end = datetime(year, target_month, target_day) + timedelta(days=Config.DAYS_AFTER)
            yearly = df[(df['Datetime'] >= start) & (df['Datetime'] <= end)]
            if not yearly.empty:
                past_data.append(yearly)
        except ValueError:
            continue

    # Son 2 yılın mevsim verisi ekle
    target_season = get_season(target_month)
    season_data = df[
        (df['Season'] == target_season) &
        (df['Datetime'].dt.year.isin([2023, 2024]))
    ]
    if not season_data.empty:
        past_data.append(season_data)

    return pd.concat(past_data, ignore_index=True) if past_data else pd.DataFrame()


# =============================================================================
# GEÇMİŞ VERİ ANALİZİ
# =============================================================================

def analyze_historical_data(target_date):
    """Geçmiş verileri analiz et ve istatistikleri döndür"""
    raw_df = load_and_convert_data(Config.RAW_DATA_PATH)

    target_month = target_date.month
    target_day = target_date.day

    historical = raw_df[
        (raw_df['Datetime'].dt.month == target_month) &
        (raw_df['Datetime'].dt.day == target_day)
    ]

    month_names = ['', 'OCAK', 'ŞUBAT', 'MART', 'NİSAN', 'MAYIS', 'HAZİRAN',
                   'TEMMUZ', 'AĞUSTOS', 'EYLÜL', 'EKİM', 'KASIM', 'ARALIK']

    print("\n" + "=" * 60)
    print(f"📊 GEÇMİŞ {target_day} {month_names[target_month]} VERİLERİ")
    print("=" * 60)

    historical_stats = {}

    if len(historical) > 0:
        # Yıllık istatistikler
        for year in sorted(historical['Datetime'].dt.year.unique()):
            year_data = historical[historical['Datetime'].dt.year == year]
            if len(year_data) > 0:
                min_t = year_data['Temperature'].min()
                max_t = year_data['Temperature'].max()
                mean_t = year_data['Temperature'].mean()
                print(f"{year}: Min={min_t:.1f}°C, Max={max_t:.1f}°C, Ort={mean_t:.1f}°C")

        # Genel istatistikler
        historical_stats['all_min'] = historical['Temperature'].min()
        historical_stats['all_max'] = historical['Temperature'].max()
        historical_stats['all_mean'] = historical['Temperature'].mean()

        # Son 5 yıl
        recent = historical[historical['Datetime'].dt.year >= 2019]
        if len(recent) > 0:
            historical_stats['recent_min'] = recent['Temperature'].min()
            historical_stats['recent_max'] = recent['Temperature'].max()
            historical_stats['recent_mean'] = recent['Temperature'].mean()
            historical_stats['recent_median'] = recent['Temperature'].median()
        else:
            historical_stats['recent_min'] = historical_stats['all_min']
            historical_stats['recent_max'] = historical_stats['all_max']
            historical_stats['recent_mean'] = historical_stats['all_mean']
            historical_stats['recent_median'] = historical_stats['all_mean']

        # Tipik günlük değişim
        daily_ranges = []
        for year in historical['Datetime'].dt.year.unique():
            y_data = historical[historical['Datetime'].dt.year == year]
            if len(y_data) > 0:
                daily_ranges.append(y_data['Temperature'].max() - y_data['Temperature'].min())

        historical_stats['daily_range'] = np.median(daily_ranges) if daily_ranges else 10.0

        print(f"\n📈 TÜM YILLAR: Min={historical_stats['all_min']:.1f}°C, "
              f"Max={historical_stats['all_max']:.1f}°C, Ort={historical_stats['all_mean']:.1f}°C")
        print(f"📈 SON 5 YIL: Min={historical_stats['recent_min']:.1f}°C, "
              f"Max={historical_stats['recent_max']:.1f}°C, Ort={historical_stats['recent_mean']:.1f}°C")
        print(f"📈 Tipik günlük değişim: {historical_stats['daily_range']:.1f}°C")
    else:
        historical_stats = None

    print("=" * 60 + "\n")

    return historical_stats


# =============================================================================
# ADAPTIVE LEARNING
# =============================================================================

def calculate_hourly_bias(weather_data, models, all_features, target_date):
    """Geçmiş performansa bakarak saatlik bias hesapla"""
    print("\n🔍 Geçmiş performans analizi...")

    try:
        # Hedef tarihten önceki veriler
        historical = weather_data[weather_data['Datetime'] < target_date].copy()

        if len(historical) < 200:
            print("⚠️ Yeterli geçmiş veri yok")
            return {}

        # Son 180 günü test et
        cutoff = historical['Datetime'].max() - timedelta(days=Config.BIAS_HISTORY_DAYS)
        test_data = historical[historical['Datetime'] >= cutoff].copy()

        if len(test_data) < 100:
            print("⚠️ Yeterli test verisi yok")
            return {}

        hourly_bias = {}

        for hour in range(24):
            hour_data = test_data[test_data['Hour'] == hour].copy()

            if len(hour_data) < Config.BIAS_MIN_SAMPLES:
                continue

            # Sekans oluştur
            available_features = [f for f in all_features if f in hour_data.columns]
            X = hour_data[available_features].values
            y_true = hour_data['Temperature'].values

            min_steps = min(24, len(X) - 1)
            if min_steps < Config.BIAS_MIN_SEQUENCE:
                continue

            X_seq, y_seq = [], []
            for i in range(len(X) - min_steps):
                X_seq.append(X[i:(i + min_steps)])
                y_seq.append(y_true[i + min_steps])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            if len(X_seq) == 0:
                continue

            # Ensemble tahmin
            predictions = []
            for model in models:
                try:
                    pred = model.predict(X_seq, verbose=0)
                    predictions.append(pred)
                except:
                    continue

            if len(predictions) == 0:
                continue

            ensemble_pred = np.mean(predictions, axis=0).flatten()
            bias = np.median(y_seq - ensemble_pred)
            hourly_bias[hour] = bias

        print(f"✅ {len(hourly_bias)} saat için bias hesaplandı")
        return hourly_bias

    except Exception as e:
        print(f"⚠️ Bias hesaplama hatası: {e}")
        return {}


def calculate_weather_similarity(data, target_temp, target_humidity, target_pressure):
    """Hava koşulu benzerliğine göre ağırlık hesapla"""
    temp_diff = abs(data['Temperature'] - target_temp)
    humidity_diff = abs(data['Humidity'] - target_humidity)
    pressure_diff = abs(data['Pressure'] - target_pressure)

    similarity = np.exp(
        -(temp_diff / 10) ** 2 -
        (humidity_diff / 20) ** 2 -
        (pressure_diff / 5) ** 2
    )
    return similarity


def smart_weighted_average(filtered_data, target_hour, target_minute, target_datetime):
    """Çok katmanlı akıllı ağırlıklandırma"""
    if filtered_data.empty:
        return None

    data = filtered_data.copy()

    # 1. Zaman benzerliği
    time_dist = abs(data['Hour'] - target_hour) + abs(data['Minute'] - target_minute) / 60
    data['time_weight'] = np.exp(-((time_dist / 2) ** 2))

    # 2. Yıl ağırlığı (son yıllara öncelik)
    current_year = data['Datetime'].dt.year.max()
    years_ago = current_year - data['Datetime'].dt.year
    data['year_weight'] = np.exp(-years_ago / 2)

    # 3. Hava benzerliği
    if 'Humidity' in data.columns and len(data) > 0:
        median_temp = data['Temperature'].median()
        median_humidity = data['Humidity'].median()
        median_pressure = data['Pressure'].median()
        data['weather_sim'] = calculate_weather_similarity(
            data, median_temp, median_humidity, median_pressure
        )
    else:
        data['weather_sim'] = 1.0

    # 4. Hafta içi/sonu
    data['is_weekend'] = data['Datetime'].dt.dayofweek >= 5
    target_is_weekend = target_datetime.weekday() >= 5
    data['weekend_weight'] = np.where(data['is_weekend'] == target_is_weekend, 1.2, 1.0)

    # Gece saatleri boost
    if 0 <= target_hour <= 6:
        data = data[data['Datetime'].dt.year >= 2023]
        data['time_weight'] = data['time_weight'] * 1.5

    # Toplam ağırlık
    data['combined_weight'] = (
        data['time_weight'] * Config.WEIGHT_TIME +
        data['year_weight'] * Config.WEIGHT_YEAR +
        data['weather_sim'] * Config.WEIGHT_WEATHER +
        data['weekend_weight'] * Config.WEIGHT_WEEKEND
    )

    # Normalize
    data['combined_weight'] = data['combined_weight'] / data['combined_weight'].sum()

    return data


# =============================================================================
# POST-PROCESSING (BASİTLEŞTİRİLMİŞ)
# =============================================================================

def determine_strategy(avg_night, avg_day, hist_stats, min_prev, max_prev):
    """Post-processing stratejisini belirle"""
    if hist_stats is None:
        return 'standard', {}

    hist_mean = hist_stats['recent_mean']
    hist_min = hist_stats['recent_min']
    hist_max = hist_stats['recent_max']

    model_too_cold = avg_night < hist_min - 1.0
    model_too_warm = avg_day > hist_max + 1.0

    is_prev_extreme = min_prev is not None and min_prev < Config.EXTREME_COLD
    is_prev_very_cold = min_prev is not None and min_prev < Config.VERY_COLD

    context = {
        'hist_mean': hist_mean,
        'hist_min': hist_min,
        'hist_max': hist_max,
        'typical_range': hist_stats['daily_range']
    }

    # Strateji seçimi
    if is_prev_extreme and model_too_cold:
        return 'warm_to_history', context
    elif is_prev_very_cold and model_too_cold:
        return 'warm_to_history', context
    elif is_prev_very_cold and not model_too_cold:
        if abs(avg_day - hist_mean) < 2.0:
            return 'minimal', context
        else:
            return 'gentle', context
    elif is_prev_very_cold and model_too_warm:
        return 'validate_warming', context
    else:
        return 'standard', context


def apply_bias_correction(predictions, times, hourly_bias, strength=1.0):
    """Saatlik bias düzeltmesi uygula"""
    corrected = predictions.copy()

    for i, time in enumerate(times):
        hour = time.hour
        if hour in hourly_bias:
            corrected[i] += hourly_bias[hour] * strength

    return corrected


def apply_monthly_limits(predictions, times, target_month, max_allowed=None):
    """Aylık maksimum limitleri uygula"""
    monthly_limits = {10: 16.0, 11: 12.0}

    if target_month in monthly_limits:
        limit = monthly_limits[target_month]
        if max_allowed is not None:
            limit = min(limit, max_allowed)

        for i, time in enumerate(times):
            if 12 <= time.hour <= 17:
                predictions[i] = min(predictions[i], limit)

    return predictions


def smooth_predictions(predictions):
    """Tahminleri yumuşat"""
    # 1. Ani değişimleri sınırla
    for i in range(1, len(predictions)):
        change = predictions[i] - predictions[i - 1]
        if abs(change) > Config.MAX_HOURLY_CHANGE:
            predictions[i] = predictions[i - 1] + np.sign(change) * Config.MAX_HOURLY_CHANGE

    # 2. Moving average
    if len(predictions) >= Config.SMOOTHING_WINDOW:
        smoothed = predictions.copy()
        for i in range(1, len(predictions) - 1):
            smoothed[i] = (
                0.2 * predictions[i - 1] +
                0.6 * predictions[i] +
                0.2 * predictions[i + 1]
            )
        predictions = smoothed

    return predictions


def adaptive_post_processing(predictions, times, hourly_bias, target_month,
                             min_prev=None, max_prev=None, hist_stats=None):
    """
    Basitleştirilmiş ve anlaşılır post-processing
    """
    processed = predictions.copy()

    # Gece/gündüz ortalamaları
    avg_night = np.mean(predictions[:14])
    avg_day = np.mean(predictions[24:34])

    print(f"🤖 Model - Gece: {avg_night:.1f}°C, Gündüz: {avg_day:.1f}°C")

    # Strateji belirle
    strategy, context = determine_strategy(avg_night, avg_day, hist_stats, min_prev, max_prev)
    print(f"📋 Strateji: {strategy}")

    # Strateji uygula
    if strategy == 'warm_to_history':
        # Pozitif bias'ı güçlendir, geçmiş minimumdan aşağı inme
        for i, time in enumerate(times):
            hour = time.hour

            if hour in hourly_bias:
                bias_val = hourly_bias[hour]
                processed[i] += bias_val * 3.0 if bias_val > 0 else bias_val * 0.5

            if 0 <= hour <= 7:
                min_allowed = max(context['hist_min'] - 2.0,
                                 min_prev - 3.0 if min_prev else -10)
                processed[i] = max(processed[i], min_allowed)

            elif 8 <= hour <= 17:
                expected = context['hist_mean']
                if max_prev:
                    expected = min(expected, max_prev + 4.0)
                processed[i] = min(processed[i], expected)

    elif strategy == 'minimal':
        # Minimal düzeltme
        processed = apply_bias_correction(processed, times, hourly_bias, strength=0.8)

    elif strategy == 'gentle':
        # Yumuşak düzeltme
        processed = apply_bias_correction(processed, times, hourly_bias, strength=1.5)

        for i, time in enumerate(times):
            if 12 <= time.hour <= 17:
                processed[i] += (context['hist_mean'] - processed[i]) * 0.2

    elif strategy == 'validate_warming':
        # Ani ısınma kontrolü
        processed = apply_bias_correction(processed, times, hourly_bias, strength=1.2)

        for i, time in enumerate(times):
            if 12 <= time.hour <= 17:
                max_allowed = context['hist_max'] + 1.0
                if max_prev:
                    max_allowed = min(max_allowed, max_prev + 5.0)
                processed[i] = min(processed[i], max_allowed)

    else:  # standard
        # Standart düzeltme
        is_frost = min_prev is not None and min_prev < Config.FROST
        is_very_cold = min_prev is not None and min_prev < Config.VERY_COLD

        if is_frost or is_very_cold:
            # Soğuk hava stratejisi
            base_corr = -3.0 if is_frost else -1.5

            for i, time in enumerate(times):
                hour = time.hour

                if 0 <= hour <= 7:
                    # Gece
                    mult = 6.0 if is_frost else 5.0
                    processed[i] += base_corr - abs(hourly_bias.get(hour, 0)) * mult
                    max_allow = max(min_prev + 1.0, -5.0) if is_frost else min_prev + 2.0
                    processed[i] = min(processed[i], max_allow)

                elif 8 <= hour <= 11:
                    # Sabah
                    mult = 5.0 if is_frost else 4.0
                    processed[i] += base_corr - abs(hourly_bias.get(hour, 0)) * mult
                    expected = min_prev + (hour - 7) * (1.2 if is_frost else 1.5)
                    processed[i] = min(processed[i], expected + 1.0)

                elif 12 <= hour <= 17:
                    # Öğle
                    mult = 4.0 if is_frost else 3.5
                    processed[i] += base_corr * 0.3 - abs(hourly_bias.get(hour, 0)) * mult

                    max_inc = 10.0 if is_frost else 12.0
                    max_day = min_prev + max_inc

                    if max_prev and max_prev < 15:
                        max_day = min(max_day, max_prev + 1.5)

                    processed[i] = min(processed[i], max_day)

                elif 18 <= hour <= 23:
                    # Akşam
                    mult = 3.0 if is_frost else 2.5
                    processed[i] += base_corr * 0.3 - abs(hourly_bias.get(hour, 0)) * mult

                    peak = max(processed[28:min(i, 35)]) if i >= 30 else (max_prev * 0.9 if max_prev else 10.0)
                    processed[i] = min(processed[i], peak)
        else:
            # Normal günler
            processed = apply_bias_correction(processed, times, hourly_bias, strength=1.2)

    # Aylık limitler
    processed = apply_monthly_limits(processed, times, target_month, max_prev)

    # Genel limitler
    processed = np.clip(processed, Config.TEMP_MIN_LIMIT, Config.TEMP_MAX_LIMIT)

    # Yumuşatma
    processed = smooth_predictions(processed)

    return processed


# =============================================================================
# MODEL OLUŞTURMA
# =============================================================================

def create_model(input_shape):
    """LSTM-GRU hibrit model oluştur"""
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(0.3))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mape'])

    return model


def train_ensemble_models(X_seq, y_seq):
    """Ensemble modelleri eğit"""
    models = []
    k_fold = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)

    print(f"\n🤖 {Config.ENSEMBLE_SIZE} model eğitiliyor...")

    for i in range(Config.ENSEMBLE_SIZE):
        print(f"  Model {i + 1}/{Config.ENSEMBLE_SIZE}...")

        for train_idx, val_idx in k_fold.split(X_seq):
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]

            model = create_model((Config.TIME_STEPS, X_seq.shape[2]))

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=Config.PATIENCE,
                restore_best_weights=True
            )

            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop, lr_scheduler]
            )

            models.append(model)
            break  # Sadece ilk fold

    print("✅ Model eğitimi tamamlandı!")
    return models


def ensemble_predict(models, input_data):
    """Ensemble tahmin"""
    predictions = np.array([model.predict(input_data, verbose=0) for model in models])
    return np.mean(predictions, axis=0)


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    """Ana program akışı"""

    # Hedef tarih
    target_date_str = input("Tahmin edilecek tarihi girin (GG.AA.YYYY): ")
    target_date = datetime.strptime(target_date_str, "%d.%m.%Y")

    # Veri yükleme
    print("\n📂 Veriler yükleniyor...")
    weather_data = load_and_convert_data(Config.RAW_DATA_PATH)

    # Veri temizleme
    weather_data = clean_data_pipeline(weather_data)

    # Özellik mühendisliği
    print("🔧 Özellikler oluşturuluyor...")
    weather_data = add_time_features(weather_data)
    weather_data = add_temperature_features(weather_data)
    weather_data = add_cyclical_features(weather_data)

    # Geçmiş analiz
    hist_stats = analyze_historical_data(target_date)

    # Veri filtreleme
    print("🔍 Veriler filtreleniyor...")
    past_data = smart_filter_data(weather_data, target_date)

    if past_data.empty:
        print("❌ Yeterli veri bulunamadı!")
        return

    print(f"✅ {len(past_data)} kayıt bulundu\n")

    # Özellikler ve hedef
    all_features = (
        Config.BASIC_FEATURES +
        ['Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos',
         'DayOfYear_sin', 'DayOfYear_cos', 'TimeOfDay_sin', 'TimeOfDay_cos',
         'Season', 'YearWeight', 'IsDayTime']
    )

    available_features = [f for f in all_features if f in past_data.columns]

    # Scaling
    scaler = RobustScaler()
    basic_features = [f for f in Config.BASIC_FEATURES if f in past_data.columns]
    past_data[basic_features] = scaler.fit_transform(past_data[basic_features])

    X = past_data[available_features].values
    y = past_data['Temperature'].values

    # Sekans oluşturma
    X_seq, y_seq = create_sequences(X, y, Config.TIME_STEPS)

    if len(X_seq) == 0:
        print("❌ Sekans oluşturulamadı!")
        return

    # Model eğitimi
    models = train_ensemble_models(X_seq, y_seq)

    # Hourly bias hesapla
    hourly_bias = calculate_hourly_bias(weather_data, models, available_features, target_date)

    # Tahmin döngüsü
    print("\n🎯 Tahminler yapılıyor...")

    time_intervals = pd.date_range(start="00:20", periods=48, freq="30T").time

    predicted_params = {feature: [] for feature in Config.BASIC_FEATURES}

    for idx, target_time in enumerate(time_intervals):
        target_hour = target_time.hour
        target_minute = target_time.minute

        # Saate uygun verileri filtrele
        filtered_data = past_data[
            (past_data['Hour'] == target_hour) &
            (abs(past_data['Minute'] - target_minute) < 30)
        ]

        if filtered_data.empty:
            # Önceki değeri kullan
            if idx > 0:
                for key in predicted_params:
                    predicted_params[key].append(predicted_params[key][-1])
            else:
                for key in predicted_params:
                    predicted_params[key].append(0)
            continue

        # Akıllı ağırlıklandırma
        weighted_data = smart_weighted_average(
            filtered_data, target_hour, target_minute, target_date
        )

        if weighted_data is None or weighted_data.empty:
            if idx > 0:
                for key in predicted_params:
                    predicted_params[key].append(predicted_params[key][-1])
            else:
                for key in predicted_params:
                    predicted_params[key].append(0)
            continue

        # Ağırlıklı ortalama
        for key in Config.BASIC_FEATURES:
            if key in weighted_data.columns:
                values = weighted_data[key].values
                weights = weighted_data['combined_weight'].values

                valid_mask = ~np.isnan(values) & ~np.isnan(weights)
                values = values[valid_mask]
                weights = weights[valid_mask]

                if len(values) == 0 or weights.sum() == 0:
                    avg_val = predicted_params[key][-1] if idx > 0 else 0.0
                else:
                    avg_val = np.average(values, weights=weights)

                predicted_params[key].append(avg_val)
            else:
                predicted_params[key].append(
                    predicted_params[key][-1] if idx > 0 else 0.0
                )

    # Model tahminleri
    print("🧠 Derin öğrenme modeli çalışıyor...")

    for i in range(len(predicted_params['Temperature'])):
        if i >= Config.TIME_STEPS:
            recent_features = []

            for idx in range(i - Config.TIME_STEPS, i):
                feature_vector = [predicted_params[f][idx] for f in Config.BASIC_FEATURES]

                curr_time = time_intervals[idx]
                curr_hour = curr_time.hour

                # Döngüsel özellikler
                month_sin = np.sin(2 * np.pi * target_date.month / 12)
                month_cos = np.cos(2 * np.pi * target_date.month / 12)
                hour_sin = np.sin(2 * np.pi * curr_hour / 24)
                hour_cos = np.cos(2 * np.pi * curr_hour / 24)
                day_of_year = target_date.timetuple().tm_yday
                day_sin = np.sin(2 * np.pi * day_of_year / 365)
                day_cos = np.cos(2 * np.pi * day_of_year / 365)

                time_of_day = 0
                if 6 <= curr_hour < 12:
                    time_of_day = 1
                elif 12 <= curr_hour < 18:
                    time_of_day = 2
                elif 18 <= curr_hour < 24:
                    time_of_day = 3

                tod_sin = np.sin(2 * np.pi * time_of_day / 3)
                tod_cos = np.cos(2 * np.pi * time_of_day / 3)

                season = get_season(target_date.month)
                year_weight = 0.5
                is_day_time = 1 if 6 <= curr_hour <= 18 else 0

                feature_vector.extend([
                    month_sin, month_cos, hour_sin, hour_cos, day_sin, day_cos,
                    tod_sin, tod_cos, season, year_weight, is_day_time
                ])

                recent_features.append(feature_vector)

            input_seq = np.array([recent_features])
            predicted_temp = ensemble_predict(models, input_seq)[0][0]
            predicted_params['Temperature'][i] = predicted_temp

    # Scaling'i geri çevir
    scaled_values = np.zeros((len(predicted_params['Temperature']), len(Config.BASIC_FEATURES)))

    for i, feature in enumerate(Config.BASIC_FEATURES):
        for j in range(len(predicted_params[feature])):
            if j < scaled_values.shape[0]:
                scaled_values[j, i] = predicted_params[feature][j]

    original_values = scaler.inverse_transform(scaled_values)

    for i, feature in enumerate(Config.BASIC_FEATURES):
        predicted_params[feature] = original_values[:, i]

    # Fahrenheit'ten Celsius'a
    predicted_params['Temperature'] = [
        round(fahrenheit_to_celsius(temp), 1)
        for temp in predicted_params['Temperature']
    ]

    # Önceki günün min/max
    min_prev, max_prev = get_previous_day_temps(target_date)

    if min_prev is not None:
        print(f"📊 Önceki gün: Min {min_prev:.1f}°C, Max {max_prev:.1f}°C")

    # Adaptive post-processing
    print("\n🔄 Adaptive post-processing...")

    processed_temps = adaptive_post_processing(
        np.array(predicted_params['Temperature']),
        time_intervals,
        hourly_bias,
        target_date.month,
        min_prev,
        max_prev,
        hist_stats
    )

    predicted_params['Temperature'] = processed_temps.tolist()

    # Sonuçları yazdır
    print("\n" + "=" * 60)
    print("🌡️  TAHMİN SONUÇLARI")
    print("=" * 60)

    for idx, time in enumerate(time_intervals):
        if idx < len(predicted_params['Temperature']):
            bias_info = f" (bias: {hourly_bias.get(time.hour, 0):+.2f}°C)" if time.hour in hourly_bias else ""
            print(f"{time} → {predicted_params['Temperature'][idx]:.1f}°C{bias_info}")

    print("=" * 60)

    # Karşılaştırma (eğer validation verisi varsa)
    try:
        validation_data = load_and_convert_data(Config.VALIDATION_DATA_PATH)
        actual_temps = validation_data['Temperature'].tolist()
        predicted_temps = predicted_params['Temperature'][:min(len(actual_temps), len(predicted_params['Temperature']))]
        actual_temps = actual_temps[:len(predicted_temps)]

        if len(actual_temps) > 0:
            rmse = np.sqrt(mean_squared_error(actual_temps, predicted_temps))
            mae = mean_absolute_error(actual_temps, predicted_temps)

            print(f"\n📊 PERFORMANS METRİKLERİ")
            print(f"   MAE:  {mae:.2f}°C")
            print(f"   RMSE: {rmse:.2f}°C")

            errors = np.abs(np.array(actual_temps) - np.array(predicted_temps))
            under_2c = sum(errors <= 2.0)

            print(f"   2°C altı tahminler: {under_2c}/{len(errors)} ({100 * under_2c / len(errors):.1f}%)")
            print(f"   Max hata: {max(errors):.2f}°C")
            print(f"   Median hata: {np.median(errors):.2f}°C")

            # Görselleştirme
            plot_results(time_intervals, actual_temps, predicted_temps, target_date_str, mae, rmse)
    except Exception as e:
        print(f"\n⚠️ Validation verisi yüklenemedi: {e}")

    print("\n✅ Tahmin tamamlandı!")


def plot_results(times, actual, predicted, date_str, mae, rmse):
    """Sonuçları görselleştir"""
    times_str = [str(t) for t in times[:len(actual)]]
    errors = np.array(actual) - np.array(predicted)

    plt.figure(figsize=(16, 10))

    # 1. Gerçek vs Tahmin
    plt.subplot(2, 2, 1)
    plt.plot(times_str, actual, label="Gerçek", marker='o', color='blue', linewidth=2)
    plt.plot(times_str, predicted, label="Tahmin", marker='x', color='red', linestyle='--', linewidth=2)
    plt.xticks(rotation=45)
    plt.title(f"{date_str} (MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C)")
    plt.xlabel("Zaman")
    plt.ylabel("Sıcaklık (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(actual, predicted, alpha=0.7)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', linewidth=2)
    plt.xlabel("Gerçek (°C)")
    plt.ylabel("Tahmin (°C)")
    plt.title("Gerçek vs Tahmin")
    plt.grid(True, alpha=0.3)

    # 3. Hata dağılımı
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=15, alpha=0.7, color='green')
    plt.xlabel("Hata (°C)")
    plt.ylabel("Frekans")
    plt.title("Hata Dağılımı")
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)

    # 4. Zamana göre hata
    plt.subplot(2, 2, 4)
    plt.plot(times_str, abs(errors), marker='o', color='purple')
    plt.xlabel("Zaman")
    plt.ylabel("Mutlak Hata (°C)")
    plt.title("Zamana Göre Hata")
    plt.xticks(rotation=45)
    plt.axhline(y=2.0, color='red', linestyle='--', label='2°C Hedef')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# PROGRAM BAŞLAT
# =============================================================================

if __name__ == "__main__":
    main()
