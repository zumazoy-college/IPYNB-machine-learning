"""
Модуль машинного обучения с использованием sklearn
Включает: линейную регрессию, кластеризацию, метрики качества
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
import json
import os
from datetime import datetime


def prepare_features_for_ml(df):
    """Подготовка признаков для машинного обучения"""
    from main import extract_rating, extract_price, extract_year

    # Создаем числовые признаки
    features = pd.DataFrame()

    # Извлекаем числовые характеристики
    features['rating'] = df['all_reviews'].apply(extract_rating)
    features['price'] = df['original_price'].apply(extract_price)
    features['year'] = df['release_date'].apply(extract_year)

    # Количество жанров
    features['genre_count'] = df['genre'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

    # Количество тегов
    features['tags_count'] = df['popular_tags'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

    # Бинарные признаки для популярных жанров
    for genre in ['Action', 'Strategy', 'RPG', 'Indie', 'Adventure']:
        features[f'is_{genre.lower()}'] = df['genre'].apply(
            lambda x: 1 if genre.lower() in str(x).lower() else 0
        )

    # Удаляем строки с NaN
    features = features.dropna()

    return features


def train_price_prediction_model(df):
    """
    Обучение модели линейной регрессии для предсказания популярности игры
    на основе её характеристик
    """
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ ЛИНЕЙНОЙ РЕГРЕССИИ")
    print("="*60)

    # Подготовка данных
    features = prepare_features_for_ml(df)

    if len(features) < 100:
        print("Недостаточно данных для обучения модели")
        return None

    # Целевая переменная - рейтинг игры
    X = features.drop('rating', axis=1)
    y = features['rating']

    print(f"\nРазмер датасета: {len(features)} игр")
    print(f"Количество признаков: {X.shape[1]}")

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Обучающая выборка: {len(X_train)} игр")
    print(f"Тестовая выборка: {len(X_test)} игр")

    # Обучение модели
    print("\nОбучение модели...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)

    # Метрики качества
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "-"*60)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print("-"*60)
    print(f"MAE (средняя абсолютная ошибка): {mae:.2f}%")
    print(f"RMSE (корень из среднеквадратичной ошибки): {rmse:.2f}%")
    print(f"R² (коэффициент детерминации): {r2:.4f}")
    print("-"*60)

    # Интерпретация R²
    if r2 >= 0.7:
        quality = "Отличное"
    elif r2 >= 0.5:
        quality = "Хорошее"
    elif r2 >= 0.3:
        quality = "Удовлетворительное"
    else:
        quality = "Плохое"

    print(f"\nКачество модели: {quality}")
    print(f"Модель объясняет {r2*100:.1f}% вариации рейтинга игр")

    # Важность признаков
    print("\nВажность признаков (коэффициенты):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")

    # Сохранение модели
    model_info = {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(features),
        'features': X.columns.tolist(),
        'metrics': {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2)
        },
        'feature_importance': feature_importance.to_dict('records')
    }

    with open('ml_regression_model.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print("\nМодель сохранена в: ml_regression_model.json")
    print("="*60)

    return model_info


def perform_clustering_analysis(df):
    """
    Применяет K-Means, Agglomerative Clustering, DBSCAN
    """
    print("\n" + "="*60)
    print("КЛАСТЕРНЫЙ АНАЛИЗ ИГР")
    print("="*60)

    # Подготовка данных
    features = prepare_features_for_ml(df)

    if len(features) < 100:
        print("Недостаточно данных для кластеризации")
        return None

    print(f"\nРазмер датасета: {len(features)} игр")
    print(f"Количество признаков: {features.shape[1]}")

    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    print("\nДанные масштабированы (StandardScaler)")

    # 1. K-Means кластеризация
    print("\n" + "-"*60)
    print("1. K-MEANS КЛАСТЕРИЗАЦИЯ")
    print("-"*60)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
    db_kmeans = davies_bouldin_score(X_scaled, labels_kmeans)
    ch_kmeans = calinski_harabasz_score(X_scaled, labels_kmeans)

    print(f"Silhouette Score: {sil_kmeans:.4f} (чем ближе к 1, тем лучше)")
    print(f"Davies-Bouldin Score: {db_kmeans:.4f} (чем меньше, тем лучше)")
    print(f"Calinski-Harabasz Score: {ch_kmeans:.2f} (чем больше, тем лучше)")

    # Распределение по кластерам
    unique, counts = np.unique(labels_kmeans, return_counts=True)
    print("\nРаспределение игр по кластерам:")
    for cluster, count in zip(unique, counts):
        print(f"  Кластер {cluster}: {count} игр ({count/len(labels_kmeans)*100:.1f}%)")

    # 2. Agglomerative Clustering
    print("\n" + "-"*60)
    print("2. ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ")
    print("-"*60)

    agglo = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels_agglo = agglo.fit_predict(X_scaled)

    sil_agglo = silhouette_score(X_scaled, labels_agglo)
    db_agglo = davies_bouldin_score(X_scaled, labels_agglo)
    ch_agglo = calinski_harabasz_score(X_scaled, labels_agglo)

    print(f"Silhouette Score: {sil_agglo:.4f}")
    print(f"Davies-Bouldin Score: {db_agglo:.4f}")
    print(f"Calinski-Harabasz Score: {ch_agglo:.2f}")

    # 3. DBSCAN
    print("\n" + "-"*60)
    print("3. DBSCAN (ПЛОТНОСТНАЯ КЛАСТЕРИЗАЦИЯ)")
    print("-"*60)

    dbscan = DBSCAN(eps=0.7, min_samples=10)
    labels_dbscan = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)

    print(f"Найдено кластеров: {n_clusters}")
    print(f"Точек шума: {n_noise} ({n_noise/len(labels_dbscan)*100:.1f}%)")

    if n_clusters > 1:
        mask = labels_dbscan != -1
        sil_dbscan = silhouette_score(X_scaled[mask], labels_dbscan[mask])
        db_dbscan = davies_bouldin_score(X_scaled[mask], labels_dbscan[mask])
        ch_dbscan = calinski_harabasz_score(X_scaled[mask], labels_dbscan[mask])

        print(f"Silhouette Score: {sil_dbscan:.4f}")
        print(f"Davies-Bouldin Score: {db_dbscan:.4f}")
        print(f"Calinski-Harabasz Score: {ch_dbscan:.2f}")

    # 4. PCA + K-Means
    print("\n" + "-"*60)
    print("4. PCA + K-MEANS")
    print("-"*60)

    # Снижение размерности
    pca = PCA(n_components=0.85, random_state=42)  # 85% дисперсии
    X_pca = pca.fit_transform(X_scaled)

    print(f"Исходная размерность: {X_scaled.shape[1]}")
    print(f"После PCA: {X_pca.shape[1]} компонент")
    print(f"Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.2%}")

    # K-Means на PCA данных
    kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_kmeans_pca = kmeans_pca.fit_predict(X_pca)

    sil_kmeans_pca = silhouette_score(X_pca, labels_kmeans_pca)
    db_kmeans_pca = davies_bouldin_score(X_pca, labels_kmeans_pca)
    ch_kmeans_pca = calinski_harabasz_score(X_pca, labels_kmeans_pca)

    print(f"\nSilhouette Score: {sil_kmeans_pca:.4f}")
    print(f"Davies-Bouldin Score: {db_kmeans_pca:.4f}")
    print(f"Calinski-Harabasz Score: {ch_kmeans_pca:.2f}")

    # Сравнение результатов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МЕТОДОВ КЛАСТЕРИЗАЦИИ")
    print("="*60)

    results = {
        'K-Means': {'sil': sil_kmeans, 'db': db_kmeans, 'ch': ch_kmeans},
        'Agglomerative': {'sil': sil_agglo, 'db': db_agglo, 'ch': ch_agglo},
        'K-Means + PCA': {'sil': sil_kmeans_pca, 'db': db_kmeans_pca, 'ch': ch_kmeans_pca}
    }

    print(f"\n{'Метод':<20} {'Silhouette':<12} {'Davies-Bouldin':<18} {'Calinski-Harabasz':<20}")
    print("-"*70)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['sil']:<12.4f} {metrics['db']:<18.4f} {metrics['ch']:<20.2f}")

    # Определение лучшего метода
    best_method = max(results.items(), key=lambda x: x[1]['sil'])
    print(f"\nЛучший метод: {best_method[0]}")
    print(f"(по Silhouette Score: {best_method[1]['sil']:.4f})")

    # Сохранение результатов
    clustering_info = {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(features),
        'methods': {
            method: {
                'silhouette': float(metrics['sil']),
                'davies_bouldin': float(metrics['db']),
                'calinski_harabasz': float(metrics['ch'])
            }
            for method, metrics in results.items()
        },
        'best_method': best_method[0],
        'pca_components': int(X_pca.shape[1]),
        'pca_variance_explained': float(pca.explained_variance_ratio_.sum())
    }

    with open('ml_clustering_results.json', 'w', encoding='utf-8') as f:
        json.dump(clustering_info, f, ensure_ascii=False, indent=2)

    print("\nРезультаты сохранены в: ml_clustering_results.json")
    print("="*60)

    return clustering_info


if __name__ == '__main__':
    print("Модуль sklearn ML готов к использованию")
