import sqlite3
from datetime import datetime
import pandas as pd


class Database:
    def __init__(self, db_name='reviews.db'):
        self.db_name = db_name
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                product TEXT NOT NULL,
                rating INTEGER NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT,
                confidence REAL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                product TEXT NOT NULL,
                rating INTEGER NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, product)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def add_review(self, user_id, product, rating, text, sentiment=None, confidence=None):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO reviews (user_id, product, rating, text, sentiment, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, product, rating, text, sentiment, confidence))

        cursor.execute('''
            INSERT OR REPLACE INTO user_ratings (user_id, product, rating, date)
            VALUES (?, ?, ?, ?)
        ''', (user_id, product, rating, datetime.now()))

        conn.commit()
        conn.close()

    def get_all_reviews(self):
        conn = self.get_connection()
        df = pd.read_sql_query('SELECT * FROM reviews ORDER BY date DESC', conn)
        conn.close()

        # Конвертация bytes в int для rating
        if len(df) > 0 and df['rating'].dtype == object:
            def safe_int_convert(x):
                if isinstance(x, bytes):
                    return int.from_bytes(x[:1], byteorder='little')
                return int(x)
            df['rating'] = df['rating'].apply(safe_int_convert)

        return df

    def get_reviews_by_product(self, product):
        conn = self.get_connection()
        df = pd.read_sql_query(
            'SELECT * FROM reviews WHERE product = ? ORDER BY date DESC',
            conn,
            params=(product,)
        )
        conn.close()

        if len(df) > 0 and df['rating'].dtype == object:
            def safe_int_convert(x):
                if isinstance(x, bytes):
                    return int.from_bytes(x[:1], byteorder='little')
                return int(x)
            df['rating'] = df['rating'].apply(safe_int_convert)

        return df

    def get_product_stats(self, product):
        conn = self.get_connection()

        df = pd.read_sql_query(
            'SELECT rating, sentiment FROM reviews WHERE product = ?',
            conn,
            params=(product,)
        )
        conn.close()

        if len(df) == 0:
            return None

        if df['rating'].dtype == object:
            def safe_int_convert(x):
                if isinstance(x, bytes):
                    return int.from_bytes(x[:1], byteorder='little')
                return int(x)
            df['rating'] = df['rating'].apply(safe_int_convert)

        total = len(df)
        avg_rating = df['rating'].mean()
        pos = (df['sentiment'] == 'позитивный').sum()
        neg = (df['sentiment'] == 'негативный').sum()

        return {
            'total_reviews': total,
            'avg_rating': avg_rating,
            'positive_count': pos,
            'negative_count': neg,
            'positive_ratio': pos / total if total > 0 else 0
        }

    def get_all_products(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT product FROM reviews')
        products = [row[0] for row in cursor.fetchall()]
        conn.close()
        return products

    def register_user(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, first_seen, last_seen)
            VALUES (
                ?,
                COALESCE((SELECT first_seen FROM users WHERE user_id = ?), CURRENT_TIMESTAMP),
                CURRENT_TIMESTAMP
            )
        ''', (user_id, user_id))

        conn.commit()
        conn.close()

    def get_user_ratings(self, user_id):
        conn = self.get_connection()
        df = pd.read_sql_query(
            'SELECT product, rating FROM user_ratings WHERE user_id = ?',
            conn,
            params=(user_id,)
        )
        conn.close()

        if len(df) > 0 and df['rating'].dtype == object:
            def safe_int_convert(x):
                if isinstance(x, bytes):
                    return int.from_bytes(x[:1], byteorder='little')
                return int(x)
            df['rating'] = df['rating'].apply(safe_int_convert)

        return df

    def get_all_user_ratings(self):
        conn = self.get_connection()
        df = pd.read_sql_query('SELECT user_id, product, rating FROM user_ratings', conn)
        conn.close()

        if len(df) > 0 and df['rating'].dtype == object:
            def safe_int_convert(x):
                if isinstance(x, bytes):
                    return int.from_bytes(x[:1], byteorder='little')
                return int(x)
            df['rating'] = df['rating'].apply(safe_int_convert)

        return df

    def get_user_count(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM users')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_review_count(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM reviews')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_reviews_by_date(self):
        conn = self.get_connection()
        df = pd.read_sql_query('''
            SELECT DATE(date) as date, COUNT(*) as count
            FROM reviews
            GROUP BY DATE(date)
            ORDER BY date
        ''', conn)
        conn.close()
        return df

    def clear_all_data(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM reviews')
        cursor.execute('DELETE FROM user_ratings')
        cursor.execute('DELETE FROM users')
        conn.commit()
        conn.close()
