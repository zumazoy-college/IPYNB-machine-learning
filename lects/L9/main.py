import pandas as pd
import random

df = pd.read_csv('Pokemons.csv')
print(f"Загружено покемонов: {len(df)}")


def simple_neural_network(query: str, top_k: int = 5):
    query = query.lower()
    scores = []

    if len(query) < 3 or query in ['покемон', 'выбери', 'кого']:
        for idx, row in df.iterrows():
            score = row['attack'] * 0.2 + row['speed'] * 0.2 + row['hp'] * 0.2 + random.uniform(0, 10)
            scores.append(row['name'], score)
    else:
        for idx, row in df.iterrows():
            score = 0
            if any(word in query for word in ['огонь', 'fire', 'огненный']):
                if row['type1'] == 'fire' or row['type2'] == 'fire':
                    score += 40
            if any(word in query for word in ['вода', 'water', 'водный']):
                if row['type1'] == 'water' or row['type2'] == 'water':
                    score += 40
            if any(word in query for word in ['трава', 'grass', 'травяной']):
                if row['type1'] == 'grass' or row['type2'] == 'grass':
                    score += 40
            if any(word in query for word in ['электричество', 'electric', 'электрический']):
                if row['type1'] == 'electric' or row['type2'] == 'electric':
                    score += 40
            if any(word in query for word in ['псих', 'psyphic', 'психический']):
                if row['type1'] == 'psyphic' or row['type2'] == 'psyphic':
                    score += 40
            if any(word in query for word in ['дракон', 'dragon']):
                if row['type1'] == 'dragon' or row['type2'] == 'dragon':
                    score += 50

            if any(word in query for word in ['сильный', 'strong', 'мощный']):
                score += row['attack'] * 0.4 + row['sp_attack'] * 0.2
            if any(word in query for word in ['быстрый', 'fast', 'скоростной']):
                score += row['speed'] * 0.6
            if any(word in query for word in ['живучий', 'tank', 'выносливый']):
                score += row['hp'] * 0.4 + row['defense'] * 0.2 + row['sp_defense'] * 0.1
            if any(word in query for word in ['умный', 'smart', 'интеллектуальный']):
                score += row['sp_attack'] * 0.5

            if score == 0:
                score = row['attack'] * 0.1 + row['speed'] * 0.1 + row['hp'] * 0.05

            score += random.uniform(0, 5)
            scores.append((row['name'], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


print("\nНейросеть - консольная версия")
print("--------------------")
print("Что я понимаю:")
print("  • Типы: огонь, вода, трава, электричество, псих, дракон")
print("  • Характеристики: сильный, быстрый, живучий, умный")
print("  • Примеры: 'сильный огонь', 'быстрый электрический', 'живучий вода'")
print("  • 'случайный' - любой покемон")
print("  • 'выход' - завершить")
print("--------------------")

while True:
    query = input('\nВаш запрос: ').strip()
    if query.lower() in ['выход', 'exit', 'quit', 'q']:
        print('Завершение программы...')
        break

    if query.lower() in ['случайный', 'рандомный', 'r']:
        print('Случайный покемон')
        random_pokemon = df.sample(1).iloc[0]
        print(f"{random_pokemon['name'].upper()}")
        print(
            f"Тип: {random_pokemon['type1']} {random_pokemon['type2'] if pd.notna(random_pokemon['type2']) else ''}")
        print(
            f"HP: {random_pokemon['hp']} | Атака: {random_pokemon['attack']} | Скорость: {random_pokemon['speed']}")
        print(f"Спец.атака: {random_pokemon['sp_attack']} |  Защита: {random_pokemon['defense']}")
        continue

    results = simple_neural_network(query)

    print('Рекомендации:')
    for i, (name, score) in enumerate(results, 1):
        pokemon = df[df['name'] == name].iloc[0]
        print(f"{i}. {name.upper()}")
        print(
            f"Тип: {pokemon['type1']} {pokemon['type2'] if pd.notna(pokemon['type2']) else ''}")
        print(
            f"HP: {pokemon['hp']} | Атака: {pokemon['attack']} | Скорость: {pokemon['speed']} | Спец.атака: {pokemon['sp_attack']}")
        print(f'Насколько покемон подходит под запрос: {score:.1f}%\n')
