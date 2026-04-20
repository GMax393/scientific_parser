import json
import os
from bs4 import BeautifulSoup
import re
from collections import Counter

"""разметка сырых данных в блоки текста с простыми признаками и эвристической разметкой 
title/author/year/journal/doi/... для обучения ML‑модели."""
class DataAnnotator:
    def __init__(self):
        self.raw_data_dir = "data/raw"
        self.annotated_data_dir = "data/annotated"
        os.makedirs(self.annotated_data_dir, exist_ok=True)

    def analyze_collected_data(self):
        """Анализируем собранные данные и создаем разметку"""
        print("🔍 Анализируем собранные данные...")

        annotated_dataset = []

        # Читаем все собранные файлы
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith('.json') and filename != 'dataset_stats.json':
                filepath = os.path.join(self.raw_data_dir, filename)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Анализируем страницу и создаем разметку для ML
                    annotated_data = self._create_annotation(data)
                    annotated_dataset.append(annotated_data)

                    # Сохраняем размеченные данные
                    annotated_path = os.path.join(self.annotated_data_dir, filename)
                    with open(annotated_path, 'w', encoding='utf-8') as f:
                        json.dump(annotated_data, f, indent=2, ensure_ascii=False)

                    print(f"✅ Размечено: {filename}")

                except Exception as e:
                    print(f"❌ Ошибка при обработке {filename}: {e}")
                    continue

        # Сохраняем общий датасет
        with open('data/annotated_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)

        print(f"✅ Размечено {len(annotated_dataset)} страниц")

        # Анализируем распределение классов
        self._analyze_class_distribution(annotated_dataset)

        return annotated_dataset

    def _create_annotation(self, data):
        """Создаем разметку для ML обучения"""
        soup = BeautifulSoup(data['html_content'], 'html.parser')

        # Извлекаем все текстовые блоки
        text_blocks = self._extract_text_blocks(soup)

        # Анализируем сложность страницы
        complexity = self._analyze_page_complexity(text_blocks, data)

        # Размечаем блоки (авторы, заголовок, журнал, год, другие)
        labeled_blocks = self._label_blocks(text_blocks, data.get('basic_metadata', {}))

        annotation = {
            'id': data.get('id', 'unknown'),
            'url': data.get('url', ''),
            'source_type': data.get('source_type', 'unknown'),
            'complexity': complexity,
            'text_blocks': labeled_blocks,
            'ground_truth': data.get('basic_metadata', {}),
            'has_cite_button': self._has_cite_button(soup),
            'has_structured_metadata': self._has_structured_metadata(soup),
            'total_blocks': len(labeled_blocks),
            'labeled_blocks': len([b for b in labeled_blocks if b['label'] != 'other'])
        }

        return annotation

    def _analyze_page_complexity(self, text_blocks, data):
        """Анализируем сложность страницы"""
        complexity_score = 0

        # Оцениваем по количеству блоков
        if len(text_blocks) > 50:
            complexity_score += 2
        elif len(text_blocks) > 20:
            complexity_score += 1

        # Оцениваем по наличию структурированных данных
        metadata = data.get('basic_metadata', {})
        if not metadata.get('title') or not metadata.get('authors'):
            complexity_score += 1

        # Оцениваем по типу источника
        source_type = data.get('source_type', '')
        if 'blog' in source_type.lower() or 'personal' in source_type.lower():
            complexity_score += 1

        # Определяем уровень сложности
        if complexity_score >= 3:
            level = "high"
        elif complexity_score >= 1:
            level = "medium"
        else:
            level = "low"

        return {
            'score': complexity_score,
            'level': level,
            'text_blocks_count': len(text_blocks)
        }

    def _has_cite_button(self, soup):
        """Проверяем, есть ли кнопка Cite"""
        text = soup.get_text().lower()
        cite_indicators = ['cite', 'citation', 'bibtex', 'export']
        return any(indicator in text for indicator in cite_indicators)

    def _has_structured_metadata(self, soup):
        """Проверяем, есть ли структурированные метаданные"""
        structured_selectors = [
            'meta[name="citation_title"]',
            'meta[name="citation_author"]',
            'meta[name="citation_doi"]'
        ]
        for selector in structured_selectors:
            if soup.select(selector):
                return True
        return False

    def _extract_text_blocks(self, soup):
        """Извлекаем текстовые блоки для разметки"""
        blocks = []

        # Ищем все потенциально релевантные элементы
        elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'div', 'span', 'li', 'td'])

        for elem in elements:
            text = elem.get_text().strip()
            # Фильтруем слишком короткие тексты и мусор
            if text and len(text) > 3 and not self._is_noise(text):
                dom_depth = self._get_dom_depth(elem)
                sibling_index = self._get_sibling_index(elem)
                link_density = self._get_link_density(elem)
                blocks.append({
                    'text': text,
                    'tag': elem.name,
                    'classes': elem.get('class', []),
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'is_visible': self._is_visible_element(elem),
                    'dom_depth': dom_depth,
                    'sibling_index': sibling_index,
                    'link_density': link_density,
                })

        return blocks

    def _get_dom_depth(self, elem):
        """Глубина элемента в DOM."""
        depth = 0
        p = elem.parent
        while p is not None and getattr(p, "name", None):
            depth += 1
            p = p.parent
        return depth

    def _get_sibling_index(self, elem):
        """Позиция элемента среди однотипных соседей."""
        parent = elem.parent
        if parent is None:
            return 0
        siblings = parent.find_all(elem.name, recursive=False)
        for idx, sibling in enumerate(siblings):
            if sibling is elem:
                return idx
        return 0

    def _get_link_density(self, elem):
        """Доля текста ссылок внутри блока."""
        total = len(elem.get_text(" ", strip=True))
        if total <= 0:
            return 0.0
        link_text = " ".join(a.get_text(" ", strip=True) for a in elem.find_all("a"))
        return min(1.0, len(link_text) / max(total, 1))

    def _is_noise(self, text):
        """Проверяем, является ли текст шумом"""
        noise_patterns = [
            r'^\d+$',  # Только цифры
            r'^[\.\,\-\+\=\*]+$',  # Только символы
            r'^[A-Z]{2,}$',  # Только заглавные буквы (акронимы)
        ]

        for pattern in noise_patterns:
            if re.match(pattern, text.strip()):
                return True

        return False

    def _is_visible_element(self, elem):
        """Проверяем, является ли элемент видимым"""
        # Простая проверка стилей (можно расширить)
        style = elem.get('style', '').lower()
        if 'display:none' in style or 'visibility:hidden' in style:
            return False
        return True

    def _label_blocks(self, blocks, ground_truth):
        """Размечаем блоки на основе ground truth данных"""
        labeled_blocks = []

        for block in blocks:
            label = self._assign_label(block, ground_truth)
            block['label'] = label
            labeled_blocks.append(block)

        return labeled_blocks

    def _assign_label(self, block, ground_truth):
        """Присваиваем label текстовому блоку"""
        text = block['text'].lower()

        # Проверяем заголовок (самый важный)
        if ground_truth.get('title'):
            title_lower = ground_truth['title'].lower()
            # Ищем точное совпадение или значительное пересечение
            if title_lower in text or self._significant_overlap(title_lower, text):
                return 'title'

        # Проверяем авторов
        if ground_truth.get('authors'):
            for author in ground_truth['authors']:
                if author and author.lower() in text:
                    return 'author'

        # Проверяем год
        if ground_truth.get('year') and ground_truth['year'] in text:
            return 'year'

        # Проверяем журнал
        if ground_truth.get('journal') and ground_truth['journal'].lower() in text:
            return 'journal'

        # Проверяем DOI
        if ground_truth.get('doi') and ground_truth['doi'].lower() in text:
            return 'doi'

        # Эвристики для ненайденных случаев
        if any(word in text for word in ['author', 'by ', 'et al', 'written by']):
            return 'author_candidate'
        elif any(word in text for word in ['volume', 'vol.', 'issue', 'no.', 'journal']):
            return 'journal_info'
        elif any(word in text for word in ['abstract', 'summary']):
            return 'abstract'
        elif re.search(r'20\d{2}', text) or re.search(r'19\d{2}', text):
            return 'year_candidate'
        elif 'doi:' in text or 'digital object identifier' in text:
            return 'doi_candidate'
        elif len(text) > 50 and len(text.split()) > 8:
            return 'content'  # Основной контент статьи
        else:
            return 'other'

    def _significant_overlap(self, str1, str2):
        """Проверяем значительное пересечение строк"""
        words1 = set(str1.split())
        words2 = set(str2.split())
        common_words = words1.intersection(words2)
        return len(common_words) >= 3  # Минимум 3 общих слова

    def _analyze_class_distribution(self, annotated_dataset):
        """Анализируем распределение классов в датасете"""
        print("\n📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ:")

        all_labels = []
        for page in annotated_dataset:
            for block in page['text_blocks']:
                all_labels.append(block['label'])

        label_counts = Counter(all_labels)

        total_blocks = len(all_labels)
        print(f"Всего текстовых блоков: {total_blocks}")
        print(f"Всего страниц: {len(annotated_dataset)}")
        print(f"Среднее блоков на страницу: {total_blocks / len(annotated_dataset):.1f}")

        print("\nРаспределение меток:")
        for label, count in label_counts.most_common():
            percentage = (count / total_blocks) * 100
            print(f"  {label:15}: {count:3} блоков ({percentage:5.1f}%)")

        # Анализ сложности страниц
        complexity_levels = [page['complexity']['level'] for page in annotated_dataset]
        complexity_counts = Counter(complexity_levels)
        print(f"\nСложность страниц:")
        for level, count in complexity_counts.most_common():
            print(f"  {level}: {count} страниц")

        # Сохраняем статистику
        stats = {
            'total_pages': len(annotated_dataset),
            'total_blocks': total_blocks,
            'average_blocks_per_page': total_blocks / len(annotated_dataset),
            'label_distribution': dict(label_counts),
            'complexity_distribution': dict(complexity_counts),
            'pages_with_cite_button': sum(1 for p in annotated_dataset if p['has_cite_button']),
            'pages_with_structured_metadata': sum(1 for p in annotated_dataset if p['has_structured_metadata'])
        }

        with open('data/annotation_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\n📈 Статистика сохранена в data/annotation_stats.json")


# Запуск разметки
if __name__ == "__main__":
    annotator = DataAnnotator()
    annotated_data = annotator.analyze_collected_data()

    print("\n🎉 Разметка данных завершена!")
    print("📁 Данные сохранены в:")
    print("   - data/annotated/ (отдельные файлы)")
    print("   - data/annotated_dataset.json (общий датасет)")
    print("   - data/annotation_stats.json (статистика)")
