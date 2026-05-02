import requests
import json
import os
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urlparse
"""сбор разнородных HTML‑страниц (ACM, arXiv, IEEE, Springer, университетские репозитории и пр.) 
+ rule-based извлечение базовых метаданных + оценка «сложности» страницы."""

class ScientificDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.dataset = []
        self.raw_data_dir = "data/raw"

        # Создаем директории
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def collect_diverse_dataset(self):
        """Собираем разнообразный датасет из РАЗНЫХ источников"""
        print("🎯 Начинаем сбор разнообразного датасета для ML...")

        # РЕАЛЬНЫЕ и РАЗНООБРАЗНЫЕ URL с разной структурой
        url_groups = {
            'arxiv_simple': [
                "https://arxiv.org/abs/2301.00001",  # Простая структура arXiv
                "https://arxiv.org/abs/2302.00123",
                "https://arxiv.org/abs/2303.00456"
            ],
            'acm_blog': [  # ACM блоги и статьи без стандартных кнопок
                "https://cacm.acm.org/research/toward-verified-artificial-intelligence/",  # Твой пример!
                "https://cacm.acm.org/magazines/2023/12/271346-ai-and-the-transformation-of-society/",
                "https://cacm.acm.org/blogs/blog-cacm/267249-ai-ethics-and-society/fulltext"
            ],
            'ieee_conference': [  # Конференции IEEE
                "https://ieeexplore.ieee.org/document/10288894",  # Real paper
                "https://ieeexplore.ieee.org/document/10289201",  # Real paper
                "https://ieeexplore.ieee.org/document/10284563"  # Real paper
            ],
            'springer_journal': [  # Журналы Springer
                "https://link.springer.com/article/10.1007/s00453-023-01142-y",  # Real
                "https://link.springer.com/article/10.1007/s10957-023-02233-0",  # Real
                "https://link.springer.com/article/10.1007/s10878-023-01058-z"  # Real
            ],
            'university_repo': [  # Университетские репозитории
                "https://dspace.mit.edu/handle/1721.1/151312",  # MIT repository
                "https://researchrepository.wvu.edu/etd/11345/",  # WVU repository
                "https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1289&context=disstheses"  # UMass
            ],
            'personal_sites': [  # Персональные сайты исследователей
                "http://www.cs.cornell.edu/home/kleinber/",  # Personal page
                "https://ai.stanford.edu/~cbfrench/",  # Research page
                "http://www.cs.cmu.edu/~epxing/"  # Faculty page
            ]
        }

        total_collected = 0

        for source_type, urls in url_groups.items():
            print(f"\n📁 Собираем данные с {source_type.upper()}...")

            for i, url in enumerate(urls):
                try:
                    print(f"  📄 Обрабатываем {i + 1}/{len(urls)}: {url}")

                    # Загружаем страницу
                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()

                    # Проверяем, что страница существует и доступна
                    if response.status_code != 200:
                        print(f"  ⚠️  Страница недоступна: {response.status_code}")
                        continue

                    # Парсим базовую информацию
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Проверяем, что страница содержит контент
                    if len(soup.get_text().strip()) < 100:
                        print(f"  ⚠️  Страница слишком короткая, пропускаем")
                        continue

                    # Извлекаем базовые метаданные
                    basic_metadata = self._extract_basic_metadata(soup, url)

                    # Анализируем сложность страницы для ML
                    complexity = self._analyze_page_complexity(soup, url)

                    # Сохраняем данные
                    page_data = {
                        'id': f"{source_type}_{i + 1}",
                        'url': url,
                        'source_type': source_type,
                        'html_content': response.text,
                        'basic_metadata': basic_metadata,
                        'page_complexity': complexity,
                        'has_cite_button': self._has_cite_button(soup),
                        'has_structured_metadata': self._has_structured_metadata(soup),
                        'collection_timestamp': time.time(),
                        'status': 'collected'
                    }

                    self.dataset.append(page_data)
                    total_collected += 1

                    # Сохраняем в отдельный файл
                    filename = f"{source_type}_{i + 1}.json"
                    filepath = os.path.join(self.raw_data_dir, filename)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(page_data, f, indent=2, ensure_ascii=False)

                    print(f"  ✅ Успешно сохранено: {filename}")
                    print(f"  📊 Сложность: {complexity['level']}, Cite button: {page_data['has_cite_button']}")

                    # Пауза между запросами
                    time.sleep(random.uniform(2, 4))

                except Exception as e:
                    print(f"  ❌ Ошибка с {url}: {e}")
                    continue

        # Сохраняем общий датасет
        self._save_dataset_stats()

        print(f"\n🎉 Сбор данных завершен!")
        print(f"📊 Всего собрано: {total_collected} страниц")
        print(f"📁 Данные сохранены в: {self.raw_data_dir}")

        return self.dataset

    def _analyze_page_complexity(self, soup, url):
        """Анализируем сложность страницы для ML"""
        text_blocks = self._extract_text_blocks(soup)
        metadata_elements = self._count_metadata_elements(soup)

        complexity_score = 0
        complexity_level = "low"

        # Оцениваем сложность по различным факторам
        if len(text_blocks) > 50:
            complexity_score += 2
        elif len(text_blocks) > 20:
            complexity_score += 1

        if metadata_elements < 3:  # Мало структурированных метаданных
            complexity_score += 2

        if not self._has_structured_metadata(soup):
            complexity_score += 1

        if 'acm.org/blogs' in url or 'personal' in url.lower():
            complexity_score += 1  # Блоги и персональные страницы сложнее

        # Определяем уровень сложности
        if complexity_score >= 3:
            complexity_level = "high"
        elif complexity_score >= 1:
            complexity_level = "medium"
        else:
            complexity_level = "low"

        return {
            'score': complexity_score,
            'level': complexity_level,
            'text_blocks_count': len(text_blocks),
            'metadata_elements_count': metadata_elements
        }

    def _extract_text_blocks(self, soup):
        """Извлекаем текстовые блоки для анализа"""
        blocks = []
        elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'div', 'span', 'li'])

        for elem in elements:
            text = elem.get_text().strip()
            if text and len(text) > 10:
                blocks.append({
                    'text': text,
                    'tag': elem.name,
                    'class': elem.get('class', [])
                })

        return blocks

    def _count_metadata_elements(self, soup):
        """Считаем количество структурированных мета-элементов"""
        count = 0
        meta_selectors = [
            'meta[name="citation_title"]',
            'meta[name="citation_author"]',
            'meta[name="citation_doi"]',
            'meta[name="citation_journal_title"]',
            'meta[property="og:title"]',
            '[class*="title"]',
            '[class*="author"]'
        ]

        for selector in meta_selectors:
            count += len(soup.select(selector))

        return count

    def _has_cite_button(self, soup):
        """Проверяем, есть ли кнопка Cite"""
        cite_indicators = [
            'cite', 'citation', 'bibtex', 'export', 'download citation'
        ]

        text = soup.get_text().lower()
        return any(indicator in text for indicator in cite_indicators)

    def _has_structured_metadata(self, soup):
        """Проверяем, есть ли структурированные метаданные"""
        structured_meta = [
            'meta[name="citation_title"]',
            'meta[name="citation_author"]',
            'meta[name="citation_doi"]'
        ]

        for selector in structured_meta:
            if soup.select(selector):
                return True
        return False

    def _extract_basic_metadata(self, soup, url):
        """Извлекаем базовые метаданные rule-based методом"""
        metadata = {
            'title': self._extract_title(soup),
            'authors': self._extract_authors(soup),
            'year': self._extract_year(soup),
            'doi': self._extract_doi(soup),
            'journal': self._extract_journal(soup, url),
            'extraction_method': 'rule_based'
        }
        return metadata

    def _extract_title(self, soup):
        """Извлекаем заголовок"""
        # Мета-теги
        title_meta = soup.find('meta', {'name': 'citation_title'})
        if title_meta:
            return title_meta.get('content')

        title_og = soup.find('meta', {'property': 'og:title'})
        if title_og:
            return title_og.get('content')

        # HTML элементы
        selectors = ['h1', '.title', '[class*="title"]', 'head title']
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text().strip()
                if text and len(text) > 10:
                    return text
        return None

    def _extract_authors(self, soup):
        """Извлекаем авторов"""
        authors = []

        # Из мета-тегов
        author_metas = soup.find_all('meta', {'name': 'citation_author'})
        if author_metas:
            authors = [author.get('content') for author in author_metas if author.get('content')]

        return authors if authors else None

    def _extract_year(self, soup):
        """Извлекаем год"""
        year_meta = soup.find('meta', {'name': 'citation_publication_date'})
        if year_meta:
            date_text = year_meta.get('content', '')
            import re
            year_match = re.search(r'20\d{2}', date_text)
            if year_match:
                return year_match.group()
        return None

    def _extract_doi(self, soup):
        """Извлекаем DOI"""
        doi_meta = soup.find('meta', {'name': 'citation_doi'})
        if doi_meta:
            return doi_meta.get('content')
        return None

    def _extract_journal(self, soup, url):
        """Извлекаем журнал"""
        journal_meta = soup.find('meta', {'name': 'citation_journal_title'})
        if journal_meta:
            return journal_meta.get('content')

        # Определяем по домену и контексту
        domain = urlparse(url).netloc
        if 'arxiv' in domain:
            return 'arXiv'
        elif 'ieee' in domain:
            return 'IEEE'
        elif 'springer' in domain:
            return 'Springer'
        elif 'acm' in domain:
            return 'ACM'
        elif 'mit.edu' in domain:
            return 'MIT Repository'
        elif 'wvu.edu' in domain:
            return 'WVU Repository'

        return 'Unknown'

    def _save_dataset_stats(self):
        """Сохраняем статистику датасета"""
        stats = {
            'total_pages': len(self.dataset),
            'sources': {},
            'complexity_distribution': {'low': 0, 'medium': 0, 'high': 0},
            'cite_button_stats': {'with_cite': 0, 'without_cite': 0},
            'structured_metadata_stats': {'with_metadata': 0, 'without_metadata': 0},
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        for item in self.dataset:
            source = item['source_type']
            if source not in stats['sources']:
                stats['sources'][source] = 0
            stats['sources'][source] += 1

            # Сложность
            complexity = item['page_complexity']['level']
            stats['complexity_distribution'][complexity] += 1

            # Кнопка Cite
            if item['has_cite_button']:
                stats['cite_button_stats']['with_cite'] += 1
            else:
                stats['cite_button_stats']['without_cite'] += 1

            # Структурированные метаданные
            if item['has_structured_metadata']:
                stats['structured_metadata_stats']['with_metadata'] += 1
            else:
                stats['structured_metadata_stats']['without_metadata'] += 1

        stats_path = os.path.join(self.raw_data_dir, 'dataset_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"📈 Статистика сохранена: {stats_path}")

        # Выводим краткую статистику
        print(f"\n📊 КРАТКАЯ СТАТИСТИКА ДАТАСЕТА:")
        print(f"   Всего страниц: {stats['total_pages']}")
        print(f"   Сложность: Low={stats['complexity_distribution']['low']}, "
              f"Medium={stats['complexity_distribution']['medium']}, "
              f"High={stats['complexity_distribution']['high']}")
        print(f"   С кнопкой Cite: {stats['cite_button_stats']['with_cite']}")
        print(f"   Со структурированными метаданными: {stats['structured_metadata_stats']['with_metadata']}")


# Тестовый запуск
if __name__ == "__main__":
    collector = ScientificDataCollector()
    dataset = collector.collect_diverse_dataset()
