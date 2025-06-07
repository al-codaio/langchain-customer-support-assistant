import json

def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Knowledge base file not found at {file_path}")
        return {}

def search_knowledge_base(query, kb_data):
    query_lower = query.lower()
    relevant_articles = []
    for category, articles in kb_data.items():
        for article_title, article_content in articles.items():
            if query_lower in article_title.lower() or query_lower in article_content.lower():
                relevant_articles.append({
                    "category": category,
                    "title": article_title,
                    "content": article_content
                })
    return relevant_articles