from rag_text2sql.models import cloud, gemini

PROVIDERS = {
    "anthropic": (cloud.generate_sql, cloud.DEFAULT_MODEL),
    "gemini": (gemini.generate_sql, gemini.DEFAULT_MODEL),
}

__all__ = ["PROVIDERS"]
