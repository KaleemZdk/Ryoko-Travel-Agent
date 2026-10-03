from tavily import TavilyClient
import os 
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")

)

def tavily_search(query):
    response = client.search(
        query=query,
        max_results=5
    )
    return response

def format_search_results(response):
    results = []

    for i, r in enumerate(response.get("results", []), 1):
        # Safely extract values
        title = r.get("title") or "Unknown"
        url = r.get("url") or ""
        snippet = r.get("content") or ""

        # Make sure values are strings
        title = str(title).strip()
        url = str(url).strip()
        snippet = str(snippet).strip()

        # Keep only the first 300 characters
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(
            f"{i}. **{title}**\n"
            f"   {url}\n"
            f"   {snippet}"
        )

    return "\n\n".join(results)


