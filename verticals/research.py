"""Research for the anti-hallucination gate: DuckDuckGo snippets + Wikipedia extracts.

Every line carries a "[source: ...]" tag so a count_up beat can cite where its
figure came from. The draft prompt only allows figures that appear here.
"""

import re

import requests
from html.parser import HTMLParser

from .config import extract_keywords
from .log import log
from .retry import with_retry


@with_retry(max_retries=2, base_delay=2.0)
def _fetch_ddg(keywords: str) -> str:
    """Fetch search snippets from DuckDuckGo HTML endpoint."""
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    r = requests.post(url, data={"q": keywords}, headers=headers, timeout=10)
    r.raise_for_status()
    return r.text


def research_topic(news: str) -> str:
    """DuckDuckGo search -> extract facts for anti-hallucination gate."""
    log("Researching topic via DuckDuckGo...")
    keywords = extract_keywords(news)

    try:
        html = _fetch_ddg(keywords)

        snippets = []
        domains = []

        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self._in = False
                self._text = []

            def handle_starttag(self, tag, attrs):
                d = dict(attrs)
                if tag == "a" and "result__url" in d.get("class", ""):
                    self._url = True
                    self._text = []
                if tag == "a" and "result__snippet" in d.get("class", ""):
                    self._in = True
                    self._text = []

            def handle_endtag(self, tag):
                if getattr(self, "_url", False) and tag == "a":
                    domains.append("".join(self._text).strip().split("/")[0])
                    self._url = False
                if self._in and tag == "a":
                    snippets.append("".join(self._text).strip())
                    self._in = False

            def handle_data(self, data):
                if self._in or getattr(self, "_url", False):
                    self._text.append(data)

        p = Parser()
        p.feed(html)
        # Sanitize snippets: truncate each to limit prompt injection surface
        snippets = [s[:300] for s in snippets]
        tagged = [(f"[source: {domains[i]}] " if i < len(domains) and domains[i] else "") + s
                  for i, s in enumerate(snippets[:8])]
        research = "\n".join(tagged)
    except Exception as e:
        log(f"DuckDuckGo research failed: {e}")
        research = ""
    if research:
        log(f"Found {len(research.splitlines())} snippets.")

    # Wikipedia extracts: reachable from CI runners when DuckDuckGo is not, and carry the
    # concrete, citable figures a count_up beat needs.
    wiki = wikipedia_research(news)
    if wiki:
        research = (research + "\n" + wiki).strip()
    if research:
        return research
    return f"Topic: {news}\n(No live research available — script must stay general.)"


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "DYPOL-verticals/1.0 (https://github.com/rstmcsoch/youtube-shorts-pipeline)"}


def _wiki_query(news: str) -> str:
    q = re.sub(r"^(how|why|what|when|does|do|can|is|are)\b", "", str(news or ""), flags=re.I)
    q = re.sub(r"\b(ai|your|you|the|a|an|with|its)\b", " ", q, flags=re.I)
    return " ".join(q.split()) or str(news or "")


def _wiki_search(query: str, limit: int) -> list[str]:
    r = requests.get(WIKI_API, params={"action": "query", "list": "search", "srsearch": query,
                                       "srlimit": limit, "format": "json"}, headers=WIKI_HEADERS, timeout=10)
    r.raise_for_status()
    return [h["title"] for h in r.json().get("query", {}).get("search", [])][:limit]


def wikipedia_research(news: str, pages: int = 3, chars: int = 800) -> str:
    """Lead-section extracts of the top Wikipedia hits for the topic plus its machine-learning
    angle ("how AI ..." topics need the mechanism, not only background), each line tagged."""
    try:
        core = _wiki_query(news)
        titles = []
        for query, limit in ((core, 2), (core + " machine learning", 2)):
            for t in _wiki_search(query, limit):
                if t not in titles and len(titles) < pages:
                    titles.append(t)
        if not titles:
            return ""
        r = requests.get(WIKI_API, params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                           "titles": "|".join(titles), "format": "json"},
                         headers=WIKI_HEADERS, timeout=10)
        r.raise_for_status()
        out = []
        for page in r.json().get("query", {}).get("pages", {}).values():
            text = " ".join(str(page.get("extract") or "").split())[:chars]
            if text:
                out.append(f"[source: Wikipedia, {page.get('title', '')}] {text}")
        if out:
            log(f"Wikipedia research: {', '.join(titles)}")
        return "\n".join(out)
    except Exception as e:
        log(f"Wikipedia research failed: {e}")
        return ""
