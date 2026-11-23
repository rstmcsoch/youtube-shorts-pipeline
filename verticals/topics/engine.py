"""TopicEngine — orchestrates multi-source discovery + Claude auto-pick."""

import concurrent.futures

from ..config import load_config, NICHE_TO_SUBREDDITS
from ..log import log
from .base import TopicCandidate


class TopicEngine:
    """Fetches from all enabled sources, deduplicates, ranks."""

    def __init__(self, niche: str = "general"):
        self._niche = niche or "general"
        self._sources = []
        self._load_sources()

    def _load_sources(self):
        """Load enabled topic sources from config.

        When a niche is set, subreddit and NewsAPI query defaults are overridden
        with niche-appropriate values (user config.json can still override).
        """
        config = load_config()
        source_config = config.get("topic_sources", {})

        # Always register these — they'll check their own enabled status
        from .reddit import RedditSource
        from .rss import RSSSource
        from .google_trends import GoogleTrendsSource

        source_map = {
            "reddit": RedditSource,
            "rss": RSSSource,
            "google_trends": GoogleTrendsSource,
        }

        # Optional sources
        try:
            from .newsapi import NewsAPISource
            source_map["newsapi"] = NewsAPISource
        except ImportError:
            pass

        try:
            from .twitter import TwitterSource
            source_map["twitter"] = TwitterSource
        except ImportError:
            pass

        try:
            from .tiktok import TikTokSource
            source_map["tiktok"] = TikTokSource
        except ImportError:
            pass

        for name, cls in source_map.items():
            src_cfg = dict(source_config.get(name, {}))  # shallow copy so we can mutate

            # Apply niche defaults when no explicit config is set by user
            if self._niche != "general":
                if name == "reddit" and "subreddits" not in src_cfg:
                    niche_subs = NICHE_TO_SUBREDDITS.get(self._niche, [])
                    if niche_subs:
                        src_cfg["subreddits"] = niche_subs
                if name == "newsapi":
                    src_cfg.setdefault("niche", self._niche)

            # NewsAPI enabled if key is present (checked by is_available); others default on/off
            default_enabled = name in ("reddit", "rss", "google_trends", "newsapi")
            if src_cfg.get("enabled", default_enabled):
                try:
                    self._sources.append(cls(src_cfg))
                except Exception as e:
                    log(f"Failed to init source {name}: {e}")

    def discover(self, limit: int = 15) -> list[TopicCandidate]:
        """Fetch from all sources in parallel, deduplicate, rank."""
        all_topics = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(src.fetch_topics, limit): src
                for src in self._sources if src.is_available
            }
            for future in concurrent.futures.as_completed(futures):
                src = futures[future]
                try:
                    topics = future.result(timeout=15)
                    all_topics.extend(topics)
                    log(f"{src.name}: found {len(topics)} topics")
                except Exception as e:
                    log(f"{src.name}: failed — {e}")

        # Deduplicate by fuzzy title matching
        seen = set()
        unique = []
        for t in all_topics:
            key = t.title.lower().strip()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(t)

        # Sort by trending score (highest first)
        unique.sort(key=lambda t: t.trending_score, reverse=True)
        return unique[:limit]

    def auto_pick(self, candidates: list[TopicCandidate]) -> str:
        """Pick the best topic for a YouTube Short with whatever LLM is configured.

        DYPOL (23 Nov 2026): the pipeline runs on Gemini, so a blank topic used to crash
        with "No Claude access found". This now uses llm.call_llm (Gemini first), prefers
        everyday-AI explainer angles, and falls back to the top-ranked candidate so a
        missing topic never fails the run.
        """
        if not candidates:
            raise RuntimeError("No topic candidates found and no topic was given.")
        topics_text = "\n".join(
            f"{i+1}. [{t.source}] {t.title} (score: {t.trending_score:.2f})"
            for i, t in enumerate(candidates[:20])
        )

        prompt = f"""Pick the single best topic from this list for a calm, clear YouTube Short (45-60 sec)
that explains how AI or technology works in everyday life.
Prefer: visual potential, broad appeal, something a viewer can learn from, no named private people.
Avoid: politics, tragedy, celebrity gossip, anything that needs a person's name.
If none fit, rewrite the most suitable one as a short everyday-AI explainer title.

{topics_text}

Reply with ONLY the topic title text, nothing else."""

        try:
            from ..llm import call_llm
            picked = (call_llm(prompt, max_tokens=200) or "").strip().strip('"').splitlines()[0].strip()
            if picked:
                return picked[:120]
        except Exception as exc:  # noqa: BLE001 - never fail the run on topic choice
            log(f"Topic auto-pick via LLM failed ({exc}); using top-ranked candidate")
        return candidates[0].title
