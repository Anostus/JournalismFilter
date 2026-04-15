# Journalism Quality Filter

An AI-powered RSS filter that monitors podcast feeds and selects the **top three episodes of the week** based on journalism and news quality.

## How It Works

1. **Fetches** episodes from five podcast RSS feeds every Friday at 3 AM Denver time.
2. **Scores** each episode on a single dimension — **Journalism and News Quality** — using DeepSeek R1, evaluating depth of reporting, use of primary sources, original investigation, fairness, clarity, and public significance.
3. **Selects the best episode from each feed**, then narrows those per-feed winners down to the **top three overall**.
4. **Publishes** the result to `docs/feed.xml`, which you can subscribe to in any podcast app or RSS reader.

## Podcast Sources

| Feed URL |
|----------|
| `https://audioboom.com/channels/5114286.rss` |
| `https://feeds.simplecast.com/82FI35Px` |
| `https://feeds.megaphone.fm/DISPME9513417677` |
| `https://feeds.megaphone.fm/BVLLC2163264914` |
| `https://feeds.simplecast.com/WCb5SgYj` |

## Setup

1. Fork this repository.
2. Add your `DEEPSEEK_API_KEY` as a repository secret (Settings → Secrets → Actions).
3. Enable GitHub Actions.
4. The workflow runs automatically every Friday, or trigger it manually via the Actions tab.

## Subscribing to the Feed

Once GitHub Pages is enabled on the `docs/` folder (or you use the raw file URL), subscribe to the output feed in your podcast app:

```
https://your-username.github.io/journalism-filter/feed.xml
```

## Configuration

All settings can be overridden via GitHub Actions repository variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RSS_FEEDS` | *(5 podcast feeds)* | JSON array of feed URLs |
| `MAX_ENTRIES_PER_FEED` | `25` | Max episodes to pull per feed |
| `BATCH_SIZE` | `10` | Episodes per LLM scoring call |
| `FEED_RETENTION_DAYS` | `14` | Days to keep past winners in the output feed |
| `LOOKBACK_DAYS` | `7` | Only consider episodes published within this window |

## Based On

Adapted from the [Eudaimonia Filter](https://github.com/your-username/eudaimonia-filter), which filters news articles across three philosophical dimensions.
