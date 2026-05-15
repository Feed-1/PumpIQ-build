# ESPN Fantasy Draft Big Board

A Chrome extension that overlays your custom player rankings on ESPN fantasy draft pages, automatically removing players as they get drafted.

## Features

- **Custom Rankings Upload** — Upload any CSV with your own rankings
- **Live Draft Tracking** — Players are automatically removed as ESPN drafts them
- **Manual Override** — Right-click any player to mark drafted/available
- **Position Filters** — Filter by QB, RB, WR, TE, K, DEF
- **Search** — Find any player instantly
- **Draggable & Resizable** — Place the board anywhere on screen
- **Hide/Show Drafted** — Toggle between clean view and struck-through view
- **Persistent State** — Rankings survive page refreshes

## Installation

1. Open Chrome and go to `chrome://extensions`
2. Enable **Developer Mode** (top-right toggle)
3. Click **Load unpacked**
4. Select this folder (`espn-draft-extension/`)
5. The extension icon will appear in your toolbar

## Usage

1. Click the extension icon in your toolbar
2. Upload your CSV rankings file (see format below)
3. Navigate to `https://fantasy.espn.com` and start your draft
4. The Big Board panel appears automatically
5. Players disappear from your board as they get drafted

## CSV Format

```
Rank,Name,Position,Team,Notes
1,Christian McCaffrey,RB,SF,Elite RB1
2,CeeDee Lamb,WR,DAL,
3,Tyreek Hill,WR,MIA,
```

- **Rank** — Integer, your overall rank (required)
- **Name** — Player full name (required)
- **Position** — QB / RB / WR / TE / K / DEF (optional but recommended)
- **Team** — NFL team abbreviation (optional)
- **Notes** — Your personal notes (optional)

A `sample-rankings.csv` file is included to get you started.

## How Draft Detection Works

The extension uses two strategies to detect drafted players:

1. **DOM Observer** — Watches for ESPN's CSS classes that mark players as drafted (`.pick--selected`, `[class*="drafted"]`, etc.)
2. **Pick List Scanner** — Every 3 seconds scans the draft board / pick history section for new player names and matches them against your rankings

If auto-detection misses a pick, **right-click** any player on your board and select "Mark as Drafted."

## Troubleshooting

**Board doesn't appear?**
- Make sure you're on a page matching `fantasy.espn.com/*draft*`
- Click the extension icon → "Show Big Board"

**Players not being auto-removed?**
- ESPN occasionally changes their CSS class names — use right-click to mark manually
- The 3-second polling should catch most picks via the pick history panel

**Rankings cleared after refresh?**
- Rankings are saved in `chrome.storage.local` — they persist across refreshes
- Only clearing via the popup removes them
