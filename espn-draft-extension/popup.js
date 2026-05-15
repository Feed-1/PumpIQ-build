'use strict';

const TEMPLATE_CSV = `Rank,Name,Position,Team,Notes
1,Christian McCaffrey,RB,SF,
2,CeeDee Lamb,WR,DAL,
3,Tyreek Hill,WR,MIA,
4,Ja'Marr Chase,WR,CIN,
5,Justin Jefferson,WR,MIN,
6,Bijan Robinson,RB,ATL,
7,Amon-Ra St. Brown,WR,DET,
8,Puka Nacua,WR,LAR,
9,Sam LaPorta,TE,DET,
10,Josh Allen,QB,BUF,
`;

function parseCSV(text) {
  const lines = text.trim().split('\n').filter(l => l.trim());
  const players = [];

  for (const line of lines) {
    const cols = parseCSVLine(line);
    if (!cols.length) continue;

    // Skip header row
    const first = cols[0].trim().toLowerCase();
    if (first === 'rank' || first === '#') continue;

    const rank = parseInt(cols[0], 10);
    if (isNaN(rank)) continue;

    const name = (cols[1] || '').trim();
    if (!name) continue;

    const position = (cols[2] || '').trim().toUpperCase();
    const team = (cols[3] || '').trim().toUpperCase();
    const notes = (cols[4] || '').trim();

    players.push({ rank, name, position, team, notes });
  }

  return players.sort((a, b) => a.rank - b.rank);
}

function parseCSVLine(line) {
  const cols = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      cols.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  cols.push(current);
  return cols;
}

function showStatus(msg, type) {
  const el = document.getElementById('uploadStatus');
  el.textContent = msg;
  el.className = `status ${type}`;
}

function updateRankingsUI(rankings) {
  const section = document.getElementById('rankingsSection');
  const summary = document.getElementById('rankingsSummary');

  if (!rankings || !rankings.length) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';

  const positions = {};
  rankings.forEach(p => {
    positions[p.position] = (positions[p.position] || 0) + 1;
  });

  const posStr = Object.entries(positions)
    .sort((a, b) => b[1] - a[1])
    .map(([pos, count]) => `${pos}: ${count}`)
    .join(' · ');

  summary.innerHTML = `
    <strong>${rankings.length} players loaded</strong><br>
    ${posStr}
  `;
}

async function loadExistingRankings() {
  const data = await chrome.storage.local.get(['rankings']);
  updateRankingsUI(data.rankings || []);
}

document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const label = document.getElementById('uploadLabel');
  document.getElementById('uploadIcon').textContent = '⏳';
  document.getElementById('uploadText').textContent = 'Processing...';

  try {
    const text = await file.text();
    const rankings = parseCSV(text);

    if (!rankings.length) {
      showStatus('No valid players found. Check your CSV format.', 'error');
      document.getElementById('uploadIcon').textContent = '📂';
      document.getElementById('uploadText').textContent = 'Choose CSV File';
      return;
    }

    await chrome.storage.local.set({ rankings, draftedPlayers: [] });

    // Notify any active draft tabs
    const tabs = await chrome.tabs.query({ url: 'https://fantasy.espn.com/*' });
    for (const tab of tabs) {
      chrome.tabs.sendMessage(tab.id, { type: 'RANKINGS_UPDATED', rankings }).catch(() => {});
    }

    showStatus(`✓ Loaded ${rankings.length} players successfully!`, 'success');
    document.getElementById('uploadIcon').textContent = '✅';
    document.getElementById('uploadText').textContent = file.name;
    updateRankingsUI(rankings);

  } catch (err) {
    showStatus('Error reading file: ' + err.message, 'error');
    document.getElementById('uploadIcon').textContent = '📂';
    document.getElementById('uploadText').textContent = 'Choose CSV File';
  }
});

document.getElementById('clearBtn').addEventListener('click', async () => {
  await chrome.storage.local.set({ rankings: [], draftedPlayers: [] });

  const tabs = await chrome.tabs.query({ url: 'https://fantasy.espn.com/*' });
  for (const tab of tabs) {
    chrome.tabs.sendMessage(tab.id, { type: 'RANKINGS_UPDATED', rankings: [] }).catch(() => {});
  }

  updateRankingsUI([]);
  document.getElementById('uploadIcon').textContent = '📂';
  document.getElementById('uploadText').textContent = 'Choose CSV File';
  document.getElementById('uploadStatus').className = 'status hidden';
  document.getElementById('fileInput').value = '';
});

document.getElementById('showBoardBtn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url && tab.url.includes('fantasy.espn.com')) {
    chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_BOARD' }).catch(() => {});
    window.close();
  } else {
    showStatus('Navigate to an ESPN fantasy draft page first.', 'error');
  }
});

document.getElementById('downloadTemplate').addEventListener('click', (e) => {
  e.preventDefault();
  const blob = new Blob([TEMPLATE_CSV], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'draft-rankings-template.csv';
  a.click();
  URL.revokeObjectURL(url);
});

loadExistingRankings();
