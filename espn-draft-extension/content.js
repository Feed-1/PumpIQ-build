'use strict';

// ─── Constants ───────────────────────────────────────────────────────────────

const POSITION_COLORS = {
  QB:  { bg: '#7f1d1d', text: '#fca5a5', border: '#991b1b' },
  RB:  { bg: '#14532d', text: '#86efac', border: '#166534' },
  WR:  { bg: '#1e3a5f', text: '#93c5fd', border: '#1e40af' },
  TE:  { bg: '#78350f', text: '#fcd34d', border: '#92400e' },
  K:   { bg: '#2d1b69', text: '#c4b5fd', border: '#4c1d95' },
  DEF: { bg: '#1c1917', text: '#d6d3d1', border: '#292524' },
  DST: { bg: '#1c1917', text: '#d6d3d1', border: '#292524' },
  '':  { bg: '#1e293b', text: '#94a3b8', border: '#334155' },
};

// Selectors ESPN uses (may vary by sport/year — we try multiple)
const DRAFTED_SELECTORS = [
  '.pick--selected',
  '.playerTableRow--drafted',
  '.player-drafted',
  '[class*="drafted"]',
  '[class*="Drafted"]',
];

const PICK_LIST_SELECTORS = [
  '.pick-history',
  '.draftboard',
  '[class*="DraftBoard"]',
  '[class*="pickHistory"]',
  '[class*="pick-history"]',
  '[class*="RecentPicks"]',
  '[class*="recentPicks"]',
  '.recent-picks',
];

const PLAYER_NAME_SELECTORS = [
  '.playerName',
  '.player-name',
  '[class*="playerName"]',
  '[class*="PlayerName"]',
  '.athlete-name',
  '.player__name',
  '.name',
];

// ─── Main Class ──────────────────────────────────────────────────────────────

class DraftBigBoard {
  constructor() {
    this.rankings = [];
    this.draftedSet = new Set();   // normalized lowercase names
    this.positionFilter = 'ALL';
    this.searchQuery = '';
    this.hideDrafted = true;
    this.panel = null;
    this.observer = null;
    this.pickObserver = null;
    this.visible = true;
    this.seenPickTexts = new Set();
    this.contextMenu = null;
    this.rightClickPlayer = null;
  }

  async init() {
    const data = await chrome.storage.local.get(['rankings', 'draftedPlayers']);
    this.rankings = data.rankings || [];
    (data.draftedPlayers || []).forEach(n => this.draftedSet.add(normalize(n)));

    this.injectPanel();
    this.startDraftMonitor();
    this.listenMessages();
  }

  // ─── Panel Injection ──────────────────────────────────────────────────────

  injectPanel() {
    if (this.panel) this.panel.remove();

    this.panel = document.createElement('div');
    this.panel.id = 'ebb-panel';
    this.panel.innerHTML = this.buildPanelHTML();
    document.body.appendChild(this.panel);

    this.attachPanelEvents();
    this.renderPlayerList();
    this.makeDraggable();
  }

  buildPanelHTML() {
    return `
      <div id="ebb-header">
        <div id="ebb-title">
          <span id="ebb-icon">🏈</span>
          <span>Big Board</span>
        </div>
        <div id="ebb-header-btns">
          <button id="ebb-minimize" title="Minimize">─</button>
          <button id="ebb-close" title="Close">✕</button>
        </div>
      </div>

      <div id="ebb-body">
        <div id="ebb-toolbar">
          <input id="ebb-search" type="text" placeholder="Search player..." autocomplete="off">
          <div id="ebb-pos-filters">
            ${['ALL','QB','RB','WR','TE','K','DEF'].map(p => `
              <button class="ebb-pos-btn ${p === 'ALL' ? 'active' : ''}" data-pos="${p}">${p}</button>
            `).join('')}
          </div>
          <div id="ebb-options">
            <label class="ebb-toggle-label">
              <input type="checkbox" id="ebb-hide-drafted" ${this.hideDrafted ? 'checked' : ''}>
              Hide drafted
            </label>
          </div>
        </div>

        <div id="ebb-stats"></div>

        <div id="ebb-list-container">
          <div id="ebb-list"></div>
        </div>

        <div id="ebb-empty" class="hidden">
          <div>📋</div>
          <div>Upload rankings in the extension popup to get started.</div>
        </div>
      </div>

      <div id="ebb-resize-handle" title="Resize">⠿</div>
    `;
  }

  attachPanelEvents() {
    this.panel.querySelector('#ebb-close').onclick = () => this.hide();
    this.panel.querySelector('#ebb-minimize').onclick = () => this.toggleMinimize();

    const search = this.panel.querySelector('#ebb-search');
    search.oninput = () => {
      this.searchQuery = search.value.trim().toLowerCase();
      this.renderPlayerList();
    };

    this.panel.querySelectorAll('.ebb-pos-btn').forEach(btn => {
      btn.onclick = () => {
        this.panel.querySelectorAll('.ebb-pos-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.positionFilter = btn.dataset.pos;
        this.renderPlayerList();
      };
    });

    this.panel.querySelector('#ebb-hide-drafted').onchange = (e) => {
      this.hideDrafted = e.target.checked;
      this.renderPlayerList();
    };

    // Right-click context menu dismiss
    document.addEventListener('click', () => this.dismissContextMenu());
  }

  // ─── Player List Rendering ────────────────────────────────────────────────

  renderPlayerList() {
    const list = this.panel.querySelector('#ebb-list');
    const empty = this.panel.querySelector('#ebb-empty');
    const stats = this.panel.querySelector('#ebb-stats');

    if (!this.rankings.length) {
      list.innerHTML = '';
      empty.classList.remove('hidden');
      stats.textContent = '';
      return;
    }

    empty.classList.add('hidden');

    const filtered = this.rankings.filter(p => {
      const isDrafted = this.draftedSet.has(normalize(p.name));
      if (this.hideDrafted && isDrafted) return false;
      if (this.positionFilter !== 'ALL') {
        const pos = p.position.toUpperCase();
        const filter = this.positionFilter;
        if (filter === 'DEF' && (pos === 'DEF' || pos === 'DST')) return true;
        if (pos !== filter) return false;
      }
      if (this.searchQuery) {
        return p.name.toLowerCase().includes(this.searchQuery) ||
               p.team.toLowerCase().includes(this.searchQuery);
      }
      return true;
    });

    const totalAvailable = this.rankings.filter(p => !this.draftedSet.has(normalize(p.name))).length;
    const totalDrafted = this.draftedSet.size;
    stats.innerHTML = `
      <span class="ebb-stat">${totalAvailable} available</span>
      <span class="ebb-stat drafted">${totalDrafted} drafted</span>
    `;

    list.innerHTML = filtered.map(p => this.buildPlayerRow(p)).join('');

    // Attach right-click handlers
    list.querySelectorAll('.ebb-player-row').forEach(row => {
      row.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const name = row.dataset.name;
        this.showContextMenu(e.clientX, e.clientY, name);
      });
    });
  }

  buildPlayerRow(player) {
    const isDrafted = this.draftedSet.has(normalize(player.name));
    const posKey = player.position.toUpperCase();
    const colors = POSITION_COLORS[posKey] || POSITION_COLORS[''];
    const draftedClass = isDrafted ? ' drafted' : '';

    return `
      <div class="ebb-player-row${draftedClass}" data-name="${escapeAttr(player.name)}">
        <span class="ebb-rank">${player.rank}</span>
        <span class="ebb-pos-badge" style="background:${colors.bg};color:${colors.text};border-color:${colors.border}">${player.position || '?'}</span>
        <div class="ebb-player-info">
          <span class="ebb-player-name">${escapeHtml(player.name)}</span>
          ${player.team ? `<span class="ebb-player-team">${escapeHtml(player.team)}</span>` : ''}
          ${player.notes ? `<span class="ebb-player-notes">${escapeHtml(player.notes)}</span>` : ''}
        </div>
        ${isDrafted ? '<span class="ebb-drafted-badge">DRAFTED</span>' : ''}
      </div>
    `;
  }

  // ─── Context Menu ─────────────────────────────────────────────────────────

  showContextMenu(x, y, playerName) {
    this.dismissContextMenu();
    this.rightClickPlayer = playerName;
    const isDrafted = this.draftedSet.has(normalize(playerName));

    const menu = document.createElement('div');
    menu.id = 'ebb-context-menu';
    menu.innerHTML = `
      <div class="ebb-ctx-player">${escapeHtml(playerName)}</div>
      ${isDrafted
        ? `<button class="ebb-ctx-btn" data-action="undraft">↩ Mark Available</button>`
        : `<button class="ebb-ctx-btn" data-action="draft">✓ Mark as Drafted</button>`
      }
    `;

    // Position near cursor, keep in viewport
    menu.style.position = 'fixed';
    menu.style.left = Math.min(x, window.innerWidth - 200) + 'px';
    menu.style.top = Math.min(y, window.innerHeight - 100) + 'px';

    document.body.appendChild(menu);
    this.contextMenu = menu;

    menu.querySelectorAll('.ebb-ctx-btn').forEach(btn => {
      btn.onclick = (e) => {
        e.stopPropagation();
        if (btn.dataset.action === 'draft') {
          this.markDrafted(playerName);
        } else {
          this.markAvailable(playerName);
        }
        this.dismissContextMenu();
      };
    });
  }

  dismissContextMenu() {
    if (this.contextMenu) {
      this.contextMenu.remove();
      this.contextMenu = null;
    }
  }

  // ─── Draft State Management ───────────────────────────────────────────────

  markDrafted(name) {
    this.draftedSet.add(normalize(name));
    this.persistDrafted();
    this.renderPlayerList();
  }

  markAvailable(name) {
    this.draftedSet.delete(normalize(name));
    this.persistDrafted();
    this.renderPlayerList();
  }

  persistDrafted() {
    chrome.storage.local.set({ draftedPlayers: [...this.draftedSet] });
  }

  // ─── ESPN Draft Monitor ───────────────────────────────────────────────────

  startDraftMonitor() {
    // Strategy 1: Watch DOM for drafted class indicators
    this.observer = new MutationObserver(() => this.scanForDraftedPlayers());
    this.observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });

    // Strategy 2: Poll pick lists for new names
    setInterval(() => this.scanPickLists(), 3000);

    // Initial scan
    setTimeout(() => this.scanForDraftedPlayers(), 2000);
  }

  scanForDraftedPlayers() {
    let found = false;

    // Check elements with drafted classes
    for (const sel of DRAFTED_SELECTORS) {
      document.querySelectorAll(sel).forEach(el => {
        const name = extractPlayerName(el);
        if (name && !this.draftedSet.has(normalize(name))) {
          const matched = this.findRankedPlayer(name);
          if (matched) {
            this.markDrafted(matched.name);
            found = true;
          }
        }
      });
    }

    return found;
  }

  scanPickLists() {
    for (const sel of PICK_LIST_SELECTORS) {
      document.querySelectorAll(sel).forEach(container => {
        extractAllPlayerNames(container).forEach(name => {
          const key = name.toLowerCase();
          if (this.seenPickTexts.has(key)) return;
          const matched = this.findRankedPlayer(name);
          if (matched) {
            this.seenPickTexts.add(key);
            if (!this.draftedSet.has(normalize(matched.name))) {
              this.markDrafted(matched.name);
            }
          }
        });
      });
    }
  }

  findRankedPlayer(rawName) {
    const normInput = normalize(rawName);
    // Exact match first
    let match = this.rankings.find(p => normalize(p.name) === normInput);
    if (match) return match;

    // Partial match — input contains player name or vice versa
    match = this.rankings.find(p => {
      const pn = normalize(p.name);
      return normInput.includes(pn) || pn.includes(normInput);
    });
    return match || null;
  }

  // ─── Draggable ────────────────────────────────────────────────────────────

  makeDraggable() {
    const header = this.panel.querySelector('#ebb-header');
    let dragging = false, startX, startY, origLeft, origTop;

    header.addEventListener('mousedown', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = this.panel.getBoundingClientRect();
      origLeft = rect.left;
      origTop = rect.top;
      document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      this.panel.style.left = Math.max(0, origLeft + dx) + 'px';
      this.panel.style.top = Math.max(0, origTop + dy) + 'px';
      this.panel.style.right = 'auto';
    });

    document.addEventListener('mouseup', () => {
      dragging = false;
      document.body.style.userSelect = '';
    });

    // Resize handle
    const handle = this.panel.querySelector('#ebb-resize-handle');
    let resizing = false, rStartX, rStartY, rOrigW, rOrigH;

    handle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      resizing = true;
      rStartX = e.clientX;
      rStartY = e.clientY;
      rOrigW = this.panel.offsetWidth;
      rOrigH = this.panel.offsetHeight;
      document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
      if (!resizing) return;
      const w = Math.max(260, rOrigW + (e.clientX - rStartX));
      const h = Math.max(200, rOrigH + (e.clientY - rStartY));
      this.panel.style.width = w + 'px';
      this.panel.style.height = h + 'px';
    });

    document.addEventListener('mouseup', () => {
      resizing = false;
      document.body.style.userSelect = '';
    });
  }

  // ─── Visibility ───────────────────────────────────────────────────────────

  toggleMinimize() {
    const body = this.panel.querySelector('#ebb-body');
    const btn = this.panel.querySelector('#ebb-minimize');
    if (body.style.display === 'none') {
      body.style.display = '';
      btn.textContent = '─';
    } else {
      body.style.display = 'none';
      btn.textContent = '□';
    }
  }

  hide() {
    this.panel.style.display = 'none';
    this.visible = false;
    this.showFAB();
  }

  show() {
    this.panel.style.display = '';
    this.visible = true;
    const fab = document.getElementById('ebb-fab');
    if (fab) fab.remove();
  }

  showFAB() {
    let fab = document.getElementById('ebb-fab');
    if (!fab) {
      fab = document.createElement('button');
      fab.id = 'ebb-fab';
      fab.title = 'Show Big Board';
      fab.textContent = '🏈';
      fab.onclick = () => this.show();
      document.body.appendChild(fab);
    }
  }

  // ─── Message Listener ─────────────────────────────────────────────────────

  listenMessages() {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg.type === 'RANKINGS_UPDATED') {
        this.rankings = msg.rankings || [];
        this.draftedSet.clear();
        this.seenPickTexts.clear();
        this.renderPlayerList();
      }
      if (msg.type === 'TOGGLE_BOARD') {
        if (this.visible) this.hide(); else this.show();
      }
    });
  }
}

// ─── Helper Functions ─────────────────────────────────────────────────────────

function normalize(name) {
  return name.toLowerCase()
    .replace(/['']/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escapeAttr(str) {
  return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function extractPlayerName(el) {
  for (const sel of PLAYER_NAME_SELECTORS) {
    const found = el.querySelector(sel);
    if (found && found.textContent.trim()) return found.textContent.trim();
  }
  // Fallback: look for text that looks like a name (2+ words, mostly letters)
  const text = el.textContent.trim();
  const match = text.match(/^([A-Z][a-z]+'?\s+[A-Z][a-z']+(?:\s+[A-Z][a-z]+)?)/);
  return match ? match[1] : null;
}

function extractAllPlayerNames(container) {
  const names = new Set();
  for (const sel of PLAYER_NAME_SELECTORS) {
    container.querySelectorAll(sel).forEach(el => {
      const t = el.textContent.trim();
      if (t.length > 3 && t.length < 60) names.add(t);
    });
  }
  // Also scan all text nodes for name-like patterns
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const text = node.textContent.trim();
    const match = text.match(/^([A-Z][a-z]+'?\.?\s+[A-Z][a-z']+(?:\s+[A-Z][a-z]+)?)$/);
    if (match) names.add(match[1]);
  }
  return [...names];
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

const board = new DraftBigBoard();
board.init();
