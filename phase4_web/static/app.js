/* Agentic AI — frontend logic.
   Wires the cards together: create project → WebSocket progress → preview →
   edit → version history. State lives in a tiny store (`current`). */

'use strict';

const $ = (id) => document.getElementById(id);
const elPrompt = $('prompt');
const elScenes = $('num-scenes');
const elStyle = $('style');
const elRun = $('run-btn');
const elRunStatus = $('run-status');
const elProgressCard = $('progress-card');
const elPhaseBadge = $('phase-badge');
const elLog = $('log');
const elPreviewCard = $('preview-card');
const elVideo = $('video');
const elPreviewVersion = $('preview-version');
const elPreviewNote = $('preview-note');
const elBodyStory = $('body-story');
const elBodyChars = $('body-characters');
const elBodyScenes = $('body-scenes');
const elBodyAudio = $('body-audio');
const elEditCard = $('edit-card');
const elEditQuery = $('edit-query');
const elEditBtn = $('edit-btn');
const elEditStatus = $('edit-status');
const elEditBadge = $('edit-badge');
const elIntentPreview = $('intent-preview');
const elHistoryCard = $('history-card');
const elHistoryList = $('history-list');
const elProjectsList = $('projects-list');
const elRefresh = $('refresh-projects');
const elHealth = $('health-badge');

const current = {
  projectId: null,
  ws: null,
};

// ── Bootstrap ─────────────────────────────────────────────
checkHealth();
loadProjects();

elRun.addEventListener('click', startProject);
elEditBtn.addEventListener('click', submitEdit);
elRefresh.addEventListener('click', loadProjects);

document.querySelectorAll('.quick-edits .chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    elEditQuery.value = chip.dataset.q || '';
    elEditQuery.focus();
  });
});
document.querySelectorAll('.meta-tabs .tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.meta-tabs .tab').forEach((t) => t.classList.remove('is-active'));
    document.querySelectorAll('.meta-body').forEach((b) => b.classList.remove('is-active'));
    tab.classList.add('is-active');
    document.getElementById('body-' + tab.dataset.tab).classList.add('is-active');
  });
});

// ── Health ────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const j = await r.json();
    if (j.status === 'ok') {
      const flags = [];
      if (j.openai_configured) flags.push('OpenAI');
      else flags.push('offline-stub');
      if (j.ffmpeg_available) flags.push('ffmpeg');
      if (j.moviepy_available) flags.push('moviepy');
      elHealth.textContent = flags.join(' · ');
      elHealth.dataset.state = 'ok';
    } else {
      elHealth.textContent = 'degraded';
      elHealth.dataset.state = 'error';
    }
  } catch {
    elHealth.textContent = 'server unreachable';
    elHealth.dataset.state = 'error';
  }
}

// ── Project list ──────────────────────────────────────────
async function loadProjects() {
  try {
    const r = await fetch('/api/projects');
    const j = await r.json();
    elProjectsList.innerHTML = '';
    (j.projects || []).slice(0, 8).forEach((p) => {
      const li = document.createElement('li');
      li.className = 'project-item';
      li.innerHTML =
        `<span class="v">v${p.version}</span>` +
        `<span class="summary">${escapeHtml(p.title)} · ${p.phase}</span>` +
        `<button class="btn ghost small">open</button>`;
      li.querySelector('button').addEventListener('click', () => openProject(p.project_id));
      elProjectsList.appendChild(li);
    });
    if (!j.projects.length) {
      elProjectsList.innerHTML = '<li class="hint">No projects yet — run one above.</li>';
    }
  } catch {
    elProjectsList.innerHTML = '<li class="hint">Could not load projects.</li>';
  }
}

// ── Run a new project ─────────────────────────────────────
async function startProject() {
  const prompt = elPrompt.value.trim();
  if (!prompt) {
    elRunStatus.textContent = 'Prompt required';
    return;
  }
  elRun.disabled = true;
  elRunStatus.textContent = 'starting…';
  elProgressCard.hidden = false;
  resetPhaseStrip();
  elLog.textContent = '';
  setBadge(elPhaseBadge, 'working', 'starting');

  try {
    const r = await fetch('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        num_scenes: parseInt(elScenes.value, 10) || 3,
        style: elStyle.value,
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    current.projectId = j.project_id;
    elRunStatus.textContent = `project ${j.project_id}`;
    openWebSocket(j.project_id, async () => {
      await refreshProject();
      loadProjects();
    });
  } catch (e) {
    setBadge(elPhaseBadge, 'error', 'error');
    elRunStatus.textContent = 'failed';
    appendLog(`[ERROR] ${e.message || e}`);
  } finally {
    elRun.disabled = false;
  }
}

// ── Open an existing project ──────────────────────────────
async function openProject(projectId) {
  current.projectId = projectId;
  elProgressCard.hidden = false;
  resetPhaseStrip();
  elLog.textContent = `loaded project ${projectId}\n`;
  setBadge(elPhaseBadge, 'ok', 'loaded');
  await refreshProject();
}

// ── WebSocket progress ────────────────────────────────────
function openWebSocket(projectId, onDone) {
  if (current.ws) {
    try { current.ws.close(); } catch {}
  }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/projects/${projectId}/ws`);
  current.ws = ws;

  ws.onmessage = (ev) => {
    const msg = ev.data;
    if (msg === '__DONE__') {
      ws.close();
      setBadge(elPhaseBadge, 'ok', 'done');
      elRunStatus.textContent = 'done';
      if (typeof onDone === 'function') onDone();
      return;
    }
    appendLog(msg);
    advancePhaseStrip(msg);
  };
  ws.onerror = () => appendLog('[ws] error');
  ws.onclose = () => { /* fine */ };
}

function appendLog(msg) {
  elLog.textContent += msg + '\n';
  elLog.scrollTop = elLog.scrollHeight;
}

function resetPhaseStrip() {
  document.querySelectorAll('.phase-strip li').forEach((li) => {
    li.classList.remove('is-active', 'is-done');
  });
}

function advancePhaseStrip(msg) {
  const map = [
    ['PHASE 1', 'phase1'],
    ['PHASE 2', 'phase2'],
    ['PHASE 3', 'phase3'],
    ['DONE', 'complete'],
  ];
  for (const [token, phase] of map) {
    if (msg.includes(token)) {
      document.querySelectorAll('.phase-strip li').forEach((li) => {
        if (li.dataset.phase === phase) {
          li.classList.add('is-active');
          li.classList.remove('is-done');
        } else if (
          map.findIndex(([, p]) => p === li.dataset.phase) <
          map.findIndex(([, p]) => p === phase)
        ) {
          li.classList.add('is-done');
          li.classList.remove('is-active');
        }
      });
      setBadge(elPhaseBadge, phase === 'complete' ? 'ok' : 'working', phase);
      break;
    }
  }
}

// ── Refresh state + history ──────────────────────────────
async function refreshProject() {
  if (!current.projectId) return;
  const r = await fetch(`/api/projects/${current.projectId}`);
  if (!r.ok) return;
  const j = await r.json();
  if (j.state) renderState(j.state);
  if (j.history) renderHistory(j.history);
}

function renderState(state) {
  elPreviewCard.hidden = false;
  elEditCard.hidden = false;
  elHistoryCard.hidden = false;

  elPreviewVersion.textContent = `v${state.version}`;
  if (state.final_video) {
    const url = `/api/projects/${current.projectId}/file/${state.final_video}?t=${Date.now()}`;
    elVideo.src = url;
    elPreviewNote.textContent = '';
  } else {
    elPreviewNote.textContent = 'No final video yet.';
  }

  elBodyStory.textContent = JSON.stringify(
    {
      project_id: state.project_id,
      version: state.version,
      phase: state.phase,
      title: state.story && state.story.title,
      logline: state.story && state.story.logline,
      style: state.story && state.story.style,
    },
    null,
    2,
  );
  elBodyChars.textContent = JSON.stringify(state.story && state.story.characters, null, 2);
  elBodyScenes.textContent = JSON.stringify(state.story && state.story.scenes, null, 2);
  elBodyAudio.textContent = JSON.stringify(state.timing, null, 2);
}

function renderHistory(history) {
  elHistoryList.innerHTML = '';
  history
    .slice()
    .reverse()
    .forEach((entry) => {
      const li = document.createElement('li');
      li.className = 'history-item';
      const intent = entry.intent
        ? ` · ${entry.intent.intent}/${entry.intent.target}/${entry.intent.scope}`
        : '';
      li.innerHTML =
        `<span class="v">v${entry.version}</span>` +
        `<span class="summary">${escapeHtml(entry.summary)}${escapeHtml(intent)}</span>` +
        `<span class="when">${entry.created_at.replace('T', ' ').slice(0, 19)}</span>` +
        `<button class="btn ghost small">revert</button>`;
      li.querySelector('button').addEventListener('click', () => revertTo(entry.version));
      elHistoryList.appendChild(li);
    });
}

// ── Edit ────────────────────────────────────────────────
async function submitEdit() {
  if (!current.projectId) {
    elEditStatus.textContent = 'no active project';
    return;
  }
  const query = elEditQuery.value.trim();
  if (!query) {
    elEditStatus.textContent = 'enter a query';
    return;
  }
  elEditBtn.disabled = true;
  elEditStatus.textContent = 'classifying…';
  setBadge(elEditBadge, 'working', 'editing');
  appendLog(`[Edit] "${query}"`);
  try {
    const r = await fetch(`/api/projects/${current.projectId}/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    elIntentPreview.hidden = false;
    elIntentPreview.textContent = JSON.stringify(j.intent, null, 2);
    openWebSocket(current.projectId, async () => {
      await refreshProject();
      setBadge(elEditBadge, 'ok', 'applied');
      elEditStatus.textContent = 'applied';
    });
  } catch (e) {
    setBadge(elEditBadge, 'error', 'error');
    elEditStatus.textContent = 'failed';
    appendLog(`[ERROR] ${e.message || e}`);
  } finally {
    elEditBtn.disabled = false;
  }
}

// ── Revert ──────────────────────────────────────────────
async function revertTo(version) {
  if (!current.projectId) return;
  if (!confirm(`Revert to v${version}? This is itself snapshotted as a new version.`)) return;
  appendLog(`[Revert] target=v${version}`);
  try {
    const r = await fetch(`/api/projects/${current.projectId}/revert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version }),
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    appendLog(`[Revert] now on v${j.new_version}`);
    await refreshProject();
  } catch (e) {
    appendLog(`[ERROR] ${e.message || e}`);
  }
}

// ── Utilities ───────────────────────────────────────────
function setBadge(el, state, label) {
  el.textContent = label;
  el.dataset.state = state;
}
function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
