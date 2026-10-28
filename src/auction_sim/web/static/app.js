(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const nodes = {
    connection: $('connection'), connectionText: $('connection-text'), runId: $('run-id'), runStatus: $('run-status'),
    toggle: $('toggle-run'), toggleLabel: $('toggle-label'), toggleIcon: $('toggle-icon'), restart: $('restart'),
    speed: $('speed'), mechanism: $('mechanism'), clock: $('sim-clock'), horizon: $('sim-horizon'),
    pending: $('pending-count'), pendingNote: $('pending-note'), active: $('active-slots'), capacity: $('slot-capacity'), capacityFill: $('capacity-fill'),
    capacityNote: $('capacity-note'), kv: $('kv-percent'), kvFill: $('kv-fill'), kvNote: $('kv-note'),
    price: $('last-price'), completed: $('completed-count'), timeout: $('timeout-count'),
    inputBody: $('input-book'), outputBody: $('output-book'), inputEmpty: $('input-empty'), outputEmpty: $('output-empty'),
    inputLevels: $('input-level-count'), outputLevels: $('output-level-count'), inputFloor: $('input-floor'), outputFloor: $('output-floor'),
    historySvg: $('price-svg'), historyEmpty: $('history-empty'), historyStart: $('history-start'), historyEnd: $('history-end'),
    seed: $('seed-value'), load: $('load-value'), notice: $('notice')
  };
  let latest = null;
  let polling = false;
  let controlBusy = false;
  let noticeTimer = null;
  let stopped = false;
  let failures = 0;
  let requestVersion = 0;
  let stateAbort = null;

  const asNumber = (value) => value === null || value === undefined || value === '' || typeof value === 'boolean'
    ? null : Number.isFinite(Number(value)) ? Number(value) : null;
  const displayInt = (value) => {
    const n = asNumber(value);
    return n === null ? '—' : Math.max(0, Math.round(n)).toLocaleString('en-US');
  };
  const fmtUsd = (perToken) => {
    const n = asNumber(perToken);
    return n === null ? '—' : '$' + (n * 1000).toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 3});
  };
  const fmtTime = (seconds) => {
    const n = asNumber(seconds);
    if (n === null) return '—';
    const total = Math.max(0, Math.floor(n));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };
  const setConnection = (state, label) => {
    nodes.connection.dataset.state = state;
    nodes.connectionText.textContent = label;
  };
  const showNotice = (message) => {
    nodes.notice.textContent = message;
    nodes.notice.hidden = false;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { nodes.notice.hidden = true; }, 4200);
  };

  function normalizeStatus(state) {
    if (state.status === 'error') return 'error';
    if (state.status === 'finished') return 'finished';
    if (state.status === 'paused') return 'paused';
    return 'running';
  }

  function setBook(body, empty, countNode, book, kind) {
    body.replaceChildren();
    const levels = Array.isArray(book?.levels) ? book.levels : [];
    countNode.textContent = book?.truncated
      ? `${levels.length} of ${displayInt(book.total_levels)}` : String(levels.length);
    empty.hidden = levels.length > 0;
    if (!levels.length) return;
    const vals = levels.map((level) => Math.max(0, asNumber(kind === 'input' ? level.input_tokens : level.output_tokens) || 0));
    const max = Math.max(1, ...vals);
    for (let i = 0; i < levels.length; i += 1) {
      const level = levels[i];
      const row = document.createElement('tr');
      const price = document.createElement('td');
      price.className = 'bid-price mono';
      price.textContent = fmtUsd(level.bid_per_token);
      row.append(price);

      const qty = kind === 'input' ? level.input_tokens : level.output_tokens;
      const cumulative = kind === 'input' ? level.cumulative_input_tokens : level.cumulative_output_tokens;
      const quantity = document.createElement('td');
      quantity.className = 'mono depth-cell';
      const bar = document.createElement('span');
      bar.className = 'depth-bar';
      bar.setAttribute('aria-hidden', 'true');
      bar.style.width = `${Math.max(0, Math.min(100, (vals[i] / max) * 100))}%`;
      quantity.append(bar, document.createTextNode(displayInt(qty)));
      row.append(quantity);

      const cumulativeCell = document.createElement('td');
      cumulativeCell.className = 'mono';
      cumulativeCell.textContent = displayInt(cumulative);
      row.append(cumulativeCell);

      const orders = document.createElement('td');
      orders.className = 'mono';
      orders.textContent = displayInt(level.orders);
      row.append(orders);
      body.append(row);
    }
  }

  function drawHistory(history) {
    const svg = nodes.historySvg;
    svg.replaceChildren();
    const records = Array.isArray(history) ? history.filter((p) => asNumber(p.time_s) !== null && asNumber(p.price_per_token) !== null) : [];
    nodes.historyEmpty.hidden = records.length > 0;
    if (!records.length) {
      nodes.historyStart.textContent = '—';
      nodes.historyEnd.textContent = '—';
      return;
    }
    const W = 720, H = 176, left = 39, right = 9, top = 13, bottom = 18;
    const prices = records.map((p) => Number(p.price_per_token) * 1000);
    const times = records.map((p) => Number(p.time_s));
    let low = Math.min(...prices), high = Math.max(...prices);
    if (high - low < .01) { low = Math.max(0, low - .5); high += .5; }
    const pad = (high - low) * .12;
    low = Math.max(0, low - pad); high += pad;
    const t0 = Math.min(...times), t1 = Math.max(...times);
    const x = (t) => left + (t1 === t0 ? (W - left - right) / 2 : ((t - t0) / (t1 - t0)) * (W - left - right));
    const y = (p) => top + ((high - p) / (high - low)) * (H - top - bottom);
    const make = (name, attrs, text) => {
      const el = document.createElementNS('http://www.w3.org/2000/svg', name);
      Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, String(value)));
      if (text !== undefined) el.textContent = text;
      svg.append(el);
      return el;
    };
    for (let i = 0; i < 3; i += 1) {
      const yy = top + i * (H - top - bottom) / 2;
      make('line', {x1:left, y1:yy, x2:W-right, y2:yy, class:'gridline'});
      const val = high - i * (high - low) / 2;
      make('text', {x:0, y:yy + 3, class:'axis-label'}, `$${val.toFixed(1)}`);
    }
    const coords = records.map((p) => [x(Number(p.time_s)), y(Number(p.price_per_token) * 1000)]);
    const linePath = coords.map((pt, i) => `${i ? 'L' : 'M'}${pt[0].toFixed(2)},${pt[1].toFixed(2)}`).join(' ');
    const areaPath = `${linePath} L${coords[coords.length - 1][0].toFixed(2)},${(H-bottom).toFixed(2)} L${coords[0][0].toFixed(2)},${(H-bottom).toFixed(2)} Z`;
    make('path', {d:areaPath, class:'area'});
    make('path', {d:linePath, class:'line'});
    coords.forEach(([cx, cy]) => make('circle', {cx, cy, r:3.2, class:'point'}));
    nodes.historyStart.textContent = fmtTime(t0);
    nodes.historyEnd.textContent = fmtTime(t1);
  }

  function render(state) {
    latest = state;
    const status = normalizeStatus(state);
    nodes.runId.textContent = state.run_id === undefined || state.run_id === null ? '—' : String(state.run_id);
    nodes.runStatus.dataset.status = status;
    nodes.runStatus.textContent = status[0].toUpperCase() + status.slice(1);
    nodes.clock.textContent = fmtTime(state.sim_time_s);
    nodes.horizon.textContent = fmtTime(state.horizon_s);
    nodes.pending.textContent = displayInt(state.pending_count);
    const pendingInput = asNumber(state.total_pending_input_tokens);
    const pendingOutput = asNumber(state.total_pending_output_tokens);
    nodes.pendingNote.textContent = pendingInput === null || pendingOutput === null
      ? 'Across the bundled queue'
      : `${displayInt(pendingInput)} input · ${displayInt(pendingOutput)} output tokens`;
    nodes.active.textContent = displayInt(state.active_count);
    const cap = asNumber(state.batch_capacity ?? state.max_active_requests ?? state.capacity);
    nodes.capacity.textContent = displayInt(cap);
    const active = asNumber(state.active_count);
    const pct = active !== null && cap !== null && cap > 0 ? Math.max(0, Math.min(100, active / cap * 100)) : 0;
    nodes.capacityFill.style.width = `${pct}%`;
    nodes.capacityNote.textContent = cap === null ? 'Active requests · capacity unavailable' : `${displayInt(state.free_batch_slots)} slots free`;
    const reserved = asNumber(state.reserved_kv_tokens), maxKv = asNumber(state.max_kv_tokens);
    const kvPct = reserved !== null && maxKv !== null && maxKv > 0 ? Math.max(0, Math.min(100, reserved / maxKv * 100)) : null;
    nodes.kv.textContent = kvPct === null ? '—' : String(Math.round(kvPct));
    nodes.kvFill.style.width = `${kvPct || 0}%`;
    nodes.kvNote.textContent = reserved === null || maxKv === null ? 'Reserved context capacity' : `${displayInt(reserved)} / ${displayInt(maxKv)} reserved tokens`;
    nodes.price.textContent = fmtUsd(state.latest_clearing_price);
    nodes.completed.textContent = displayInt(state.completed_count);
    nodes.timeout.textContent = displayInt(state.timeout_count);
    const input = state.books?.input || {};
    const output = state.books?.output || {};
    setBook(nodes.inputBody, nodes.inputEmpty, nodes.inputLevels, input, 'input');
    setBook(nodes.outputBody, nodes.outputEmpty, nodes.outputLevels, output, 'output');
    const floor = state.reserve_per_token ?? state.reserve_floor_per_token ?? state.floor_per_token ?? null;
    nodes.inputFloor.textContent = fmtUsd(floor);
    nodes.outputFloor.textContent = fmtUsd(floor);
    drawHistory(state.price_history);
    nodes.seed.textContent = state.seed === undefined || state.seed === null ? '—' : String(state.seed);
    nodes.load.textContent = state.load_profile || state.label || '—';
    const running = status === 'running';
    // GPU work must not disable pause, restart, or other dashboard controls.
    const busy = controlBusy;
    nodes.toggle.disabled = busy || status === 'finished' || status === 'error';
    nodes.toggleLabel.textContent = running ? 'Pause' : status === 'paused' ? 'Resume' : 'Pause';
    nodes.toggleIcon.textContent = running ? 'Ⅱ' : '▶';
    nodes.toggle.setAttribute('aria-label', running ? 'Pause simulation' : 'Resume simulation');
    nodes.restart.disabled = busy;
    nodes.speed.disabled = busy || status === 'finished' || status === 'error';
    nodes.mechanism.disabled = busy || status === 'finished' || status === 'error';
    const speed = String(state.speed ?? 1);
    if (![...nodes.speed.options].some((option) => option.value === speed)) {
      nodes.speed.add(new Option(`${speed}×`, speed));
    }
    nodes.speed.value = speed;
    const mechanism = state.mechanism === 'first_price_hybrid' ? 'first_price_hybrid' : 'uniform';
    nodes.mechanism.value = mechanism;
    if (status === 'error' && state.error) showNotice(`Simulation error: ${String(state.error)}`);
  }

  async function fetchState() {
    if (stopped || polling) return;
    if (controlBusy) {
      setTimeout(fetchState, 250);
      return;
    }
    polling = true;
    const version = requestVersion;
    stateAbort = new AbortController();
    try {
      const response = await fetch('/api/state', {cache:'no-store', headers:{Accept:'application/json'}, signal:stateAbort.signal});
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const state = await response.json();
      if (!state || typeof state !== 'object') throw new Error('State response was not valid JSON');
      if (version !== requestVersion || stopped) return;
      const recovered = failures > 0;
      failures = 0;
      setConnection('live', 'Connected');
      if (recovered) nodes.notice.hidden = true;
      render(state);
    } catch (error) {
      if (error.name === 'AbortError' || version !== requestVersion || stopped) return;
      failures += 1;
      setConnection(latest ? 'stale' : 'offline', latest ? 'Connection lost · showing last update' : 'Offline');
      if (!latest) {
        nodes.runStatus.dataset.status = 'error';
        nodes.runStatus.textContent = 'Unavailable';
        showNotice('Could not connect to the local simulation. Retrying automatically.');
      }
      if (failures === 1 && latest) showNotice('Connection lost. The displayed order book is the last received update.');
    } finally {
      polling = false;
      if (!stopped) setTimeout(fetchState, 250);
    }
  }

  async function control(payload, confirmation) {
    if (controlBusy) return;
    controlBusy = true;
    requestVersion += 1;
    if (stateAbort) stateAbort.abort();
    if (latest) render(latest);
    try {
      const response = await fetch('/api/control', {
        method:'POST', headers:{'Content-Type':'application/json', Accept:'application/json'}, body:JSON.stringify(payload)
      });
      let result = {};
      try { result = await response.json(); } catch (_) { /* a successful empty response is acceptable */ }
      if (!response.ok || result.ok === false || result.error) throw new Error(result.error || `Server returned ${response.status}`);
      if (confirmation) showNotice(confirmation);
      if (result.state && typeof result.state === 'object') render(result.state);
      else if (result.status && result.run_id) render(result);
    } catch (error) {
      showNotice(`Control could not be applied: ${error.message}`);
      if (latest) render(latest);
    } finally {
      controlBusy = false;
      if (latest) render(latest);
    }
  }

  nodes.toggle.addEventListener('click', () => {
    if (!latest) return;
    const action = normalizeStatus(latest) === 'running' ? 'pause' : 'resume';
    control({action}, action === 'pause' ? 'Simulation paused.' : 'Simulation resumed.');
  });
  nodes.restart.addEventListener('click', () => control({action:'restart'}, 'Simulation restarted.'));
  nodes.speed.addEventListener('change', () => control({action:'speed', speed:Number(nodes.speed.value)}, `Speed set to ${nodes.speed.value}×.`));
  nodes.mechanism.addEventListener('change', () => {
    const mechanism = nodes.mechanism.value === 'first_price_hybrid' ? 'first_price_hybrid' : 'uniform_auction';
    control({action:'mechanism', mechanism}, 'Mechanism changed; simulation restarted.');
  });
  window.addEventListener('pagehide', () => {
    stopped = true;
    requestVersion += 1;
    if (stateAbort) stateAbort.abort();
  });
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
      stopped = false;
      fetchState();
    }
  });
  fetchState();
})();
