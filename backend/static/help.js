/* Local documentation UI; independent of backend startup and remote content. */
(() => {
  'use strict';
  const manual = window.SpectraScopeManual;
  const standalone = document.body.dataset.manualPage === 'true';
  const root = document.getElementById(standalone ? 'helpCenter' : 'helpDialog');
  if (!root || !manual) return;
  const chapters = manual.chapters;
  const byId = new Map(chapters.map(chapter => [chapter.id, chapter]));
  const normalized = value => String(value).normalize('NFKD').toLowerCase();
  // Only our authored manual HTML is parsed. Queries and UI labels use textContent.
  const searchable = new Map(chapters.map(chapter => {
    const holder = document.createElement('div');
    holder.innerHTML = chapter.html;
    return [chapter.id, normalized(`${chapter.title} ${chapter.summary} ${chapter.keywords} ${holder.textContent}`)];
  }));
  let active = 'start';
  let opener = null;
  root.classList.add('help-center');
  root.setAttribute('aria-labelledby', 'helpTitle');
  root.innerHTML = `
    <header class="help-header">
      <div class="help-header-identity"><img class="brand-logo" data-brand-logo src="${standalone ? 'branding/geo-intelligence-dark.svg' : '/static/branding/geo-intelligence-dark.svg'}" alt="Geo Intelligence logo"><div><span class="help-eyebrow">GEO INTELLIGENCE HELP</span><h2 id="helpTitle">${standalone ? 'User manual' : 'How can we help?'}</h2><p>From finding imagery to interpreting a result.</p></div></div>
      ${standalone ? '<a class="help-back" id="helpBackToApp" href="/workspace">← Back to workspace</a>' : '<button class="help-close" type="button" aria-label="Close help">×</button>'}
    </header>
    <div class="help-tools">
      <div class="help-search-wrap"><label for="helpSearch">Search the manual</label><div><span aria-hidden="true">⌕</span><input id="helpSearch" type="search" maxlength="200" placeholder="Try “local path”, “disagreement” or “fine-tuning”" autocomplete="off" aria-controls="helpNav"><button class="help-clear" type="button" aria-label="Clear help search" hidden>×</button></div></div>
      <div class="help-document-actions">${standalone ? '<button type="button" data-help-print>Print / save PDF</button>' : '<a href="/static/user-manual.html" target="_blank" rel="noopener noreferrer">Full manual ↗</a>'}<button type="button" data-help-download>Save offline copy ↓</button>${standalone ? '' : '<button type="button" data-help-validation>Release validation</button>'}</div>
    </div>
    <div class="help-body"><aside class="help-navigation"><p id="helpSearchStatus" role="status" aria-live="polite"></p><nav id="helpNav" aria-label="Manual chapters"></nav></aside><div id="helpArticle" class="help-reading" tabindex="-1"></div></div>
    <footer class="help-footer"><span id="helpEdition"></span><span>${standalone ? 'All chapters print together. No account required to read help.' : 'F1: open help · Esc: close · No work is submitted by opening help.'}</span></footer>`;

  const search = root.querySelector('#helpSearch');
  const nav = root.querySelector('#helpNav');
  const article = root.querySelector('#helpArticle');
  const status = root.querySelector('#helpSearchStatus');
  const clear = root.querySelector('.help-clear');
  root.querySelector('#helpEdition').textContent = manual.edition;
  if (standalone && location.protocol === 'file:') root.querySelector('#helpBackToApp').href = 'http://localhost:8080/workspace';

  function results() {
    const words = normalized(search.value.trim()).split(/\s+/).filter(Boolean);
    return chapters.filter(chapter => words.every(word => searchable.get(chapter.id).includes(word)))
      .sort((a, b) => Number(normalized(b.title).includes(normalized(search.value.trim()))) - Number(normalized(a.title).includes(normalized(search.value.trim()))));
  }

  function chapterElement(chapter) {
    const element = document.createElement('article');
    element.className = 'manual-article';
    element.id = chapter.id;
    const category = document.createElement('div');
    category.className = 'help-eyebrow';
    category.textContent = chapter.group;
    const title = document.createElement('h2');
    title.textContent = chapter.title;
    title.id = `guide-title-${chapter.id}`;
    title.tabIndex = -1;
    element.setAttribute('aria-labelledby', title.id);
    const summary = document.createElement('p');
    summary.className = 'help-lede';
    summary.textContent = chapter.summary;
    const body = document.createElement('div');
    body.className = 'manual-content';
    body.innerHTML = chapter.html;
    element.append(category, title, summary, body);
    if (!standalone) {
      const position = chapters.indexOf(chapter);
      const paging = document.createElement('nav');
      paging.className = 'help-paging';
      paging.setAttribute('aria-label', 'Adjacent chapters');
      for (const [index, label] of [[position - 1, '← Previous'], [position + 1, 'Next →']]) {
        if (!chapters[index]) continue;
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.helpTopic = chapters[index].id;
        button.textContent = `${label} · ${chapters[index].title}`;
        paging.append(button);
      }
      element.append(paging);
    }
    return element;
  }

  function renderNavigation(matches) {
    nav.replaceChildren();
    let group = '';
    for (const chapter of matches) {
      if (chapter.group !== group) {
        const label = document.createElement('h3');
        label.textContent = chapter.group;
        nav.append(label);
        group = chapter.group;
      }
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.helpTopic = chapter.id;
      button.className = 'help-topic';
      if (chapter.id === active) button.setAttribute('aria-current', 'page');
      const title = document.createElement('span');
      title.textContent = chapter.title;
      const hint = document.createElement('small');
      hint.textContent = chapter.summary;
      button.append(title, hint);
      nav.append(button);
    }
    const isQuery = Boolean(search.value.trim());
    status.textContent = isQuery ? `${matches.length} matching chapter${matches.length === 1 ? '' : 's'}` : `${chapters.length} chapters · choose your next step`;
    clear.hidden = !isQuery;
  }

  function render(focus = false) {
    root.classList.remove('is-validating');
    const validationToggle = root.querySelector('[data-help-validation]');
    if (validationToggle) validationToggle.textContent = 'Release validation';
    const matches = results();
    renderNavigation(matches);
    if (standalone) {
      const included = new Set(matches.map(chapter => chapter.id));
      article.querySelectorAll('.manual-article').forEach(element => { element.hidden = !included.has(element.id); });
    } else {
      article.replaceChildren();
      if (matches.length) {
        if (!matches.some(chapter => chapter.id === active)) active = matches[0].id;
        article.append(chapterElement(byId.get(active)));
        nav.querySelectorAll('[data-help-topic]').forEach(button => {
          if (button.dataset.helpTopic === active) button.setAttribute('aria-current', 'page');
          else button.removeAttribute('aria-current');
        });
        article.scrollTop = 0;
        if (focus) article.querySelector('h2').focus({preventScroll: true});
      }
    }
    let empty = article.querySelector('.help-no-results');
    if (!matches.length && !empty) {
      empty = document.createElement('div');
      empty.className = 'help-no-results';
      const title = document.createElement('h2');
      title.textContent = 'No matching chapters';
      const tip = document.createElement('p');
      tip.textContent = 'Try a shorter phrase such as “bands”, “training” or “local path”. Search runs locally and does not change your project.';
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.helpClear = 'true';
      button.textContent = 'Show all chapters';
      empty.append(title, tip, button);
      article.append(empty);
    } else if (matches.length && empty) empty.remove();
  }

  function chooseTopic(id, focus = false) {
    if (!byId.has(id)) id = 'start';
    active = id;
    search.value = '';
    render(focus);
    if (standalone) {
      document.getElementById(id)?.scrollIntoView({behavior: 'auto', block: 'start'});
      if (focus) document.getElementById(`guide-title-${id}`)?.focus({preventScroll: true});
    }
  }

  function contextTopic() {
    const visible = id => {
      const node = document.getElementById(id);
      return node && !node.closest('.hidden') && node.getClientRects().length;
    };
    if (document.getElementById('insightStudio')?.open) return 'insights';
    if (visible('anomalyInspector')) return 'anomalies';
    if (visible('manualReviewForm')) return 'review';
    if (visible('segmentationPanel')) return 'segmentation';
    if (visible('missionDetail')) {
      const tab = document.querySelector('.mission-tab.active')?.dataset.missionTab;
      return {plan: 'running', rasters: 'rasters', reports: 'reports', anomalies: 'anomalies', training: 'training'}[tab] || 'missions';
    }
    if (visible('missionHub')) return 'missions';
    return 'start';
  }

  function openHelp(id) {
    if (standalone) return chooseTopic(id);
    if (root.open) {
      if (id) chooseTopic(id);
      search.focus();
      return;
    }
    opener = document.activeElement;
    chooseTopic(id || contextTopic());
    if (!root.open) root.showModal();
    search.focus();
  }

  function saveOfflineCopy() {
    const styles = `body{font:16px/1.7 system-ui,sans-serif;color:#1b3035;max-width:980px;margin:40px auto;padding:0 24px}h1,h2,h3{line-height:1.25;color:#153f3b}h1{font-size:40px}h2{margin-top:36px;font-size:28px}h3{margin-top:28px}a{color:#096658}nav{padding:20px;background:#eef6f3;border-radius:12px}nav li{margin:6px 0}article{border-top:1px solid #c6dad4;padding-top:24px;margin-top:40px}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}td,th{padding:12px;border:1px solid #c6dad4;text-align:left;vertical-align:top}th,.guide-callout{background:#eef6f3}.guide-callout{padding:16px;border-left:4px solid #208a71;margin:20px 0}.guide-callout p{margin:8px 0 0}.guide-flow{display:flex;flex-wrap:wrap;gap:10px}.guide-flow span{padding:8px 12px;background:#eef6f3;border-radius:6px}code,kbd{font-size:.88em;background:#edf2f2;padding:2px 5px;border-radius:4px}dt{font-weight:700;margin-top:14px}dd{margin-left:0}summary{font-weight:600;cursor:pointer}article{scroll-margin-top:20px}@media print{body{margin:0;max-width:none;font-size:11pt}nav{display:none}article{break-before:page}h2,h3{break-after:avoid}tr{break-inside:avoid}a{color:inherit;text-decoration:none}details>p{display:block!important}}`;
    const titles = chapters.map(chapter => `<li><a href="#${chapter.id}">${chapter.title}</a></li>`).join('');
    // Titles, IDs and HTML below are exclusively the authored manual, not user input.
    const content = chapters.map(chapter => {
      const body = chapter.html.replace(/<details><summary>(.*?)<\/summary>/g, '<section><h3>$1</h3>').replace(/<\/details>/g, '</section>');
      return `<article id="${chapter.id}"><p>${chapter.group}</p><h2>${chapter.title}</h2><p>${chapter.summary}</p>${body}</article>`;
    }).join('');
    const logo = window.SpectraScopeBranding?.current().svg || '';
    const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Geo Intelligence user manual</title><style>${styles}header>svg{width:76px;height:76px}</style></head><body><header>${logo}<p>GEO INTELLIGENCE HELP</p><h1>User manual</h1><p>${manual.edition}</p><p>A complete offline copy. Use your browser’s Print command to print or save as PDF.</p></header><nav aria-label="Contents"><h2>Contents</h2><ol>${titles}</ol></nav>${content}</body></html>`;
    const url = URL.createObjectURL(new Blob([html], {type: 'text/html;charset=utf-8'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'Geo_Intelligence_User_Manual.html';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  function showValidation() {
    if (!window.GeoValidation) return;
    root.classList.add('is-validating');
    root.querySelector('[data-help-validation]').textContent = 'Back to manual';
    article.replaceChildren();
    const host = document.createElement('section');
    host.className = 'manual-article validation-checklist';
    article.append(host);
    window.GeoValidation(host);
    host.querySelector('h2').focus();
    article.scrollTop = 0;
    nav.querySelectorAll('[aria-current]').forEach(button => button.removeAttribute('aria-current'));
  }

  search.addEventListener('input', () => render());
  clear.addEventListener('click', () => { search.value = ''; render(); search.focus(); });
  root.addEventListener('click', event => {
    const button = event.target.closest('[data-help-topic]');
    if (button) { event.preventDefault(); chooseTopic(button.dataset.helpTopic, true); }
    if (event.target.closest('[data-help-clear]')) { search.value = ''; render(); search.focus(); }
    if (event.target.closest('[data-help-validation]')) {
      if (root.classList.contains('is-validating')) { render(); search.focus(); }
      else showValidation();
    }
    if (event.target.closest('[data-help-download]')) saveOfflineCopy();
    if (event.target.closest('[data-help-print]')) window.print();
    if (event.target.closest('.help-close')) root.close();
  });

  if (standalone) {
    chapters.forEach(chapter => article.append(chapterElement(chapter)));
    // Expose details for a complete printed manual, including the class lists.
    article.querySelectorAll('details').forEach(details => { details.open = true; });
    let printDetails = [];
    let printArticles = [];
    window.addEventListener('beforeprint', () => {
      printDetails = [...article.querySelectorAll('details')].map(details => [details, details.open]);
      printArticles = [...article.querySelectorAll('.manual-article')].map(section => [section, section.hidden]);
      printDetails.forEach(([details]) => { details.open = true; });
      printArticles.forEach(([section]) => { section.hidden = false; });
    });
    window.addEventListener('afterprint', () => {
      printDetails.forEach(([details, wasOpen]) => { details.open = wasOpen; });
      printArticles.forEach(([section, wasHidden]) => { section.hidden = wasHidden; });
      printDetails = [];
      printArticles = [];
    });
    render();
    const initial = location.hash.slice(1);
    if (byId.has(initial)) chooseTopic(initial);
  } else {
    document.getElementById('helpButton')?.addEventListener('click', () => openHelp());
    root.addEventListener('close', () => {
      if (opener?.isConnected) opener.focus({preventScroll: true});
    });
    root.addEventListener('click', event => {
      if (event.target !== root) return;
      const bounds = root.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) root.close();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'F1') { event.preventDefault(); openHelp(); }
    });
    document.addEventListener('click', event => {
      const button = event.target.closest('.context-help[data-help-topic]');
      if (!button || root.contains(button)) return;
      event.preventDefault();
      openHelp(button.dataset.helpTopic);
    });

    const contexts = [
      ['.sidebar-heading h1', 'start'], ['.aoi-section h2', 'aoi'],
      ['#missionHub h1', 'missions'], ['#detailMissionName', 'missions'],
      ['.hierarchy-card h2', 'missions'], ['.algorithm-card h2', 'models'],
      ['.inference-card h2', 'running'], ['#rasterPanel h2', 'rasters'],
      ['#reportsPanel h2', 'reports'], ['.training-heading h2', 'ground-truth'],
      ['.vector-import-card h3', 'vectors'], ['#automaticTrainingPolicy h3', 'training'],
      ['.band-tool h3', 'alignment'], ['#segmentationPanel h3', 'segmentation'],
      ['#anomalyPanel .anomaly-heading h2', 'anomalies'],
      ['#anomalyInspector header h2', 'anomalies'], ['#analysisInspector header h2', 'review'],
      ['#manualReviewForm h3', 'review'], ['#pixelVotePanel h3', 'review']
    ];
    function attachContextHelp() {
      const all = contexts.map(([selector, topic]) => [document.querySelector(selector), topic]);
      for (const [id, topic] of [['catalog', 'search'], ['dateFrom', 'search'], ['uploadButton', 'rasters']]) {
        const section = document.getElementById(id)?.closest('.panel-section, .upload-card');
        all.push([section?.querySelector('h2, .upload-copy strong'), topic]);
      }
      for (const [heading, topic] of all) {
        if (!heading || heading.querySelector('.context-help')) continue;
        const button = document.createElement('button');
        button.className = 'context-help';
        button.type = 'button';
        button.dataset.helpTopic = topic;
        button.textContent = '?';
        button.title = `Help: ${byId.get(topic).title}`;
        button.setAttribute('aria-label', button.title);
        heading.append(button);
      }
    }
    let scheduled = false;
    const observer = new MutationObserver(records => {
      if (scheduled || records.every(record => root.contains(record.target))) return;
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; attachContextHelp(); });
    });
    observer.observe(document.querySelector('.shell') || document.body, {childList: true, subtree: true});
    attachContextHelp();
    render();
    if (new URLSearchParams(location.search).get('validation') === '1') { openHelp(); showValidation(); }
  }
})();
