/* Explicit local downloads; no mission required and no automatic map windows. */
(() => {
  const esc = escapeHtml;
  const isCopernicus = image => ['sentinel-2', 'sentinel-2-l1c'].includes(image.catalog);
  const isHLS = image => ['hls-l30', 'hls-s30'].includes(image.catalog);
  const safeId = /^S2[ABCD]_MSI(L1C|L2A)_\d{8}T\d{6}_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_\d{8}T\d{6}$/;
  const hlsId = /^HLS\.(L30|S30)\.T\d{2}[A-Z]{3}\.\d{7}T\d{6}\.v2\.0$/;

  function request(path, body, token = '') {
    if (!body) return api('/api/catalog-downloads' + path);
    const headers = {'Content-Type': 'application/json'};
    if (token) headers['X-Earthdata-Token'] = token;
    return api('/api/catalog-downloads' + path, {method: 'POST', headers, body: JSON.stringify(body)});
  }

  async function open(savedSelection) {
    const selected = savedSelection || selectedCatalogImages().filter(image => image.catalog !== 'local');
    const direct = selected.filter(image => !isCopernicus(image));
    const copernicus = selected.filter(isCopernicus);
    const d = document.createElement('dialog');
    d.className = 'wf wf-dialog';
    d.setAttribute('aria-label', 'Download online imagery');
    d.innerHTML = `<header class="wf-head"><h2>Download online imagery</h2><button data-close>Close</button></header>
      <p>Completed downloads appear in Local imagery with the original bands and checksums.</p>
      <label data-destination>Local destination folder<input data-directory placeholder="Use the app downloads folder"></label>
      <p data-default></p><p data-selected></p>
      <section data-copernicus hidden>
        <p><b>Sign in to Copernicus</b> to save and import the original Sentinel‑2 SAFE packages.</p>
        <label>Copernicus username<input data-cdse-user autocomplete="off" maxlength="256"></label>
        <label>Copernicus password<input type="password" data-cdse-password autocomplete="off" maxlength="1024"></label>
        <label>Two-factor code (if enabled)<input type="password" inputmode="numeric" data-cdse-totp autocomplete="off" maxlength="8"></label>
        <button class="primary" data-cdse-start>Sign in and download Sentinel‑2</button>
        <p>The app uses your password once to sign in. The download session is kept in memory.</p>
      </section>
      <section data-earthdata hidden>
        <label>NASA Earthdata user token<input type="password" data-token autocomplete="off" spellcheck="false" maxlength="2048"></label>
        <p><a href="https://urs.earthdata.nasa.gov/profile" target="_blank" rel="noopener noreferrer">Open NASA Earthdata profile</a>
          → Generate Token. Enter the token here. It is used for these downloads and is never saved.</p>
      </section>
      <button class="primary" data-start>Download selected online images</button>
      <button class="primary" data-build hidden>Build 3-date HLS bundle</button>
      <p data-message role="status"></p><div data-tasks></div>`;
    document.body.append(d);
    d.showModal();

    const q = selector => d.querySelector(selector);
    const message = text => { q('[data-message]').textContent = text; };
    const token = needed => {
      if (!needed) return '';
      const field = q('[data-token]');
      if (!field.value.trim()) {
        q('[data-earthdata]').hidden = false;
        field.focus();
        throw new Error('Enter your NASA Earthdata user token to download these files.');
      }
      return field.value.trim();
    };
    q('[data-selected]').textContent = selected.length
      ? selected.length + ' online images selected: ' + selected.map(image => image.name || image.id).join(', ')
      : 'No online images selected. You can resume earlier downloads below.';
    q('[data-copernicus]').hidden = !copernicus.length;
    q('[data-start]').hidden = !direct.length;
    q('[data-destination]').hidden = !selected.length;
    q('[data-default]').hidden = !selected.length;
    if (copernicus.length && direct.length) q('[data-start]').textContent = 'Download ' + direct.length + ' other selected images';
    const threeHLS = selected.length === 3 && selected.every(isHLS);
    let timer, busy = false, localIds = [];
    const queued = new Map();
    let recent = [], resumeCopernicus = null;
    const copernicusButton = q('[data-cdse-start]');
    copernicusButton.onclick = async () => {
      const username = q('[data-cdse-user]').value.trim();
      const password = q('[data-cdse-password]').value;
      const totp = q('[data-cdse-totp]').value.trim();
      if (!username || !password) return message('Enter your Copernicus username and password.');
      q('[data-cdse-password]').value = '';
      q('[data-cdse-totp]').value = '';
      copernicusButton.disabled = true;
      try {
        const batch = resumeCopernicus ? {resume_id: resumeCopernicus}
          : {products: copernicus.map(image => ({catalog: image.catalog, id: image.id})),
             directory: q('[data-directory]').value.trim()};
        await request('/copernicus', {...batch, username, password, totp});
        resumeCopernicus = null;
        copernicusButton.textContent = 'Sign in and download Sentinel‑2';
        q('[data-copernicus]').hidden = true;
        message('Copernicus downloads queued. Verified SAFE products will appear in Local imagery.');
        await update();
      } catch (error) { message(error.message); }
      finally { copernicusButton.disabled = false; }
    };

    async function update() {
      if (!d.open || busy) return;
      busy = true;
      try {
        const value = await request('');
        if (!d.open) return;
        q('[data-default]').textContent = 'Default: ' + value.default_directory;
        q('[data-earthdata]').hidden = !(selected.some(isHLS) || value.items.some(task =>
          isHLS(task.image) && ['failed', 'interrupted', 'stopped'].includes(task.status)));
        recent = value.items;
        if (threeHLS) {
          const completed = selected.map(image => value.items.find(task =>
            task.id === queued.get(image.id) || (!queued.has(image.id) && task.image.id === image.id && task.status === 'complete')));
          localIds = completed.every(task => task?.status === 'complete' && task.local_images?.length === 1)
            ? completed.map(task => task.local_images[0]) : [];
          q('[data-build]').hidden = localIds.length !== 3;
          if (localIds.length === 3) q('[data-start]').disabled = true;
        }
        q('[data-tasks]').innerHTML = value.items.map(task => {
          const active = ['running', 'queued'].includes(task.status);
          const action = active ? 'stop' : ['failed', 'interrupted', 'stopped'].includes(task.status) ? 'resume' : '';
          return `<article class="wf-card"><b>${esc(task.image.name || task.image.id)}</b>
            <p>${esc(task.status)} · ${esc(task.stage)}</p>
            <progress class="wf-progress" max="${Math.max(1, task.files.length)}" value="${task.completed_files || 0}"></progress>
            <small>${task.completed_files || 0}/${task.files.length} ${isCopernicus(task.image) ? 'ZIP archives' : 'bands'} · ${(task.bytes / 1048576).toFixed(1)} MB received</small>
            <p>${esc(task.folder)}</p>
            ${action ? `<button data-task="${esc(task.id)}" data-action="${action}">${active ? 'Stop' : 'Resume'}</button>` : ''}
            ${task.issues?.length ? '<p>Some band grouping details need review in Local imagery.</p>' : ''}</article>`;
        }).join('') || '<p>No downloads yet.</p>';
        q('[data-tasks]').querySelectorAll('[data-task]').forEach(button => button.onclick = async () => {
          const task = value.items.find(item => item.id === button.dataset.task);
          if (isCopernicus(task.image) && button.dataset.action === 'resume') {
            resumeCopernicus = task.id;
            q('[data-copernicus]').hidden = false;
            copernicusButton.textContent = 'Sign in and resume Sentinel‑2';
            q('[data-cdse-user]').focus();
            return;
          }
          try {
            await request('/' + task.id + '/' + button.dataset.action, {}, token(isHLS(task.image) && button.dataset.action === 'resume'));
            q('[data-token]').value = '';
            await update();
          } catch (error) { message(error.message); }
        });
      } catch (error) { message(error.message); }
      finally { busy = false; }
    }

    q('[data-start]').onclick = async () => {
      q('[data-start]').disabled = true;
      try {
        const credential = token(selected.some(isHLS));
        for (const image of direct) {
          if (queued.has(image.id)) continue;
          const previous = isHLS(image) && !q('[data-directory]').value.trim() &&
            recent.find(task => task.image.id === image.id && task.image.catalog === image.catalog);
          let task;
          if (previous && ['failed', 'interrupted', 'stopped'].includes(previous.status))
            task = await request('/' + previous.id + '/resume', {}, credential);
          else if (previous) task = previous;
          else task = await request('', {image, directory: q('[data-directory]').value.trim()}, isHLS(image) ? credential : '');
          queued.set(image.id, task.id);
        }
        q('[data-token]').value = '';
        message('Downloads queued. Progress is saved and downloads continue while the app is running.');
        await update();
      } catch (error) {
        message(error.message);
        q('[data-start]').disabled = false;
      }
    };

    q('[data-build]').onclick = async () => {
      const button = q('[data-build]');
      button.disabled = true;
      try {
        const result = await api('/api/local-imagery/hls-bundles', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({image_ids: localIds})
        });
        const library = await api('/api/local-imagery');
        updateLocalCatalogItems(library.items, true);
        // Make the bundle the selected product for the next mission action.
        saveCatalogTab();
        for (const image of selected) state.catalogTabs.online.selected.delete(image.id);
        state.catalogTabs.local.selected.add(result.image.id);
        selectCatalogTab('local');
        d.close();
        toast('HLS bundle created and selected for Prithvi Crop/LULC.');
      } catch (error) { message(error.message); }
      finally { button.disabled = false; }
    };

    q('[data-close]').onclick = () => d.close();
    d.addEventListener('close', () => {
      clearInterval(timer);
      q('[data-token]').value = '';
      q('[data-cdse-password]').value = '';
      q('[data-cdse-totp]').value = '';
      d.remove();
    }, {once: true});
    await update();
    if (d.open) timer = setInterval(update, 1500);
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('catalogDownloadsButton').onclick = () => open();
    const params = new URLSearchParams(location.hash.slice(1));
    const safe = params.get('sentinel-download')?.split(',') || [];
    if (safe.length === 3 && new Set(safe).size === 3 && safe.every(id => safeId.test(id))) {
      open(safe.map(id => ({id, catalog: id.match(safeId)[1] === 'L1C' ? 'sentinel-2-l1c' : 'sentinel-2'})));
      return;
    }
    const ids = params.get('hls-download')?.split(',') || [];
    if (ids.length === 3 && new Set(ids).size === 3 && ids.every(id => hlsId.test(id))) {
      open(ids.map(id => ({id, catalog: 'hls-' + id.match(hlsId)[1].toLowerCase()})));
    }
  });
  window.CatalogDownloads = {open};
})();
