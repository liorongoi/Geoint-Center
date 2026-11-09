/* Operator evidence is stored in this browser. The requirements register is reviewed separately. */
(() => {
  'use strict';
  const storageKey = 'geo-intelligence-validation-v1';
  const checks = [
    {id:'DEPLOY-01', name:'Windows Web · current-user installation', target:'Windows x64 with a standard user account',
      steps:['Use a disposable account or a clean test machine. Disconnect the internet and use the existing Windows Web prerequisites media, then its matching application installer.','Open the Windows Web shortcut. Verify that Chrome or the default browser opens the workspace without an administrator prompt. Close and reopen it once to check that setup does not repeat.','Import a local image, select the bundled map, then open a saved or new Detection experiment. If online access is permitted, reconnect and confirm that street and satellite maps and online catalogs become selectable while Local imagery stays available.']},
    {id:'DEPLOY-02', name:'Windows offline media and launch receipt', target:'Windows x64 with a standard user account',
      steps:['Copy the complete, previously supplied media from USB, including its adjacent payload folder. Confirm that your workstation can read the files and that setup verification finishes.','With internet disconnected, use the existing Windows Desktop media on a disposable account or machine. If you are only testing Windows Web media, record that edition in the result.','Launch the installed edition twice. Confirm that first-launch setup finishes and reopening uses the saved verification receipt. Open the bundled map and the app log.']},
    {id:'DEPLOY-04', name:'Windows update and workspace recovery', target:'Disposable Windows x64 test workspace with matching update media',
      steps:['When installer testing resumes, use a disposable Windows workspace with a mission, a correction and an exported result. Record their identities and file hashes before the update.','Run the current-user update with the approved failure-injection fixture. After recovery, verify the prior version launches and the saved workspace and results match the originals.','Retry with the normal update. Verify the new version and the same saved records and results. If matching media or the failure fixture is unavailable, record Blocked.']},
    {id:'BRAND-01', name:'Installed product identity', target:'An existing Windows or macOS desktop package',
      steps:['In an already installed test copy, compare the setup screen, app icon and window title against the approved Geo Intelligence globe mark. Record the platform and package version.','Open the entry page. Check the Geo Intelligence heading, “From imagery to intelligence.” and the Yanogal footer.']},
    {id:'ARCH-02', name:'Distributed inference on two machines', target:'Two physical machines on the approved internal network',
      steps:['Open Services & workers → Worker pool on the coordinator. Enroll each physical worker separately using its private enrollment and verify that both identities appear.','In Manual inference, choose a qualified supported model. Queue two independent images or jobs with worker routing enabled.','Check each completed run for the actual worker and device, including CUDA when testing a supported Windows GPU. Open both results and export them. Record the worker devices and run IDs.']},
    {id:'ARCH-04', name:'Distributed U-Net training and recovery', target:'Two enrolled physical CPU or GPU machines and a disposable mission',
      steps:['Prepare fitting and separate validation references. In Training & qualification, choose U-Net with two synchronized workers.','Verify both workers contribute to a round. In the disposable test environment, restart the coordinator after a checkpoint is saved and confirm that training resumes without repeating the committed round.','Review the held-out metrics. If the candidate qualifies, release it, run inference and export one result. Record the worker devices and the recovered round.']},
    {id:'POLICY-02', name:'Connected network boundary', target:'An approved test deployment with network logging and disposable data',
      steps:['Use the updated audit candidate and synthetic data. Record the app version, target OS and which services are on a separate host.','Run a public catalog search, load a basemap, and send a synthetic chat request to the managed local model. Check the deployment’s network log for approved destinations and verify that the model sends no traffic outside the system.','Try an unapproved test relay from the private model settings. Verify the app refuses it before sending the prompt or credentials. Verify enrolled or independently administered hosts with the network owner. Record only destinations, pass/fail evidence and any blocked checks.']}
  ];
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  window.GeoValidation = function renderValidation(host) {
    let records = {};
    let readError = '';
    try { records = JSON.parse(localStorage.getItem(storageKey) || '{}'); }
    catch { readError = 'Saved results could not be read. Export any available evidence before recording a new result.'; }
    const active = checks.find(check => records?.[check.id]?.outcome !== 'passed') || checks[0];
    function show(check, focus = false) {
      const record = records?.[check.id];
      host.replaceChildren();
      const title = element('h2','Release validation');
      title.tabIndex = -1;
      const remaining = checks.filter(item => records?.[item.id]?.outcome !== 'passed').length;
      host.append(title,element('p',remaining ? `${remaining} of ${checks.length} checks awaiting a recorded pass.` : 'A pass is recorded for every check. Export the results for review.','help-lede'));
      const picker = element('label','Check');
      const select = document.createElement('select');
      select.id = 'validation-check';
      picker.htmlFor = select.id;
      for (const item of checks) {
        const saved = records?.[item.id];
        const option = element('option',item.name + (saved?.outcome === 'passed' ? ' ✓' : ''));
        option.value = item.id;
        select.append(option);
      }
      select.value = check.id;
      select.addEventListener('change',() => show(checks.find(item => item.id === select.value)));
      host.append(picker,select,element('h3',check.name),element('p',check.target));
      const steps = element('ol');
      for (const step of check.steps) steps.append(element('li',step));
      host.append(steps);
      const form = element('form',null,'validation-form');
      const versionLabel = element('label','Version and test machine');
      const version = document.createElement('input');
      version.id = 'validation-version';
      versionLabel.htmlFor = version.id;
      version.maxLength = 180;
      version.placeholder = 'For example, 0.2.25 / Windows 11 test PC';
      version.value = record?.version || '';
      version.required = true;
      const outcomeLabel = element('label','Result');
      const outcome = document.createElement('select');
      outcome.id = 'validation-outcome';
      outcomeLabel.htmlFor = outcome.id;
      for (const [value, label] of [['','Choose a result'],['passed','Passed'],['failed','Failed'],['blocked','Blocked']]) {
        const option = element('option',label);
        option.value = value;
        outcome.append(option);
      }
      outcome.value = record?.outcome || '';
      outcome.required = true;
      const notesLabel = element('label','Evidence or error');
      const notes = document.createElement('textarea');
      notes.id = 'validation-notes';
      notesLabel.htmlFor = notes.id;
      notes.maxLength = 2000;
      notes.value = record?.notes || '';
      notes.rows = 2;
      notes.required = true;
      const actions = element('div',null,'help-document-actions');
      const save = element('button','Save result');
      save.type = 'submit';
      const next = element('button','Next waiting check');
      next.type = 'button';
      const position = checks.indexOf(check);
      const nextCheck = [...checks.slice(position + 1), ...checks.slice(0, position)].find(item => records?.[item.id]?.outcome !== 'passed');
      next.disabled = !nextCheck;
      next.addEventListener('click',() => show(nextCheck,true));
      const download = element('button','Export results');
      download.type = 'button';
      download.addEventListener('click',() => {
        const data = checks.map(item => ({requirement:item.id, ...(records?.[item.id] || {outcome:'waiting'})}));
        const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
        const link = element('a');
        link.href = url;
        link.download = 'Geo-Intelligence-validation.json';
        link.click();
        setTimeout(() => URL.revokeObjectURL(url),30000);
      });
      actions.append(save,next,download);
      const status = element('p',readError || (record ? 'Saved in this browser.' : 'Waiting for validation. Results are saved in this browser.'));
      status.setAttribute('role','status');
      form.append(versionLabel,version,outcomeLabel,outcome,notesLabel,notes,actions,status);
      form.addEventListener('submit',event => {
        event.preventDefault();
        if (!version.value.trim() || !notes.value.trim()) {
          status.textContent = 'Enter the test version and a short description of what you observed.';
          return;
        }
        const updated = {...records,[check.id]:{version:version.value.trim(),outcome:outcome.value,notes:notes.value.trim(),recorded_at:new Date().toISOString()}};
        try { localStorage.setItem(storageKey,JSON.stringify(updated)); }
        catch { status.textContent = 'This browser could not save the result.'; return; }
        records = updated;
        readError = '';
        show(check,true);
      });
      host.append(form);
      if (focus) {
        host.parentElement.scrollTop = 0;
        title.focus({preventScroll:true});
      }
    }
    show(active);
  };
})();
