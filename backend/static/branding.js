(() => {
  'use strict';
  const brands = window.SpectraScopeBrands || [];
  if (!brands.length) return;
  // User-selected product identity; preserve legacy data/installation names.
  const current = () => brands.find(brand => brand.id === 'geo-intelligence') || brands[0];
  const source = brand => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(brand.svg)}`;

  function refresh() {
    const brand = current();
    document.querySelectorAll('[data-brand-logo]').forEach(image => {
      image.src = source(brand);
      image.alt = brand.name;
    });
    document.querySelectorAll('[data-brand-slogan]').forEach(node => { node.textContent = brand.slogan; });
    document.querySelectorAll('link[rel="icon"]').forEach(icon => { icon.href = brand.id==='geo-intelligence'?'/static/branding/geo-intelligence-icon.svg':source(brand); });
    document.querySelectorAll('[data-brand-name]').forEach(node=>{node.textContent=brand.name;});
    const notice = document.getElementById('brandSelectionStatus');
    if (notice) notice.textContent = `${brand.name} is the app header and browser-tab identity · ${brand.slogan}`;
  }
  document.querySelectorAll('[data-brand-choices]').forEach(grid => {
    brands.forEach((brand, index) => {
      const card = document.createElement('article');
      card.className = 'brand-option';
      const preview = document.createElement('img');
      preview.src = source(brand);
      preview.alt = `${brand.name} logo concept`;
      preview.className = 'brand-option-icon';
      const label = document.createElement('span');
      label.className = 'brand-option-number';
      label.textContent = `0${index + 1}`;
      const title = document.createElement('h3');
      title.textContent = brand.name;
      const slogan = document.createElement('p');
      slogan.className = 'brand-option-slogan';
      slogan.textContent = brand.slogan;
      const description = document.createElement('p');
      description.textContent = brand.description;
      const actions = document.createElement('div');
      actions.className = 'brand-option-actions';
      const chosen = document.createElement('span');
      chosen.className = 'brand-option-tag';
      chosen.textContent = 'Original concept';
      const download = document.createElement('a');
      download.href = source(brand);
      download.download = `GeoIntelligence-${brand.id}.svg`;
      download.textContent = 'SVG ↓';
      actions.append(chosen, download);
      card.append(label, preview, title, slogan, description, actions);
      grid.append(card);
    });
  });
  window.SpectraScopeBranding = Object.freeze({current, refresh});
  refresh();
})();
