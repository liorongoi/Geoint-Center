"""All algorithms inherit the shared correction-to-training policy."""

PORTABLE_MODEL = 'portable_linear_lulc'
PORTABLE_DETAILS = {
    'name':'Portable linear LULC','task':'lulc','onboarded_only':True,
    'trainable':False,'training_mode':'unsupported',
    'reason':'Fixed spectral coefficients, evaluated against independent local references before release',
}

# Object masks have no LULC class taxonomy and are never classifier votes.
SEGMENTATION_DETAILS = {
    'sam3_tracker': {'name':'SAM 3 Tracker','task':'prompted object segmentation',
        'trainable':False,'training_mode':'unsupported',
        'reason':'Pretrained mask inference; a supervised SAM training adapter is not yet implemented. Human-approved masks feed LULC references.'},
}

ANOMALY_DETAILS = {
    'rx': {'name':'Regularized RX','training_mode':'reviewed_background_refit','trainable':True},
    'local_rx': {'name':'Local RX','training_mode':'reviewed_background_refit','trainable':True},
    'isolation_forest': {'name':'Isolation Forest','training_mode':'reviewed_background_refit','trainable':True},
    'autoencoder': {'name':'Spectral autoencoder','training_mode':'reviewed_background_refit','trainable':True},
}

MODEL_DETAILS = {
    'random_forest': {'name':'Random Forest', 'reason':'robust spectral baseline', 'minimum':20, 'training_mode':'refit'},
    'svm': {'name':'Support Vector Machine', 'reason':'high-dimensional spectral classification', 'minimum':15, 'training_mode':'refit'},
    'xgboost': {'name':'XGBoost', 'reason':'non-linear spectral relationships', 'minimum':30, 'training_mode':'refit'},
    'unet': {'name':'U-Net', 'reason':'spatial context and class boundaries', 'minimum':100, 'training_mode':'checkpoint'},
    'prithvi': {'name':'Prithvi-EO-2.0-300M', 'reason':'pretrained spatial features from six mapped reflectance bands', 'minimum':75, 'training_mode':'frozen_encoder_head'},
    'ensemble': {'name':'LULC Ensemble', 'reason':'equal soft voting across compatible RF, SVM, XGBoost, U-Net and Prithvi members', 'minimum':20, 'training_mode':'refit_members'},
    'dynamicworld': {'name':'Dynamic World · pretrained LULC', 'reason':'Released global nine-class pixel classifier; Sentinel-2 L1C nine-band 10m input', 'training_mode':'checkpoint','pretrained_lulc':True},
    'prithvi_crop': {'name':'Prithvi crop/LULC · pretrained head','reason':'Released 13-class temporal pixel classifier; three-date HLS stack','training_mode':'checkpoint','pretrained_lulc':True},
    'openearthmap': {'name':'OpenEarthMap · pretrained U-Net','reason':'Released eight-class pixel model; very high resolution RGB','training_mode':'checkpoint','pretrained_lulc':True},
}


# Describes this application's adapters, not measured accuracy on every sensor.
SATELLITE_SUITABILITY = {
    'dynamicworld': {
        'satellites':'Sentinel-2 MSI · Level-1C',
        'requirements':'TOA reflectance; B02, B03, B04, B05, B06, B07, B08, B11 and B12 on a 10 m analysis grid.',
        'limitations':'Not Landsat or L2A. Locally exercised on one Sentinel-2A product; this is not an accuracy guarantee for every scene.',
        'source':'https://github.com/google/dynamicworld'},
    'prithvi_crop': {
        'satellites':'Landsat and Sentinel-2 through HLS L30 / S30',
        'requirements':'Three real HLS acquisition dates, six surface-reflectance bands per date, 18 channels total at 30 m.',
        'limitations':'Raw Landsat or Sentinel products are not automatically HLS. One date repeated three times is not valid input.',
        'source':'https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification'},
    'openearthmap': {
        'satellites':'Very high resolution RGB · e.g. suitable WorldView / Pleiades products; aerial and UAV orthophotos',
        'requirements':'Georeferenced RGB with actual 0.25–0.5 m source detail and compatible radiometry.',
        'limitations':'Not Sentinel-2 or Landsat. Upsampling coarse imagery does not create the required detail. Validate transfer to your imagery.',
        'source':'https://open-earth-map.org/overview.html'},
    'prithvi': {
        'satellites':'HLS-like Sentinel-2 and Landsat 8/9 reflectance',
        'requirements':'BLUE, GREEN, RED, narrow NIR, SWIR1 and SWIR2; mapped physical reflectance and labeled classes. Sentinel-2 NIR uses B8A.',
        'limitations':'HLS 30 m is the pretrained input domain. This adapter freezes the encoder and trains a downstream head; it does not use a full hyperspectral cube.',
        'source':'https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M'},
    'ensemble': {
        'satellites':'Sentinel-2 / Landsat with Prithvi-compatible reflectance bands',
        'requirements':'The full five-member ensemble needs the six Prithvi bands, labeled classes and compatible member taxonomies/grids.',
        'limitations':'It inherits its members’ restrictions. Use individual sensor-flexible models for RGB-only or unmapped hyperspectral inputs.'},
    'sam3_tracker': {
        'satellites':'Any sensor with a mapped three-band display composite · Sentinel-2, Landsat, VHR or hyperspectral-derived RGB',
        'requirements':'Three explicitly selected display bands plus foreground points or a box prompt.',
        'limitations':'Uses three bands, not the full spectrum. Detail is limited by ground resolution. Proposes boundaries, not LULC classes; human class approval is required.',
        'source':'https://github.com/facebookresearch/sam3'},
}

for _model in ('random_forest','svm','xgboost','unet'):
    SATELLITE_SUITABILITY[_model] = {
        'satellites':'Sensor-flexible · Sentinel-2, Landsat and mapped hyperspectral GeoTIFFs such as Hyperion, EnMAP or PRISMA',
        'requirements':('Mapped bands and spatial training masks; train or adapt for the sensor, resolution and classes.' if _model=='unet' else 'Consistent band identities, units and masks; labeled examples for the sensor and class taxonomy.'),
        'limitations':'A trainable architecture, not a ready pretrained model for every satellite. Sensors without an adapter require prepared local GeoTIFFs; accuracy must be validated.'}

for _model in ANOMALY_DETAILS:
    SATELLITE_SUITABILITY[_model] = {
        'satellites':'Multispectral / hyperspectral · Sentinel-2, Landsat, Hyperion, EnMAP or PRISMA with mapped local rasters',
        'requirements':'All connected science bands on a common grid; meaningful background pixels and consistent calibration.',
        'limitations':('Spectral reconstruction model fitted to your background; not a pretrained satellite classifier.' if _model=='autoencoder' else 'Fits background statistics for your dataset. Anomaly scores do not identify a semantic class or establish detection accuracy.')}

for _registry in (MODEL_DETAILS, ANOMALY_DETAILS, SEGMENTATION_DETAILS):
    for _identifier, _details in _registry.items():
        _details['satellite_suitability'] = SATELLITE_SUITABILITY[_identifier]


def algorithm_training_rule(model_id: str) -> dict:
    """Future registered algorithms opt in by default; unsupported workers must say so."""
    if model_id == PORTABLE_MODEL:
        model=PORTABLE_DETAILS
    elif model_id not in MODEL_DETAILS:
        from object_detection import DETAILS
        model=DETAILS[model_id]
    else:model = MODEL_DETAILS[model_id]
    return {'enabled':model.get('trainable',True),
            'mode':model.get('training_mode','refit'),
            'minimum_training_pixels_per_class':model.get('minimum_training_pixels_per_class',5)}

# Detection is a separate output family; no classifier worker fallback is permitted.

