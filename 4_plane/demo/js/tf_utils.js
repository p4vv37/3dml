async function load_model(sampleSize) {
  window.model = null;
  model = await tf.loadLayersModel(
    "http://0.0.0.0:8003/models/model_multiple_points_" + sampleSize + "_js/model.json"
  );
  window.model = model;
}
