async function load_model(name) {
  window.model = null;
  console.log(name);
  model = await tf.loadLayersModel("models/" + name + "_js/model.json"
  );
  window.model = model;
}
