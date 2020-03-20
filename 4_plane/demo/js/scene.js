var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.001,
  100
);
camera.position.set(-10, 2, 1);
camera.lookAt(new THREE.Vector3(0, 0, 0));

var grid = new THREE.Mesh(
  new THREE.PlaneGeometry(25, 25, 5, 5),
  new THREE.MeshStandardMaterial({
    color: "white",
    flatShading: true,
    side: THREE.DoubleSide,
    wireframe: false
  })
);
grid.rotateX(Math.PI / 2);
grid.receiveShadow = true;
scene.add(grid);

var dl = new THREE.DirectionalLight(0xffffff, 0.5);
dl.castShadow = true;
dl.shadow.mapSize.width = 1024;
dl.shadow.mapSize.height = 1024;
scene.add(dl);

var hl = new THREE.HemisphereLight(0xffffff, 0.1);
scene.add(hl);