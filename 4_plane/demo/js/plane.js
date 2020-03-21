var Plane = function(data) {
  this.data = data;
  THREE.Group.apply(this, arguments);
  this.rotationSpeed = Math.random() * 0.02 + 0.005;
  this.rotationPosition = Math.random();

  this.geom = new THREE.Geometry();

  var v1 = new THREE.Vector3(0, 0, 0);
  var v2 = new THREE.Vector3(0, 0, 0);
  var v3 = new THREE.Vector3(0, 0, 0);
  var v4 = new THREE.Vector3(0, 0, 0);
  var v5 = new THREE.Vector3(0, 0, 0);
  var v6 = new THREE.Vector3(0, 0, 0);

  this.geom.vertices.push(v1);
  this.geom.vertices.push(v2);
  this.geom.vertices.push(v3);
  this.geom.vertices.push(v4);
  this.geom.vertices.push(v5);
  this.geom.vertices.push(v6);

  this.geom.faces.push(new THREE.Face3(4, 5, 2));
  this.geom.faces.push(new THREE.Face3(0, 1, 2));
  this.geom.faces.push(new THREE.Face3(2, 3, 4));
  this.geom.faces.push(new THREE.Face3(1, 3, 2));

  // The main bauble is an Octahedron
  this.paper = new THREE.Mesh(
    this.geom,
    new THREE.MeshStandardMaterial({
      color: "#47689b",
      flatShading: true,
      side: THREE.DoubleSide,
      wireframe: false
    })
  );
  this.castShadow = true;
  this.receiveShadow = true;
  this.add(this.paper);

  this.reset = function() {
      this.frame = 0;
    this.cache = new Array()
    var sl = data.sampleLength;
    this.cache.push = function (){
        if (this.length >= sl) {
            this.shift();
        }
        return Array.prototype.push.apply(this,arguments);
    }
    var pts = [
      0.0,
      0.053722500801086426,
      -0.10000000149011612,
      0.0,
      0.053722500801086426,
      -0.008631999604403973,
      -0.29947200417518616,
      0.053722500801086426,
      0.0,
      0.2003210037946701,
      this.data.startHeight,
      0.0,
      0.0,
      0.053722500801086426,
      0.008631999604403973,
      0.0,
      0.053722500801086426,
      0.10000000149011612
    ];
    this.setPts(pts)
  };

  this.setPts = function(pts) {
    this.frame += 1;
    for (i = 0; i < 6; i++) {
      if (i == 3) {
        continue;
      }
      pts[i * 3] += pts[3 * 3];
      pts[i * 3 + 1] += pts[3 * 3 + 1];
      pts[i * 3 + 2] += pts[3 * 3 + 2];
    }
    var i;
    for (i = 0; i < 6; i++) {
      this.geom.vertices[i].set(pts[3 * i + 0], pts[3 * i + 1], pts[3 * i + 2]);
    }
    this.geom.verticesNeedUpdate = true;
  };

  this.getPts = function() {
    var pts = [];

    for (i = 0; i < 6; i++) {
      pts.push(this.geom.vertices[i].x);
      pts.push(this.geom.vertices[i].y);
      pts.push(this.geom.vertices[i].z);
    }

    for (i = 0; i < 6; i++) {
      if (i == 3) {
        continue;
      }
      pts[i * 3] -= pts[3 * 3];
      pts[i * 3 + 1] -= pts[3 * 3 + 1];
      pts[i * 3 + 2] -= pts[3 * 3 + 2];
    }
    var is_wind = (this.data.windStart < this.frame) && (this.frame < this.data.windStart + this.data.windLength);
    
    this.cache.push([...pts, is_wind?1:0]);
    return pts;
  };
};

Plane.prototype = Object.create(THREE.Group.prototype);
Plane.prototype.constructor = Plane;
