var simData = function() {
    this.playNextFrame = false;
  this.startHeight = 1.5;
  this.windStart = 13;
  this.windLength = 36;
  this.sampleLength = 5;
  this.apply = function() {
    load_model(this.sampleLength);
    this.plane.reset();
  };
  this.pause = false;
  this.nextFrame = function() {
      this.playNextFrame = true;
  };
};
