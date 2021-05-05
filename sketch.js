let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "";

function setup() {
  let div = createCanvas(640, 480);
  div.style('background-color', 'gray');
  div.center('horizontal');
  div.position(width/2 , height/10);
  video = createCapture(VIDEO);
  video.hide();

//    let poseOptions = {
//  imageScaleFactor: 0.3,
//  outputStride: 16,
//  flipHorizontal: false,
//  minConfidence: 0.75,
//  maxPoseDetections: 5,
//  scoreThreshold: 0.75,
//  nmsRadius: 20,
//  detectionType: 'single',
//  multiplier: 0.75,
// }

  poseNet = ml5.poseNet(video,modelLoaded);
  poseNet.on('pose', gotPoses);

 

  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  const modelInfo = {
    model: 'model2/model.json',
    metadata: 'model2/model_meta.json',
    weights: 'model2/model.weights.bin',
  };
  brain.load(modelInfo, brainLoaded);
}

function brainLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}
var tag;
function gotResult(error, results) {
  
  if (results[0].confidence > 0.75) {
    tag = results[0].confidence;
    poseLabel = results[0].label.toUpperCase();
  }
  //console.log(results[0].confidence);
  classifyPose();
}


function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}


function modelLoaded() {
  console.log('poseNet ready');
  // poseNet.multiPose(video);
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
      // console.log(pose.rightEar.x);
      fill(255,255,255, 0);
      stroke(255);
      var wheelx = (pose.rightEar.x + pose.leftEar.x)/2;
      var wheely = (pose.rightEar.y + pose.leftEar.y)/2;
      ellipse(wheelx, wheely, pose.rightEar.x - pose.leftEar.x, 3*(pose.leftEye.x - pose.rightEye.x));
      line(pose.rightEar.x, pose.rightEar.y, pose.leftEar.x, pose.leftEar.y);
    
  fill(255,255,25);
  noStroke();
  textSize(100);
  textAlign(CENTER, CENTER);
  text(poseLabel, pose.nose.x , pose.nose.y + pose.rightEar.x - pose.leftEar.x);
//  textSize(50);
  // text(tag, pose.nose.x , pose.nose.y + pose.rightEar.x - pose.leftEar.x);


      for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(10);
      stroke(255);
      ellipse(x, y, 15, 15);
    }
  }
  pop();

  // fill(255,255,25);
  // noStroke();
  // textSize(100);
  // textAlign(CENTER, CENTER);
  // text(poseLabel, width / 2, height / 2);
}