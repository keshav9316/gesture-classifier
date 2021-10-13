let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "";
// start --> single pose estimation
function setup() {                    //implicit p5.js call.....run only once
  let div = createCanvas(640, 480);   //create canvas
  div.style('background-color', 'gray'); 
  div.center('horizontal');
  div.position(width/2 , height/10);  // positioning of canvas
  video = createCapture(VIDEO);       // establish a video connection to the canvas (VIDEO is implicitly provided by p5)
  video.hide();                       // hide video because we don't want to display camera stream without any processing

//    let poseOptions = {   // multi pose estimation arguments
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

  poseNet = ml5.poseNet(video,modelLoaded);  // preload poseNet model from ml5 API
  poseNet.on('pose', gotPoses);              // if found a pose call gotPoses callback function
//-> Defining a Neural Network
   let options = {       // configure the architecture of neural network
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);   // create a neural network with specified config.
  
//-> Initializing Neural Network with our trained model        // Teachable Machine
  const modelInfo = {   
    model: 'model2/model.json',
    metadata: 'model2/model_meta.json',
    weights: 'model2/model.weights.bin',
  };
  brain.load(modelInfo, brainLoaded);   // on successful initialization of NN give a callback to brainLoaded which makes classification calls
}

function brainLoaded() {    // Callback Function
  console.log('pose classification ready!');
  classifyPose();       // make classification call
}

function classifyPose() {  // classify pose
  if (pose) {                                         // if got a pose object in console
    let inputs = [];                                  
    for (let i = 0; i < pose.keypoints.length; i++) { // traverse over all keypoints and push their coordinates in input array
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);              // pose estimation model detect our poses and the neural network classifies them
  } else {
    setTimeout(classifyPose, 100);                  // if we didn't got a pose wait for .1s and then, again call classify pose
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
