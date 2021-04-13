clear all;
close all;
clc;

gTruth=open('mask_gTruth.mat');

net = mobilenetv2();
lgraph = layerGraph(net);

imageInputSize = [416 416 3];

imgLayer = imageInputLayer(imageInputSize,"Name","input_1");

lgraph = replaceLayer(lgraph,"input_1",imgLayer);

featureExtractionLayer = "block_12_add";

modified = load("mobilenetv2Block12Add.mat");
lgraph = modified.mobilenetv2Block12Add;

filterSize = [3 3];
numFilters = 96;

detectionLayers = [
    convolution2dLayer(filterSize,numFilters,"Name","yolov2Conv1","Padding", "same", "WeightsInitializer",@(sz)randn(sz)*0.01)
    batchNormalizationLayer("Name","yolov2Batch1")
    reluLayer("Name","yolov2Relu1")
    convolution2dLayer(filterSize,numFilters,"Name","yolov2Conv2","Padding", "same", "WeightsInitializer",@(sz)randn(sz)*0.01)
    batchNormalizationLayer("Name","yolov2Batch2")
    reluLayer("Name","yolov2Relu2")
    ];

numClasses = 2;

anchorBoxes = [
    16 16
    32 16
    ];

numAnchors = size(anchorBoxes,1);
numPredictionsPerAnchor = 2;
numFiltersInLastConvLayer = numAnchors*(numClasses+numPredictionsPerAnchor);

detectionLayers = [
    detectionLayers
    convolution2dLayer(1,numFiltersInLastConvLayer,"Name","yolov2ClassConv",...
    "WeightsInitializer", @(sz)randn(sz)*0.01)
    yolov2TransformLayer(numAnchors,"Name","yolov2Transform")
    yolov2OutputLayer(anchorBoxes,"Name","yolov2OutputLayer")
    ];

lgraph = addLayers(lgraph,detectionLayers);
lgraph = connectLayers(lgraph,featureExtractionLayer,"yolov2Conv1");

[imds,blds] = pixelLabelTrainingData(gTruth.gTruth);

cds = combine(imds,blds);

options = trainingOptions('adam', ...
       'InitialLearnRate', 0.001, ...
       'Verbose',true, ...
       'MiniBatchSize',2, ...
       'MaxEpochs',30, ...
       'Plots','training-progress', ...
       'Shuffle','every-epoch', ...
       'VerboseFrequency',10); 
   
[detector,info] = trainYOLOv2ObjectDetector(cds,lgraph,options);