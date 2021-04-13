gTruth=open('yolo_gTruth.mat');
sz = size(gTruth.gTruth.LabelData);
[imds,blds] = objectDetectorTrainingData(gTruth.gTruth);
savepath = 'boxres/';

data = load('yolo_detector.mat');
detector = data.detector;

for i=1:sz(1) 
    img_num = i;

    test_img = readimage(imds,img_num);

    [box, score, label] = detect(detector,test_img);
    detectedimg = insertObjectAnnotation(test_img,'Rectangle',box,label);
    imwrite(detectedimg,savepath+string(i)+'.jpg')
    %imshow(detectedimg);
end



