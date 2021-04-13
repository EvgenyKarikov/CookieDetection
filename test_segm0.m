img_num = 20;

test_img = readimage(imds,img_num);
[box, score, label] = detect(detector,test_img);
detectedimg = insertObjectAnnotation(test_img,'Rectangle',box,label);

imshow(detectedimg);