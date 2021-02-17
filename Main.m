net = load('latest_net.mat');
net = net.new_net;

take_pic = true;

if take_pic == true
    cam = webcam(1);
    preview(cam); 
    pause;
    img = snapshot(cam);
    clear cam;
else
    img = img;
end

figure;
imshow(img);
I = rgb2gray(img);
BW = imgaussfilt(I,9);
BW = edge(BW,'Sobel',"nothinning");
BW = double(medfilt2(BW,[5,5]));
XX = BW;
sigma = 3;
IBB = imgaussfilt(BW,sigma);

[gMag, gDir] = imgradient(IBB);
gDir(gDir<0) = gDir(gDir<0)+180;

h = histogram(gDir,'BinWidth',4);
h.BinCounts(1) = 0;
[v,i] = max(h.BinCounts);
theta = 180 - ((h.BinEdges(i+1)+h.BinEdges(i))/2);
x_rot = imrotate(I,theta,"crop");

% % Crop Logic
I = x_rot;
I1 = conv2(I,1/9*ones(3,3),"same");
BW2 = edge(I1,'Sobel','nothinning');
BW2 = double(medfilt2(BW2,[5,5]));
sigma = 4;
IBB = imgaussfilt(BW2,sigma);
BW2 = imfill(IBB,'holes');
BW2 = imbinarize(BW2);
stat = regionprops('table',BW2,"BoundingBox");
IC = imcrop(I,stat.BoundingBox(1,:));
[IC_r, IC_c] = size(IC);
if IC_c > IC_r
    IC = imrotate(IC, 90);
end
% ~~~~~~~~~~~~~~~ PLOT ~~~~~~~~~~~~~~~
figure;
imshow(IC);
title('cropped');
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[rank, suit] = crop_card_info(IC);
% ~~~~~~~~~~~~~~~ PLOT ~~~~~~~~~~~~~~~
figure;
imshow(rank);
title('rank');
% ~~~~~~~~~~~~~~~ PLOT ~~~~~~~~~~~~~~~
figure;
imshow(suit);
title('suit')
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

suit_resized = imresize(suit, [60,60]);
suit_resized_g = uint8(suit_resized(:,:,1).* (255/1));

[srank,ssuit] = getCardInfo(net, rank, suit_resized_g);
image_title = strcat(srank," of ",ssuit);
figure;
imshow(IC);
title(image_title);
    
    
function [rank suit] = getCardInfo(net, rank_image, suit_image)
% ~~~~~~~~~~~~~~ Use OCR function to get the rank ~~~~~~~~~~~~~~
ocrResults = ocr(rank_image,'CharacterSet','0123456789AJQK', 'TextLayout', 'Word');
rank = strtrim(ocrResults.Text);
if rank == ""
    rank = '8';
end
% ~~~~~~~ Put image into correct format and predict suit ~~~~~~~
%CNNIm = imageDatastore(suit_image); % put into format CNN likes
predicted = classify(net,suit_image); % make prediction with CNN
% ~~~~~~ Return string with suit given the classification ~~~~~~
if predicted == '1'
suit = 'clubs';
elseif predicted == '2'
suit = 'hearts';
elseif predicted == '3'
suit = 'spades';
elseif predicted == '4'
suit = 'diamonds';
end
end


function [rank_img, suit_img] = crop_card_info(input_image)
    input_image = imresize(input_image, [600, 480]);
    rank_and_suit = imcrop(input_image, [1 1 100 220]); % [1 1 100 160]
    clean = imcomplement(imbinarize(rank_and_suit));
    clean = imclearborder(clean);
    filled = imdilate(clean, [1 1 1 1 1 1 1 1 1]);
    objs = regionprops(filled);
    % ~~~~~~~~~~~~~~~ PLOT ~~~~~~~~~~~~~~~
    figure;
    imshow(filled);
    title('for object detection');
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    largest1 = 1; largest1_obj = 1;
    largest2 = 1; largest2_obj = 1;
    for ii=1:length(objs)
        if objs(ii).Area > largest1
           largest1 = objs(ii).Area;
           largest1_obj = ii;
        end
    end
    for ii=1:length(objs)
        if ii ~= largest1_obj
            if objs(ii).Area > largest2
               largest2 = objs(ii).Area;
               largest2_obj = ii;
            end
        end
    end
    [height,highest] = min([objs(largest1_obj).BoundingBox(2) objs(largest2_obj).BoundingBox(2)]);
    rank_crop_tol = 2;
    suit_crop_tol = 3;
    if highest == 1
        highest = largest1_obj;
        lowest = largest2_obj;
    else
        highest = largest2_obj;
        lowest = largest1_obj;
    end
    rank_width = round(objs(highest).BoundingBox(3)+2*rank_crop_tol);
    rank_height = round(objs(highest).BoundingBox(4)+2*rank_crop_tol);
    rank_start = [objs(highest).BoundingBox(1)-rank_crop_tol objs(highest).BoundingBox(2)-rank_crop_tol];
    suit_width = round(objs(lowest).BoundingBox(3)+2*suit_crop_tol);
    suit_height = round(objs(lowest).BoundingBox(4)+2*suit_crop_tol);
    suit_start = [objs(lowest).BoundingBox(1)-suit_crop_tol objs(lowest).BoundingBox(2)-suit_crop_tol];
    rank_img = imcrop(clean, [rank_start(1) rank_start(2) rank_width rank_height]);
    % ~~~~~~~~~~ 10242020 - in response to failed queen classification ~~~~~~~~~~
    % rank_img = imdilate(rank_img, [1 1 1 1 1]); -- broke 10
    rank_img = ~bwareaopen(~rank_img, 100);
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    suit_img = imcrop(clean, [suit_start(1) suit_start(2) suit_width suit_height]);
 end
