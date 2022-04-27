
function [ output_args ] = PR_Project( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    NofXtrain= 1000;
    NofXtest = 1000;
    alpha = 0.5; % learning Rate
    threshold=0.9997;
    k=3;
    choice=2;    
    Layers = [ 10;5;3 ; 1];
    NumOfLayers= size(Layers , 1);
    [XTrain, YTrain, XTest, YTest ,class1Matrix,class2Matrix]= PrepareData(NofXtrain,NofXtest);
    prompt = 'Press n if you want Nural Network , k if KNN , b if baysian classifier ? ';
    choiceStr = input(prompt , 's');
    
    if (choiceStr=='n')
        [YPredict , theta] = Neural (Layers,XTrain (:,:) , YTrain , alpha , threshold);
        [A,Z]= FeedForwardPropagation (XTest,NofXtrain , theta , NumOfLayers);
        YPredict = A{NumOfLayers+1};
        YPredict(A{NumOfLayers+1} > threshold )=2;
        YPredict(A{NumOfLayers+1} < threshold )=1;
    elseif (choiceStr=='k')
        YPredict =  KNN(XTrain , XTest ,YTrain , NofXtrain ,  k , choice);
        
    elseif (choiceStr=='k2')
         NofXtrain= 350;
         NofXtest = 50;
         [XTrain, YTrain, XTest, YTest ,class1Matrix,class2Matrix]= PrepareData2(NofXtrain,NofXtest);
          YPredict =  KNN2(XTrain , XTest ,YTrain , NofXtrain ,  k , choice);
    elseif (choiceStr=='b')
       Mus = EstimateMus(class1Matrix , class2Matrix);
       Sigmas = EstimateSigmas(class1Matrix , class2Matrix, Mus );
       [YPredict] = GeneralBayesClassifier(XTest, Mus, Sigmas, YTest);
       
    end

   accuracy = CalculateAccuracy(YTest, YPredict)
   
% KNN(XTrain , XTest ,YTrain , NofXtrain);
end

function [XTrain, YTrain, XTest, YTest, class1Matrix, class2Matrix]=PrepareData(NofXtrain,NofXtest)
    numOfFeatures =20;
    numofFiles=5;
    allrecords =NofXtrain+NofXtest;
    XTrain= zeros(NofXtrain ,numOfFeatures);
    YTrain = zeros(NofXtrain ,1);
    XTest = zeros(NofXtest,numOfFeatures);
    YTest= zeros(NofXtest,1);
    records = 400;
    %allData= zeros (records * numofFiles , numOfFeatures);
    %[a,b]= size(allData)
    classLable = zeros(records * numofFiles ,1 );
    strFile = ["z","O","N","S","F"];
    count1=1;
    for i=1:5
        filename = strFile(i);
        extension = '.txt';
        Myfile = strcat(filename,extension);   
        FileID = fopen(Myfile,'r');
        formatSpec = '%f';
        data = fscanf(FileID,formatSpec ,[records , numOfFeatures ] ); %[numOfFeatures , records]
        count2= count1+records -1;
        allData (count1 :count2 ,:) =data;
        count1= count2+1;
        %[s,f]= size(allData)
    end
    %/////******  assign class lable ******/////////
    OneDdata = allData ./ max (allData);
    
    count1= 2*records ; % first 2 files
    classLable(1: count1, 1)=1;
    count1=count1+1;
    count2= 5*records;
    classLable(count1:count2 , 1)=2;
    
    
    OneDdata = [OneDdata classLable]; 
    
    finalData=OneDdata(randperm(size(OneDdata, 1)),:);
    lastColumn = finalData(:, end);
    class1Matrix = finalData(lastColumn == 1, 1:numOfFeatures);
    class2Matrix = finalData(lastColumn == 2, 1:numOfFeatures);
    XTrain(:,:)= finalData(1:NofXtrain , 1:numOfFeatures);
    YTrain= finalData(1:NofXtrain , end);
    XTest = finalData(NofXtrain+1 :allrecords , 1:numOfFeatures);
    YTest= finalData(NofXtrain+1 :allrecords , end);
    
end
% %         dataMatrix= reshape(OneDdata,400,[]);

function [XTrain, YTrain, XTest, YTest] = PrepareData2(NofXtrain,NofXtest)
    numOfFeatures =20;
    numofFiles=5;
    XTrain= zeros(NofXtrain *numofFiles ,numOfFeatures);
    YTrain = zeros(NofXtrain *numofFiles ,1);
    XTest = zeros(numofFiles*NofXtest,numOfFeatures);
    YTest= zeros(numofFiles *NofXtest,1);
    records = NofXtrain+NofXtest;
    strFile = ["z","O","N","S","F"];
    count1=1;
    count2=1;
    for i=1:5
        filename = strFile(i);
        extension = '.txt';
        Myfile = strcat(filename,extension);   
        FileID = fopen(Myfile,'r');
        formatSpec = '%f';
        data = fscanf(FileID,formatSpec ,[records , numOfFeatures ] ); %[numOfFeatures , records]
%         dataMatrix= reshape(OneDdata,400,[]);
%         [x,y,z] = size(OneDdata)
        OneDdata = data ./ max (data);
        count= count1+NofXtrain-1;
        XTrain ([count1:count],: )= OneDdata([1:NofXtrain],:);
        count=count2 +NofXtest -1; 
        XTest([count2 : count] ,:) =OneDdata([NofXtrain+1 :records],:) ;

        if(i<3)
            YTrain([count1:count] ,1) = 1; %class 1
            YTest([count2 :count] , 1)=1;
        else  
            YTrain([count1:count] , 1) = 2; %class 2
            YTest([count2: count] , 1)=2;
        end
        
        count1 = count1+NofXtrain;
        count2 = count2+NofXtest;

    end
end

function [YPredict , theta] = Neural (HidddenLayers ,XTrain  , YTrain , alpha , threshold )

    epochs = 50;
    [trainingSamples , NumOfFeatures] = size (XTrain );
    cost = zeros(epochs , 1);
    NumOfLayers= size(HidddenLayers , 1);  % HidddenLayers + ouutput
    %NumOfLayers = 2; % hidde nurons 
    parameter1 = size (XTrain , 2);
    parameter2 = HidddenLayers(1);
    
    for k =1:NumOfLayers
        theta{k} = rand ( parameter1 +1 ,parameter2 );
        if k < NumOfLayers 
            parameter1 = parameter2;
            parameter2 = HidddenLayers(k +1);
        end
    end
    x=1;
    for i=1:epochs
      % *** Forward Propagation Step ***
      [A,Z]= FeedForwardPropagation (XTrain,trainingSamples , theta , NumOfLayers);
      % *** Calculate the Cost ***
      cost(i) = calc_cost2(A{NumOfLayers+1} ,YTrain , trainingSamples);
      
      % *** Backward Propagation Step ***
      [GZdash , S]= BackwardPropagation (NumOfLayers ,YTrain ,trainingSamples ,Z , A , theta);
      
      % *** Update Weights ****
      
      for k =1:NumOfLayers
          delta{k+1}= (S{k+1}.' * A{k} )./trainingSamples; 
          theta{k}= theta{k} - alpha * (delta{k+1}.');
      end
      
    end 
    plot(cost);
    YPredict = A{NumOfLayers+1};
    YPredict(A{NumOfLayers+1} > threshold )=2;
    YPredict(A{NumOfLayers+1} < threshold )=1;
end

function [A,Z]= FeedForwardPropagation (X,SamplesNum , theta , NumOfLayers)

  A{1} = [ ones(SamplesNum ,1) X(:,:)];% a0 = a{1}
  for j=2:NumOfLayers+1 % 2 iterations 2w3
      Z{j} = A{j-1} * theta{j-1};   % z{2} >> z1
      A{j} = 1./(1+ exp(-Z{j}) ); % a{2} = a1   Siigmoid 
      if j < NumOfLayers+1
        A{j} = [ ones(SamplesNum ,1) A{j} ];
      end
       
  end 
  
end

function cost = calc_cost ( h0 , YTrain  ,m)
    eq= -1 .* YTrain .* log10(h0) - (1-YTrain).*log10(1-h0);
    summ = sum (eq);
    cost = summ ./ m;
%     cost = 1/m .*(
end

function cost = calc_cost2 ( h0 , YTrain  ,m)
    eq= (h0-YTrain).^2;
    summ = sum (eq);
    cost = summ ./ 2*m;
%     cost = 1/m .*(
end

function [GZdash , S]= BackwardPropagation (NumOfLayers ,YTrain ,trainingSamples ,Z , A , theta)

  S{NumOfLayers+1} = A{NumOfLayers+1} - YTrain ; % S{3}
  for j= NumOfLayers :-1:2
      GZ{j}= 1./(1+ exp(-Z{j}) );
      GZdash{j}= GZ{j} .* (1-GZ{j});
      S{j}=  ( theta{j} * (S{j+1}.')  ).' .* [ ones(trainingSamples,1) GZdash{j}]; % A{2}= A1 * A{j}
      S{j}(:,1) = [];
  end
  
end

function Neural2 (HidddenLayers ,XTrain )

    epochs = 50;
    trainingSamples = size (XTrain , 1);
    NumOfLayers = size (HidddenLayers , 1);
    Input = size (XTrain , 2);
    hidden1 = 2;
    output = 2;
    theta1 = rand ( Input+1 ,hidden1 )
    theta2 = rand ( hidden1 ,output )
    x=1;
    for i=1:epochs
      % *** Forward Step ***
      A0 = [ ones(trainingSamples ,1) XTrain(:,:)] % a0 = a{1}
      Z1 = theta1 .* A0 % z{1}
      A1 = 1./(1+ exp(-Z1) ) % a{1}
      A1 = [ones(trainingSamples ,1) A1]
       
 
    end 

end

function [YPredict] = GeneralBayesClassifier(X, Mus, Sigmas ,Y ) 

    count1 = [1,histc(Y,1)];
    count2 = [2,histc(Y,2)];
    rows = size (X,1);
    Pw = [count1(1,2)/rows  count2(1,2)/rows]; % class1,class2
    C = 2; % number of classes
    F = 20; %number of features
% p(x1|c1) p(x2|c1) p(x3|c1) 
% p(x1|c2) p(x2|c2) p(x3|c2) 

YPredict = zeros(rows ,1);
for k=1:rows
    Pxjwi = zeros(C,F); %likelihoods "p(x|ck)" in ex: 2*3
    for i=1:C
        for j=1:F
            
           Pxjwi(i,j) =mynormalfn(X(k,j), Mus(i,j), Sigmas(i,j));
        end
    end

    % PXw >> p(ck)p(x|ck) >> perior* likelihood

    PXw = ones(C,1);
    for i=1:C
        for j=1:F        
            PXw(i) = PXw(i)* Pxjwi(i,j) ;
        end
    end

    %2) Compute the posterior probabilities P(wi|x1,x2) by which we decide
    % that the given features belongs to which class!
    PwX = zeros(C,1); %Posteriors
    sum = 0;
    for i=1:C
        PwX(i) = PXw(i) * Pw(i);
        sum = sum + PwX(i); % calculate the P(x)
    end
    PX = sum ;% >> p(x)


    %Normalize the posteriors
    maxprob =0;
    class_index =1;
    for i=1:C
        PwX(i) = PwX(i) / PX;
        if (PwX(i) >maxprob)
            maxprob = PwX(i);
            class_index = i;
        end
    end
YPredict(k,1)= class_index;
end
end

function p = mynormalfn(x, mu, sigma)
p = (1 / sqrt(2 * pi) * sigma) * exp(-(x - mu).^2/(2 * sigma^2));
end

function Mus = EstimateMus(class1Matrix , class2Matrix)   
   %Hint:: you can validate the output by using "mean(X)"
   
   %Your code goes here ...
%         mus2 = mean(X);
%         mus2
    Mus = zeros(2, 20); % 2 classes w 20 feature 
    class1Eements= size(class1Matrix,1);
    class2Eements= size(class2Matrix,1);
    summ1 = sum (class1Matrix);
    summ2 = sum (class2Matrix);         
    Mus(1,: ) = summ1 / class1Eements ; 
    Mus(2,: ) = summ2 / class2Eements;
    mus1(1,: ) = mean(class1Matrix);
    mus1(2,: ) = mean(class2Matrix);
  % mus2
  % Mus

   
end

function Sigmas = EstimateSigmas(class1Matrix ,class2Matrix, Mus )
    %Hint:: you can validate the output by using 0"std(X)"
        class1Eements= size(class1Matrix,1);
        class2Eements= size(class2Matrix,1);
        Sigmas(1,:)=sqrt((sum(class1Matrix.^2) - class1Eements*Mus(1,:).^2)/(class1Eements-1));
        Sigmas(2,:)=sqrt((sum(class2Matrix.^2) - class2Eements*Mus(2,:).^2)/(class2Eements-1));
        sigmas2(1,:)= std(class1Matrix);
        sigmas2(2,:)= std(class2Matrix);
end

function YPredict =  KNN(XTrain , XTest ,YTrain , NofXtrain , k  , choice)

    [testingSamples ,numOfFeatures ] = size(XTest); %testing samples 
    YPredict = zeros (testingSamples , 1);
    numOfFiles=5;
    trainingSamples = size(XTrain, 1);
    dist = zeros(trainingSamples,2);
     for i=1:testingSamples
         if (choice ==1)
            dist = get_euclidean_distance(XTrain(:,:),YTrain ,XTest(i,:) , NofXtrain);
         elseif (choice ==2)
             dist= get_cosin_simlarity (XTrain(:,:),YTrain ,XTest(i,:) , NofXtrain);
         end
        [sorted ,dist] = sort(dist);
%         dist = sortrows(dist , 1);
        count1 = [1,histc(dist(1:k,2),1)];
        count2 = [2,histc(dist(1:k,2),2)];
        if count1(:,2)>= count2(:,2) 
            YPredict(i,1)=1;
        else
             YPredict(i,1)=2;
        end                        
     end
end

function YPredict =  KNN2(XTrain , XTest ,YTrain , NofXtrain, k  , choice )

    [testingSamples ,numOfFeatures ] = size(XTest) %testing samples 
    YPredict = zeros (testingSamples , 1);
    numOfFiles=5;
    trainingSamples = size(XTrain, 1);
    dist = zeros(trainingSamples,2);
     for i=1:testingSamples
        dist = get_euclidean_distance(XTrain(:,:),XTest(i,:) , NofXtrain);
        dist = sortrows(dist , 1);
        count1 = [1,histc(dist(1:k ,2),1)];
        count2 = [2,histc(dist(1:k ,2),2)];
        if count1>= count2 
            YPredict(i,1)=1;
        else
             YPredict(i,1)=2;
        end
            
            
     end
end

function dist = get_euclidean_distance(X,YTrain, sample , NofXtrain) 
    m = size(X,1);    
    dist = zeros(m,2);
    for i=1: m
        diff = X(i ,:)-sample (:,:);
        power = diff.^2;
        res = sum(power);
        dist(i , 1) = sqrt(res);
        dist(i,2)= YTrain(i,1);
    end
end

function dist = get_euclidean_distance2(X, sample , NofXtrain)
    m = size(X,1);    
    dist = zeros(m,2);
    class1Eements = 2* NofXtrain; % healthy
    class2Eements = 3* NofXtrain; % non healthy
    for i=1: m
        diff = X(i ,:)-sample (:,:);
        power = diff.^2;
        res = sum(power);
        dist(i , 1) = sqrt(res);
    end
   dist ([1 : class1Eements] , 2 ) = 1;
   dist ([ class1Eements +1: class2Eements] , 2 ) = 2;
end

function dist = get_euclidean_distance3(X, sample , NofXtrain)
        dist = zeros(20,1);
        diff=bsxfun(@minus,X,sample)       
        power = diff.^2;
        res = sum(power);
        dist = sqrt(res);
end

function dist= get_cosin_simlarity (X,YTrain, sample , NofXtrain)
    m = size(X,1);    
    dist = zeros(m,2);
    for i=1: m
        summ1= sum(X .* sample);
        demonitor =sqrt (sum( X .^2)) * sqrt(sum(sample .^2));               
        dist(i , 1) = summ1/demonitor;
        dist(i,2)= YTrain(i,1);
    end
end

function accuracy = CalculateAccuracy(YTrue, YPredict)
    sz = size(YTrue, 1);
    truePredict = YTrue == YPredict;
    accuracy = (sum(truePredict) / sz)*100;
end

