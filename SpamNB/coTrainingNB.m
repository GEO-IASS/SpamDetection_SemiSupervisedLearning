labelledEx = input('Input vector like [10 20 30 ..] where each term is number of labelled document');
temp = labelledEx;
temp = temp * -1;
temp = 4140+temp;
UnlabelledEx = temp;
disp('labelled Distribution: ');
disp(labelledEx);
disp('unlabelled Distribution: ');
disp(UnlabelledEx);

% labelledEx = [50 100 200 500 900 1000 1500 2000 2500 3000 ];
% UnlabelledEx = [4140-50 4140-100 4140-200 4140-500 4140-900 4140-1000 4140-1500 4140-2000 4140-2500 4140-3000];

part1 = input('Enter number of feature in subset 1');
part2 = part1+1;
rem = 57-part1;
disp('Distribution of feature in two partition is');
disp(part1);
disp(rem);
noOfItr = input('Enter number iteration before each distribution converges');

%Build the model for classifier
load sSpamDatabase.dat;
testData  =  sSpamDatabase(4141:size(sSpamDatabase,1),:);

accuracyVector = ones(size(labelledEx,2),4);

for colVal = 1:size(labelledEx,2)
    % accuracyVector = ones(size(labelledEx,2),2);
    ConfMatP1 = zeros(2,2);
    ConfMatP2 = zeros(2,2);
    ConfMatComb = zeros(2,2);
    
    %Only Labled data
    lCount = labelledEx(1,colVal);
    ulCount = UnlabelledEx(1,colVal);
    
    lTrainData = sSpamDatabase(1:lCount,:); % labelled Set
    ulTrainData = sSpamDatabase(lCount+1:ulCount,:); % unlabelled Set
    
    nonSpamF = zeros(0,57); %empty matrix just memory allocation
    spamF = zeros(0,57);
    
    for rno = 1:size(lTrainData,1)%returns number of rows rows(M)=4601
        trainRow = lTrainData(rno,:);% gives whole row at a time from M
        if trainRow(1,58) == 0 % checks class
            nonSpamF = [nonSpamF;trainRow(1,1:57)]; % only features not class --NON SPAM 61%  [2788 57]
        else
            spamF = [spamF;trainRow(1,1:57)]; % Spam class 39% [1813  57]
        end
    end
    for itr = 1:noOfItr % repeat for an times
        if size(ulTrainData,1)>0
            rowsNonSpam = size(nonSpamF,1);
            rowsSpam = size(spamF,1);
            
            priors = [rowsNonSpam/size(lTrainData,1), rowsSpam/size(lTrainData,1)];%priors
            
            %Likelyhood with idf smoothing - set1
            likelihoodMatP1= zeros(2,part1);
            
            %         for cno = 1:size(nonSpamF,2)%returns number of rows rows(M)=4601
            for cno = 1:part1
                nonSpamFCol = nonzeros(nonSpamF(:,cno));% gives whole column at a time from M
                sumCol = sum(nonSpamFCol);
                likelihood = sumCol/size(nonSpamF,1);
                likelihoodMatP1(1,cno) = likelihood;
            end
            
            for cno = 1:part1 %returns number of rows rows(M)=4601
                spamFCol = nonzeros(spamF(:,cno));% gives whole column at a time from M
                sumCol = sum(spamFCol);
                likelihood = sumCol/size(spamF,1);
                likelihoodMatP1(2,cno) = likelihood;
            end
            
            %Likelyhood with idf smoothing -Set 2
            likelihoodMatP2= zeros(2,rem);
            
            %         for cno = 1:size(nonSpamF,2)%returns number of rows rows(M)=4601
            for cno = part2:57
                nonSpamFCol = nonzeros(nonSpamF(:,cno));% gives whole column at a time from M
                sumCol = sum(nonSpamFCol);
                likelihood = sumCol/size(nonSpamF,1);
                likelihoodMatP2(1,cno-part1) = likelihood;
            end
            
            for cno = part2:57 %returns number of rows rows(M)=4601
                spamFCol = nonzeros(spamF(:,cno));% gives whole column at a time from M
                sumCol = sum(spamFCol);
                likelihood = sumCol/size(spamF,1);
                likelihoodMatP2(2,cno-part1) = likelihood;
            end
            %pridicting Unlabeled Data - set 1
            pLikelyHoodP1 = ones(size(ulTrainData,1),2);
            predictClassP1 = ones(size(ulTrainData,1),1);
            mulWithPriorsP1 = ones(size(ulTrainData,1),2);
            
            
            for rowno = 1:size(ulTrainData,1)% all rows in testdata
                currRow = ulTrainData(rowno,1:part1);
                for colno = 1:size(currRow,2) % all columns in current rows
                    if currRow(1,colno) ~= 0 && pLikelyHoodP1(rowno,1) ~= 0 && likelihoodMatP1(1,colno) ~= 0 && likelihoodMatP1(2,colno) ~= 0
                        pLikelyHoodP1(rowno,1) = pLikelyHoodP1(rowno,1)*likelihoodMatP1(1,colno);
                        pLikelyHoodP1(rowno,2) = pLikelyHoodP1(rowno,2)*likelihoodMatP1(2,colno);
                    end
                end
            end
            
            mulWithPriorsP1(:,1) = pLikelyHoodP1(:,1) * priors(1,1);%multiple with the
            mulWithPriorsP1(:,2) = pLikelyHoodP1(:,2) * priors(1,2);%
            
            for rowno = 1:size(mulWithPriorsP1,1)
                if mulWithPriorsP1(rowno,1) > mulWithPriorsP1(rowno,2)
                    predictClassP1(rowno,1) = 0;
                else
                    predictClassP1(rowno,1) = 1;
                end
            end
            
            %pridicting Unlabeled Data - set 2
            pLikelyHoodP2 = ones(size(ulTrainData,1),2);
            predictClassP2 = ones(size(ulTrainData,1),1);
            mulWithPriorsP2 = ones(size(ulTrainData,1),2);
            
            
            for rowno = 1:size(ulTrainData,1)% all rows in testdata
                currRow = ulTrainData(rowno,part2:57);
                for colno = 1:size(currRow,2) % all columns in current rows
                    if currRow(1,colno) ~= 0 && pLikelyHoodP2(rowno,1) ~= 0 && likelihoodMatP2(1,colno) ~= 0 && likelihoodMatP2(2,colno) ~= 0
                        pLikelyHoodP2(rowno,1) = pLikelyHoodP2(rowno,1)*likelihoodMatP2(1,colno);
                        pLikelyHoodP2(rowno,2) = pLikelyHoodP2(rowno,2)*likelihoodMatP2(2,colno);
                    end
                end
            end
            
            mulWithPriorsP2(:,1) = pLikelyHoodP2(:,1) * priors(1,1);%multiple with the
            mulWithPriorsP2(:,2) = pLikelyHoodP2(:,2) * priors(1,2);%
            
            for rowno = 1:size(mulWithPriorsP2,1)
                if mulWithPriorsP2(rowno,1) > mulWithPriorsP2(rowno,2)
                    predictClassP2(rowno,1) = 0;
                else
                    predictClassP2(rowno,1) = 1;
                end
            end
            
            indexToBeRem = [];
            for rowno = 1:size(ulTrainData,1)
                if predictClassP1(rowno,1) == predictClassP2(rowno,1)
                    if predictClassP2(rowno,1) == 0
                        nonSpamF = [nonSpamF;trainRow(1,1:57)]; % only features not class --NON SPAM
                    else
                        spamF = [spamF;trainRow(1,1:57)]; % Spam class
                    end
                    indexToBeRem = [indexToBeRem rowno];
                end
            end
            for col = 1:size(indexToBeRem,2)
                tobeRem = indexToBeRem(1,col);
                if size(ulTrainData,1) > tobeRem
                    ulTrainData(tobeRem,:)=[];
                end
            end
        end
    end
    
    %calculate combined liklihood
    
    %Likelyhood with idf smoothing - set1
    likelihoodMatComb= zeros(2,57);
    
    %         for cno = 1:size(nonSpamF,2)%returns number of rows rows(M)=4601
    for cno = 1:57
        nonSpamFCol = nonzeros(nonSpamF(:,cno));% gives whole column at a time from M
        sumCol = sum(nonSpamFCol);
        likelihood = sumCol/size(nonSpamF,1);
        likelihoodMatComb(1,cno) = likelihood;
    end
    
    for cno = 1:57 %returns number of rows rows(M)=4601
        spamFCol = nonzeros(spamF(:,cno));% gives whole column at a time from M
        sumCol = sum(spamFCol);
        likelihood = sumCol/size(spamF,1);
        likelihoodMatComb(2,cno) = likelihood;
    end
    
    
    %--------------------------------test 1------------------------------------------------------%
    %classification test data
    actClass = testData(:,58);
    
    tLikelyHoodP1 = ones(size(testData,1),2);
    tPredictClassP1 = ones(size(testData,1),1);
    tMulWithPriorsP1 = ones(size(testData,1),2);
    
    
    for rowno = 1:size(testData,1)% all rows in testdata
        currRow = testData(rowno,1:part1);
        for colno = 1:size(currRow,2) % all columns in current rows
            if currRow(1,colno) ~= 0 && tLikelyHoodP1(rowno,1) ~= 0 && likelihoodMatP1(1,colno) ~= 0 && likelihoodMatP1(2,colno) ~= 0
                tLikelyHoodP1(rowno,1) = tLikelyHoodP1(rowno,1)*likelihoodMatP1(1,colno);
                tLikelyHoodP1(rowno,2) = tLikelyHoodP1(rowno,2)*likelihoodMatP1(2,colno);
            end
        end
    end
    
    tMulWithPriorsP1(:,1) = tLikelyHoodP1(:,1) * priors(1,1);%multiple with the
    tMulWithPriorsP1(:,2) = tLikelyHoodP1(:,2) * priors(1,2);%
    
    for rowno = 1:size(tMulWithPriorsP1,1)
        if tMulWithPriorsP1(rowno,1) > tMulWithPriorsP1(rowno,2)
            tPredictClassP1(rowno,1) = 0;
        else
            tPredictClassP1(rowno,1) = 1;
        end
    end
    
    %populating confusion matric
    for rowno = 1:size(tPredictClassP1,1)
        % Calculating CM
        if tPredictClassP1(rowno,1) == 1 && actClass(rowno,1) == 1
            ConfMatP1(1,1) = ConfMatP1(1,1) + 1;
        elseif tPredictClassP1(rowno,1) == 0 && actClass(rowno,1) == 0
            ConfMatP1(2,2) = ConfMatP1(2,2) + 1;
        elseif tPredictClassP1(rowno,1) == 1 && actClass(rowno,1) == 0
            ConfMatP1(1,2) = ConfMatP1(1,2) + 1;
        elseif tPredictClassP1(rowno,1) == 0 && actClass(rowno,1) == 1
            ConfMatP1(2,1) = ConfMatP1(2,1) + 1;
        end
    end
    
    %Finding valuation paramenters
    
    laccuracyP1 = (ConfMatP1(1,1) + ConfMatP1(2,2))/(ConfMatP1(1,1) + ConfMatP1(2,2) + ConfMatP1(1,2) + ConfMatP1(2,1));
    lprecisionP1 = ConfMatP1(1,1)/(ConfMatP1(1,1) + ConfMatP1(1,2));
    lrecallP1 = ConfMatP1(1,1)/(ConfMatP1(1,1) + ConfMatP1(2,1));
    lf1P1 = (2 * lprecisionP1 * lrecallP1)/(lprecisionP1 + lrecallP1);
    
    %--------------------------------test 2------------------------------------------------------%
    %classification test data
    actClass = testData(:,58);
    
    tLikelyHoodP2 = ones(size(testData,1),2);
    tPredictClassP2 = ones(size(testData,1),1);
    tMulWithPriorsP2 = ones(size(testData,1),2);
    
    
    for rowno = 1:size(testData,1)% all rows in testdata
        currRow = testData(rowno,part2:57);
        for colno = 1:size(currRow,2) % all columns in current rows
            if currRow(1,colno) ~= 0 && tLikelyHoodP2(rowno,1) ~= 0 && likelihoodMatP2(1,colno) ~= 0 && likelihoodMatP2(2,colno) ~= 0
                tLikelyHoodP2(rowno,1) = tLikelyHoodP2(rowno,1)*likelihoodMatP2(1,colno);
                tLikelyHoodP2(rowno,2) = tLikelyHoodP2(rowno,2)*likelihoodMatP2(2,colno);
            end
        end
    end
    
    tMulWithPriorsP2(:,1) = tLikelyHoodP2(:,1) * priors(1,1);%multiple with the
    tMulWithPriorsP2(:,2) = tLikelyHoodP2(:,2) * priors(1,2);%
    
    for rowno = 1:size(tMulWithPriorsP2,1)
        if tMulWithPriorsP2(rowno,1) > tMulWithPriorsP2(rowno,2)
            tPredictClassP2(rowno,1) = 0;
        else
            tPredictClassP2(rowno,1) = 1;
        end
    end
    
    %populating confusion matric
    for rowno = 1:size(tPredictClassP2,1)
        % Calculating CM
        if tPredictClassP2(rowno,1) == 1 && actClass(rowno,1) == 1
            ConfMatP2(1,1) = ConfMatP2(1,1) + 1;
        elseif tPredictClassP2(rowno,1) == 0 && actClass(rowno,1) == 0
            ConfMatP2(2,2) = ConfMatP2(2,2) + 1;
        elseif tPredictClassP2(rowno,1) == 1 && actClass(rowno,1) == 0
            ConfMatP2(1,2) = ConfMatP2(1,2) + 1;
        elseif tPredictClassP2(rowno,1) == 0 && actClass(rowno,1) == 1
            ConfMatP2(2,1) = ConfMatP2(2,1) + 1;
        end
    end
    
    %Finding valuation paramenters
    
    laccuracyP2 = (ConfMatP2(1,1) + ConfMatP2(2,2))/(ConfMatP2(1,1) + ConfMatP2(2,2) + ConfMatP2(1,2) + ConfMatP2(2,1));
    lprecisionP2 = ConfMatP2(1,1)/(ConfMatP2(1,1) + ConfMatP2(1,2));
    lrecallP2 = ConfMatP2(1,1)/(ConfMatP2(1,1) + ConfMatP2(2,1));
    lf1P2 = (2 * lprecisionP2 * lrecallP2)/(lprecisionP2 + lrecallP2);
    
    %--------------------------------test 3------------------------------------------------------%
    %classification test data
    actClass = testData(:,58);
    
    tLikelyHoodComb = ones(size(testData,1),2);
    tPredictClassComb = ones(size(testData,1),1);
    tMulWithPriorsComb = ones(size(testData,1),2);
    
    
    for rowno = 1:size(testData,1)% all rows in testdata
        currRow = testData(rowno,1:57);
        for colno = 1:size(currRow,2) % all columns in current rows
            if currRow(1,colno) ~= 0 && tLikelyHoodComb(rowno,1) ~= 0 && likelihoodMatComb(1,colno) ~= 0 && likelihoodMatComb(2,colno) ~= 0
                tLikelyHoodComb(rowno,1) = tLikelyHoodComb(rowno,1)*likelihoodMatComb(1,colno);
                tLikelyHoodComb(rowno,2) = tLikelyHoodComb(rowno,2)*likelihoodMatComb(2,colno);
            end
        end
    end
    
    tMulWithPriorsComb(:,1) = tLikelyHoodComb(:,1) * priors(1,1);%multiple with the
    tMulWithPriorsComb(:,2) = tLikelyHoodComb(:,2) * priors(1,2);%
    
    for rowno = 1:size(tMulWithPriorsComb,1)
        if tMulWithPriorsComb(rowno,1) > tMulWithPriorsComb(rowno,2)
            tPredictClassComb(rowno,1) = 0;
        else
            tPredictClassComb(rowno,1) = 1;
        end
    end
    
    %populating confusion matric
    for rowno = 1:size(tPredictClassComb,1)
        % Calculating CM
        if tPredictClassComb(rowno,1) == 1 && actClass(rowno,1) == 1
            ConfMatComb(1,1) = ConfMatComb(1,1) + 1;
        elseif tPredictClassComb(rowno,1) == 0 && actClass(rowno,1) == 0
            ConfMatComb(2,2) = ConfMatComb(2,2) + 1;
        elseif tPredictClassComb(rowno,1) == 1 && actClass(rowno,1) == 0
            ConfMatComb(1,2) = ConfMatComb(1,2) + 1;
        elseif tPredictClassComb(rowno,1) == 0 && actClass(rowno,1) == 1
            ConfMatComb(2,1) = ConfMatComb(2,1) + 1;
        end
    end
    
    %Finding valuation paramenters
    
    laccuracyComb = (ConfMatComb(1,1) + ConfMatComb(2,2))/(ConfMatComb(1,1) + ConfMatComb(2,2) + ConfMatComb(1,2) + ConfMatComb(2,1));
    lprecisionComb = ConfMatComb(1,1)/(ConfMatComb(1,1) + ConfMatComb(1,2));
    lrecallComb = ConfMatComb(1,1)/(ConfMatComb(1,1) + ConfMatComb(2,1));
    lf1Comb = (2 * lprecisionComb * lrecallComb)/(lprecisionComb + lrecallComb);
    
    accuracyVector(colVal,1) = lCount;
    accuracyVector(colVal,2) = laccuracyP1;
    accuracyVector(colVal,3) = laccuracyP2;
    accuracyVector(colVal,4) = laccuracyComb;
    disp('End of iteration with lablelled count : ');
    disp(lCount);
end
%Results
disp(sprintf('\n\n'));
disp(sprintf('1-54 Confusion Matrix\n\n'));
disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',ConfMatP1(1,1), ConfMatP1(1,2), ConfMatP1(2,1), ConfMatP1(2,2)));
disp(sprintf('\nACCURACY : %f',laccuracyP1));
disp(sprintf('\nPRECISION : %f',lprecisionP1));
disp(sprintf('\nRECALL : %f',lrecallP1));
disp(sprintf('\nF1-Measure : %f',lf1P1));
disp(sprintf('\n\n'));

disp(sprintf('\n\n'));
disp(sprintf('55-57 Confusion Matrix\n\n'));
disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',ConfMatP2(1,1), ConfMatP2(1,2), ConfMatP2(2,1), ConfMatP2(2,2)));
disp(sprintf('\nACCURACY : %f',laccuracyP2));
disp(sprintf('\nPRECISION : %f',lprecisionP2));
disp(sprintf('\nRECALL : %f',lrecallP2));
disp(sprintf('\nF1-Measure : %f',lf1P2));
disp(sprintf('\n\n'));

disp(sprintf('\n\n'));
disp(sprintf('Combined Confusion Matrix\n\n'));
disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',ConfMatComb(1,1), ConfMatComb(1,2), ConfMatComb(2,1), ConfMatComb(2,2)));
disp(sprintf('\nACCURACY : %f',laccuracyComb));
disp(sprintf('\nPRECISION : %f',lprecisionComb));
disp(sprintf('\nRECALL : %f',lrecallComb));
disp(sprintf('\nF1-Measure : %f',lf1Comb));
disp(sprintf('\n\n'));


title('Graph comparing Co-Training and Supervised Learning'); 
plot(accuracyVector(:,1),accuracyVector(:,2),accuracyVector(:,1),accuracyVector(:,3),':r*',accuracyVector(:,1),accuracyVector(:,4),'--go');
ylim([0 1]);
legend('y = Subset feature 1','y = Subset feature 2','y = Combined feature');
xlabel('Number of labeled document');
ylabel('Accuracy');
