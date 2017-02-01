
%Build the model for classifier
load sSpamDatabase.dat;
testData  =  sSpamDatabase(4141:size(sSpamDatabase,1),:);

ConfMat = zeros(2,2);

%Only Labled data
lTrainData = sSpamDatabase(1:4140,:); % labelled Set

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

        rowsNonSpam = size(nonSpamF,1);
        rowsSpam = size(spamF,1);

        priors = [rowsNonSpam/size(lTrainData,1), rowsSpam/size(lTrainData,1)];%priors  [0.6060    0.3940]
        %Likelyhood with idf smoothing
        likelihoodMat= zeros(2,57);

        for cno = 1:size(nonSpamF,2)%returns number of rows rows(M)=4601
             nonSpamFCol = nonzeros(nonSpamF(:,cno));% gives whole column at a time from M
             sumCol = sum(nonSpamFCol);
             likelihood = sumCol/size(nonSpamF,1);
             likelihoodMat(1,cno) = likelihood;
        end

        for cno = 1:size(spamF,2)%returns number of rows rows(M)=4601
             spamFCol = nonzeros(spamF(:,cno));% gives whole column at a time from M
             sumCol = sum(spamFCol);
             likelihood = sumCol/size(spamF,1);
             likelihoodMat(2,cno) = likelihood;
        end

%classification test data
actClass = testData(:,58);

pLikelyHood = ones(size(testData,1),2);
predictClass = ones(size(testData,1),1);
mulWithPriors = ones(size(testData,1),2);


for rowno = 1:size(testData,1)% all rows in testdata
    currRow = testData(rowno,1:57);
    for colno = 1:size(currRow,2) % all columns in current rows
        if currRow(1,colno) ~= 0 && pLikelyHood(rowno,1) ~= 0 && likelihoodMat(1,colno) ~= 0 && likelihoodMat(2,colno) ~= 0
            pLikelyHood(rowno,1) = pLikelyHood(rowno,1)*likelihoodMat(1,colno);
            pLikelyHood(rowno,2) = pLikelyHood(rowno,2)*likelihoodMat(2,colno);
        end
    end
end

mulWithPriors(:,1) = pLikelyHood(:,1) * priors(1,1);%multiple with the 
mulWithPriors(:,2) = pLikelyHood(:,2) * priors(1,2);%

for rowno = 1:size(mulWithPriors,1)
    if mulWithPriors(rowno,1) > mulWithPriors(rowno,2)
        predictClass(rowno,1) = 0;
    else
        predictClass(rowno,1) = 1;
    end    
end

%populating confusion matric
for rowno = 1:size(predictClass,1)
    % Calculating CM
    if predictClass(rowno,1) == 1 && actClass(rowno,1) == 1		        
        ConfMat(1,1) = ConfMat(1,1) + 1;
    elseif predictClass(rowno,1) == 0 && actClass(rowno,1) == 0
        ConfMat(2,2) = ConfMat(2,2) + 1;
    elseif predictClass(rowno,1) == 1 && actClass(rowno,1) == 0
        ConfMat(1,2) = ConfMat(1,2) + 1;
    elseif predictClass(rowno,1) == 0 && actClass(rowno,1) == 1
        ConfMat(2,1) = ConfMat(2,1) + 1;
    end	
end

%Finding valuation paramenters

laccuracy = (ConfMat(1,1) + ConfMat(2,2))/(ConfMat(1,1) + ConfMat(2,2) + ConfMat(1,2) + ConfMat(2,1));
lprecision = ConfMat(1,1)/(ConfMat(1,1) + ConfMat(1,2));
lrecall = ConfMat(1,1)/(ConfMat(1,1) + ConfMat(2,1));
lf1 = (2 * lprecision * lrecall)/(lprecision + lrecall);


%Results
disp(sprintf('\n\n'));
disp(sprintf('Labelled Only Confusion Matrix\n\n'));
disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',ConfMat(1,1), ConfMat(1,2), ConfMat(2,1), ConfMat(2,2)));
disp(sprintf('\nACCURACY : %f',laccuracy));
disp(sprintf('\nPRECISION : %f',lprecision));
disp(sprintf('\nRECALL : %f',lrecall));
disp(sprintf('\nF1-Measure : %f',lf1));
disp(sprintf('\n\n'));

splitFeature = input('Do you what to run NB on subset of features. Type 1 for yes and 2 for No');
if splitFeature==1
    part1 = input('Enter number of feature in subset 1');
    part2 = part1+1;
    rem = 57-part1;
    disp('Distribution of feature in two partition is');
    disp(part1);
    disp(rem);
    
    %Likelyhood with idf smoothing - set1
            likelihoodMatP1= zeros(2,part1);         
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
             disp(likelihoodMatP1);
             
            %Likelyhood with idf smoothing -Set 2
            likelihoodMatP2= zeros(2,rem);
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
                
                disp(sprintf('\n\n'));
                disp(sprintf('Subset 1 Confusion Matrix\n\n'));
                disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',ConfMatP1(1,1), ConfMatP1(1,2), ConfMatP1(2,1), ConfMatP1(2,2)));
                disp(sprintf('\nACCURACY : %f',laccuracyP1));
                disp(sprintf('\nPRECISION : %f',lprecisionP1));
                disp(sprintf('\nRECALL : %f',lrecallP1));
                disp(sprintf('\nF1-Measure : %f',lf1P1));
                disp(sprintf('\n\n'));

                disp(sprintf('\n\n'));
                disp(sprintf('Subset 2 Confusion Matrix\n\n'));
                disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',ConfMatP2(1,1), ConfMatP2(1,2), ConfMatP2(2,1), ConfMatP2(2,2)));
                disp(sprintf('\nACCURACY : %f',laccuracyP2));
                disp(sprintf('\nPRECISION : %f',lprecisionP2));
                disp(sprintf('\nRECALL : %f',lrecallP2));
                disp(sprintf('\nF1-Measure : %f',lf1P2));
                disp(sprintf('\n\n'));
    
end