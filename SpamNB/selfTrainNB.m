labelledEx = input('Input vector like [10 20 30 ..] where each term is number of labelled document');
temp = labelledEx;
temp = temp * -1;
temp = 4140+temp;
UnlabelledEx = temp;
disp('labelled Distribution: ');
disp(labelledEx);
disp('unlabelled Distribution: ');
disp(UnlabelledEx);

noOfItr = input('Enter number iteration before each distribution converges');


loEX = labelledEx;%[50 100 200 500 900 1000 1500 2000 2500 3000 3500 4000 ];

% labelledEx = [50 100 200 500 900 1000 1500 2000 2500 3000 3500 4000 ];
% UnlabelledEx = [4140-50 4140-100 4140-150 4140-200 4140-250 4140-300 4140-350 4140-400 4140-450 4140-500 4140-550 4140-600 4140-650 4140-700 4140-750 4140-800 4140-850 4140-900 4140-950 4140-1000];

%Build the model for classifier
load sSpamDatabase.dat;
testData  =  sSpamDatabase(4141:size(sSpamDatabase,1),:);

accuracyVector = ones(size(labelledEx,2),4);

lConfMat = zeros(2,2);
luConfMat = zeros(2,2);

for colVal = 1:size(labelledEx,2) % loop over each label separation
        lCount = labelledEx(1,colVal);
        ulCount = UnlabelledEx(1,colVal);
        
        lTrainData = sSpamDatabase(1:lCount,:); % labelled Set
        ulTrainData = sSpamDatabase(lCount+1:ulCount,:); % Unlabelled Set

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
        
         for itr = 1:noOfItr 
                rowsNonSpam = size(nonSpamF,1);
                rowsSpam = size(spamF,1);

                priors = [rowsNonSpam/size(lTrainData,1), rowsSpam/size(lTrainData,1)];%priors  [0.6060    0.3940]
%                 disp(priors);
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
%                 disp(likelihoodMat);
                
                %Classify Unlablelled data
                ulpLikelyHood = ones(size(ulTrainData,1),2);
                ulpredictClass = ones(size(ulTrainData,1),1);
                ulmulWithPriors = ones(size(ulTrainData,1),2);

                for ulrowno = 1:size(ulTrainData,1)% all rows in testdata
                    ulcurrRow = ulTrainData(ulrowno,1:57);
                    for ulcolno = 1:size(ulcurrRow,2) % all columns in current rows
                        if ulcurrRow(1,ulcolno) ~= 0 && ulpLikelyHood(ulrowno,1) ~= 0 && likelihoodMat(1,ulcolno) ~= 0 && likelihoodMat(2,ulcolno) ~= 0
                            ulpLikelyHood(ulrowno,1) = ulpLikelyHood(ulrowno,1)*likelihoodMat(1,ulcolno);
                            ulpLikelyHood(ulrowno,2) = ulpLikelyHood(ulrowno,2)*likelihoodMat(2,ulcolno);
                        end
                    end
                end

                ulmulWithPriors(:,1) = ulpLikelyHood(:,1) * priors(1,1); 
                ulmulWithPriors(:,2) = ulpLikelyHood(:,2) * priors(1,2);

                maxPositiveNS = max(ulmulWithPriors(:,1));
                minPositiveNS = min(ulmulWithPriors(:,1));
                confNS = (maxPositiveNS+minPositiveNS)/2;
                
                maxPositiveS = max(ulmulWithPriors(:,2));
                minPositiveS = min(ulmulWithPriors(:,2));
                confS = (maxPositiveS+minPositiveS)/2;
                
                for rowno = 1:size(ulmulWithPriors,1)
                    if ulmulWithPriors(rowno,1) > ulmulWithPriors(rowno,2) && ulmulWithPriors(rowno,1) >= confNS
                        nonSpamF = [nonSpamF;ulTrainData(rowno,1:57)];
                    elseif ulmulWithPriors(rowno,2) > ulmulWithPriors(rowno,1) && ulmulWithPriors(rowno,2) >= confS
                        spamF = [spamF;ulTrainData(rowno,1:57)];
                    end    
                end
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
       
%         luConfMat = zeros(2,2);
        %populating confusion matric
        for rowno = 1:size(predictClass,1)
            % Calculating CM
            if predictClass(rowno,1) == 1 && actClass(rowno,1) == 1		        
                luConfMat(1,1) = luConfMat(1,1) + 1;
            elseif predictClass(rowno,1) == 0 && actClass(rowno,1) == 0
                luConfMat(2,2) = luConfMat(2,2) + 1;
            elseif predictClass(rowno,1) == 1 && actClass(rowno,1) == 0
                luConfMat(1,2) = luConfMat(1,2) + 1;
            elseif predictClass(rowno,1) == 0 && actClass(rowno,1) == 1
                luConfMat(2,1) = luConfMat(2,1) + 1;
            end	
        end

        %Finding valuation paramenters

        luaccuracy = (luConfMat(1,1) + luConfMat(2,2))/(luConfMat(1,1) + luConfMat(2,2) + luConfMat(1,2) + luConfMat(2,1));
        luprecision = luConfMat(1,1)/(luConfMat(1,1) + luConfMat(1,2));
        lurecall = luConfMat(1,1)/(luConfMat(1,1) + luConfMat(2,1));
        f1 = (2 * luprecision * lurecall)/(luprecision + lurecall);
        accuracyVector(colVal,1) = lCount;
        accuracyVector(colVal,2) = luaccuracy;

        disp('SELF TRAINING - End of iteration with labelled data: ');
        disp(lCount);
end


%Only Labled data

for colVal = 1:size(loEX,2) % loop over each label separation
        lCount = loEX(1,colVal);
                
        lTrainData = sSpamDatabase(1:lCount,:); % labelled Set

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
%                 disp(priors);
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
%         lConfMat = zeros(2,2);
       %populating confusion matric
        for rowno = 1:size(predictClass,1)
            % Calculating CM
            if predictClass(rowno,1) == 1 && actClass(rowno,1) == 1		        
                lConfMat(1,1) = lConfMat(1,1) + 1;
            elseif predictClass(rowno,1) == 0 && actClass(rowno,1) == 0
                lConfMat(2,2) = lConfMat(2,2) + 1;
            elseif predictClass(rowno,1) == 1 && actClass(rowno,1) == 0
                lConfMat(1,2) = lConfMat(1,2) + 1;
            elseif predictClass(rowno,1) == 0 && actClass(rowno,1) == 1
                lConfMat(2,1) = lConfMat(2,1) + 1;
            end	
        end
      %Finding valuation paramenters
        laccuracy = (lConfMat(1,1) + lConfMat(2,2))/(lConfMat(1,1) + lConfMat(2,2) + lConfMat(1,2) + lConfMat(2,1));
        lprecision = lConfMat(1,1)/(lConfMat(1,1) + lConfMat(1,2));
        lrecall = lConfMat(1,1)/(lConfMat(1,1) + lConfMat(2,1));
        lf1 = (2 * lprecision * lrecall)/(lprecision + lrecall);
        accuracyVector(colVal,3) = lCount;
        accuracyVector(colVal,4) = laccuracy;
        disp('SUPERVISED TRAINING- End of iteration with labelled data: ');
        disp(lCount);
end




%Results
        disp(sprintf('\n\n'));
        disp(sprintf('Labelled Unlablleled Confusion Matrix\n\n'));
%         disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',luConfMat(1,1), luConfMat(1,2), luConfMat(2,1), luConfMat(2,2)));
        disp(sprintf('\nACCURACY : %f',luaccuracy));
        disp(sprintf('\nPRECISION : %f',luprecision));
        disp(sprintf('\nRECALL : %f',lurecall));
        disp(sprintf('\nF1-Measure : %f',f1));
        disp(sprintf('\n\n'));
        
        
        %Results
        disp(sprintf('\n\n'));
        disp(sprintf('Labelled Only Confusion Matrix\n\n'));
%         disp(sprintf('\t| %5d | %5d |\n\t| %5d | %5d |\n\n',lConfMat(1,1), lConfMat(1,2), lConfMat(2,1), lConfMat(2,2)));
        disp(sprintf('\nACCURACY : %f',laccuracy));
        disp(sprintf('\nPRECISION : %f',lprecision));
        disp(sprintf('\nRECALL : %f',lrecall));
        disp(sprintf('\nF1-Measure : %f',lf1));
        disp(sprintf('\n\n'));
        
title('Graph comparing Self Training and Supervised Learning');        
plot(accuracyVector(:,1),accuracyVector(:,2),':r*',accuracyVector(:,3),accuracyVector(:,4),'--go');
ylim([0 1]);
legend('y = Self Training','y = Supervised Learning');
xlabel('Number of labeled document');
ylabel('Accuracy');
