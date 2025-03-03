function fsnames = selectFeaturesSequentially(TrainData)
% Select necessary feature names

% Numeric values
X = TrainData;
X.Action = [];
X = table2array(X);
Y = double(TrainData.Action);

% Setup
gcp; % open a pool of workers
opts = statset('display','iter','UseParallel','always','TolFun',1e-2);

% use our cv partition object with 3-fold cross validation
cv = cvpartition(size(X,1),'holdout',0.5);

% Determine important features
fs = sequentialfs(@featureTest,X,Y,'options',opts,'cv',cv);

% Extract
fsnames = TrainData.Properties.VariableNames(fs);
  
end

function criterion = featureTest(Xtrain, Ytrain, Xtest, Ytest)

    % Template and fit
    t = templateTree('MinLeaf',1);
    tbfit = fitensemble(Xtrain,Ytrain,'Bag',60,t,'type','classification');
    yfitTBLS = predict(tbfit,Xtest);

    resid = yfitTBLS - Ytest;

    % Square residuals
    resid_sqrd = resid.*resid;

    % Take the sum of the squared residuals
    criterion = sum(resid_sqrd);

end
