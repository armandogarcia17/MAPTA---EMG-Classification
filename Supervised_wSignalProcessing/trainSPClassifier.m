function [trainedClassifier, validationAccuracy] = trainSPClassifier(trainingData)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a table containing the same predictor and response
%       columns as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 19-Apr-2017 10:36:28


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'var1', 'env1', 'tpwr1', 'var2', 'env2', 'tpwr2', 'var3', 'env3', 'tpwr3', 'var4', 'env4', 'tpwr4', 'var5', 'env5', 'tpwr5', 'var6', 'env6', 'tpwr6', 'var7', 'env7', 'tpwr7', 'var8', 'env8', 'tpwr8'};
predictors = inputTable(:, predictorNames);
response = inputTable.Action;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 1349);
if iscategorical(response)
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 60, ...
    'Learners', template,...
    'ClassNames',categorical({'Chuck Grip'; 'Fine Pinch'; 'H. Open'; 'Hook Grip'; 'Key Grip'; 'No Move'; 'Power Grip'; 'Thumb Enclosed'; 'Tool Grip'; 'W. Abduction'; 'W. Adduction'; 'W. Extension'; 'W. Flexion'; 'W. Pronation'; 'W. Supination'}));
else
    classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 60, ...
    'Learners', template,...
    'ClassNames',1:15);

end

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'var1', 'env1', 'tpwr1', 'var2', 'env2', 'tpwr2', 'var3', 'env3', 'tpwr3', 'var4', 'env4', 'tpwr4', 'var5', 'env5', 'tpwr5', 'var6', 'env6', 'tpwr6', 'var7', 'env7', 'tpwr7', 'var8', 'env8', 'tpwr8'};
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2017a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'var1', 'env1', 'tpwr1', 'var2', 'env2', 'tpwr2', 'var3', 'env3', 'tpwr3', 'var4', 'env4', 'tpwr4', 'var5', 'env5', 'tpwr5', 'var6', 'env6', 'tpwr6', 'var7', 'env7', 'tpwr7', 'var8', 'env8', 'tpwr8'};
predictors = inputTable(:, predictorNames);
response = inputTable.Action;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
