function preprocessing(filename, output)
%loadDataSet receives a filename and generates a training, validation and
%testset

    load(filename);

    sizeTrainSet = size(Xtrain, 1);
   
    % get a row permutation vector
    trainPermutation = randperm(sizeTrainSet);
    
    Xtrain = Xtrain(trainPermutation, :);
    Ytrain = Ytrain(trainPermutation, :);
    
    % do normalization:
    % Search for minimal & maximal coeff
    minCoeff = min(min(Xtrain));
    maxCoeff = max(max(Xtrain));
    
    % normalize training and test
    
    Xtrain = 1 / (maxCoeff - minCoeff) * (Xtrain - minCoeff);
    Xtest = 1 / (maxCoeff - minCoeff) * (Xtest - minCoeff);
    
    % save to files
    save(output, 'Xtrain', 'Xtest', 'Ytrain', 'Ytest');

end

