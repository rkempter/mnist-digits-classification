% load variables

v_tau = [0.064];
%v_tau = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005];

%v_C = 2.^(0:9);
v_C = 2;
load('files/mp_4-9_preprocessed.mat');

svm = SVMProject(Xtrain, Ytrain, Xtest, Ytest);

o_classifier = svm.crossvalidation(v_C, v_tau);

[s_C, s_tau] = svm.getBestParameters();

disp(s_C);
disp(s_tau);