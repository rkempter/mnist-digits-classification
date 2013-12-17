classdef SVMProject < handle
    
    properties(GetAccess = 'private', SetAccess = 'private')
        
        % number of training input vectors
        c_s_nbr_input_train
        
        c_s_fold_size
        
        % Best parameter for C (box constraint)
        c_s_best_C
        
        % best parameter for tau
        c_s_best_tau
        
        % Precomputed ||x_i - x_j||^2 for fast kernel computation
        c_m_A
        
        % Training input set
        c_m_train
        
        % Training targets
        c_v_train_targets
        
        % Test input set
        c_m_test
        
        % Test targets
        c_v_test_targets
        
        % 10-fold cross validation
        c_s_fold_nbr = 10;
    end
    
    
    methods
        
        function svm = SVMProject(m_train_input, v_train_targets, m_test_input, v_test_targets)
            
            svm.c_m_train = m_train_input;
            % store them as row vectors
            svm.c_v_train_targets = v_train_targets';
            svm.c_m_test = m_test_input;
            % store them as row vectors
            svm.c_v_test_targets = v_test_targets';
            
            % do random permutation of input
            
            % Number of input vectors is equal to the number of rows
            svm.c_s_nbr_input_train = size(m_train_input, 1);
            
            % determine the size of each fold (input should be dividable by
            % 10)
            svm.c_s_fold_size = svm.c_s_nbr_input_train / svm.c_s_fold_nbr;
            
            % precompute the norm difference ||x_i - x_j||^2
            svm.c_m_A = svm.getNormDiff(m_train_input);
        end
        
        
        function o_best_classifier = crossvalidation(svm, v_C, v_tau)
            % loop over all tau
            
            s_min_error = 10000;
            
            m_error = zeros(size(v_tau, 2), size(v_C, 2));
            
            for i = 1:size(v_tau, 2)
                
               % store current tau
               s_tau = v_tau(i);
               
               for j = 1:size(v_C, 2)
                   
                   s_C = v_C(j);
                   
                   disp('===== Try new configuration =======');
                   fprintf('Current configuration: C = %d, tau = %d\n', s_C, s_tau);
                   
                   % set current error to 0
                   s_error = 0;
                   for s_k = 0:(svm.c_s_fold_nbr - 1)
                       
                       % get training and validation set for this fold.
                       % Adjust kernel for current fold.
                       [m_train_input, v_train_target, m_valid_input, v_valid_target] = svm.getTrainTestKernel(s_k);
                       
                       m_A = svm.getNormDiff(m_train_input);
                       
                       kernel = exp(-s_tau * m_A);
                       
                       % Create instance of SMOClassifier and train it
                       [o_classifier, s_nbr_iter] = svm.trainClassifier(s_C, s_tau, v_train_target, kernel, 0);
                       
                       % Check with the validation set
                       [s_nbr_error, s_nbr_correct] = o_classifier.classify(m_valid_input, v_valid_target, m_train_input);
                       
                       fprintf('Fold %d with %d iterations. %d are wrong, %d are correct.\n', s_k, s_nbr_iter, s_nbr_error, s_nbr_correct);
                       
                       s_error = s_error + s_nbr_error;
                   end
                   
                   % Average the error
                   s_error = s_error / svm.c_s_fold_nbr;
                   m_error(i, j) = s_error;
                   
                   fprintf('Error is %d\n', s_error);
                   disp('=========');
                   
                   % store the best hyper parameter pair
                   if s_min_error > s_error
                       svm.c_s_best_C = s_C;
                       svm.c_s_best_tau = s_tau;
                       s_min_error = s_error;
                   end
               end
            
            end
            
            % show cross validation error matrix
            fig1 = figure;
            sb_1 = subplot(1,1,1);
            imagesc(m_error);
            
            % Do final training
            m_kernel = exp(-svm.c_s_best_tau * svm.c_m_A);
            
            o_best_classifier = SMOClassifier(m_kernel, svm.c_v_train_targets, svm.c_s_best_C, svm.c_s_best_tau, 1);
            
            s_nbr_iter = o_best_classifier.doTraining();
            
            % report training error
            [s_nbr_error_train, s_nbr_correct_train] = o_best_classifier.getTrainingError();
            
            % report test error
            [s_nbr_error_test, s_nbr_correct_test] = o_best_classifier.classify(svm.c_m_test, svm.c_v_test_targets, svm.c_m_train);
            
            % display result
            fprintf('Training, #iterations: %d\n', s_nbr_iter);
            fprintf('Training: %d wrong, %d correct\n', s_nbr_error_train, s_nbr_correct_train);
            fprintf('Test: %d wrong, %d correct\n', s_nbr_error_test, s_nbr_correct_test);
            
            % display bar chart of the test error / training error in
            % percentage
            end
        
        function [s_C, s_tau] = getBestParameters(svm)
            s_C = svm.c_s_best_C;
            s_tau = svm.c_s_best_tau;
        end
        
        function [o_classifier, s_nbr_iter] = trainClassifier(svm, s_C, s_tau, v_target, m_kernel, b_figure)
            
            o_classifier = SMOClassifier(m_kernel, v_target, s_C, s_tau, b_figure);
                       
            s_nbr_iter = o_classifier.doTraining();
        end
    end
    
    
    methods(Access='private')
        
        function m_A = getNormDiff(svm, m_input)
            
            inputSize = size(m_input, 1);

            % Compute [||x_i||^2]
            d = sum(m_input .* m_input, 2);

            o = ones(inputSize, 1);

            m_A = 1/2 * d * o' + 1/2 * o * d' - m_input * m_input';
        end
        
        function [m_train_input, m_train_target, m_valid_input, m_valid_target] = getTrainTestKernel(svm, fold)
           mask = ones(svm.c_s_nbr_input_train, 1);
           mask((fold * svm.c_s_fold_size + 1):((fold+1)* svm.c_s_fold_size),1) = 0;
           
           % use mask to generate the current validation and training set;
           m_train_input = svm.c_m_train(mask == 1,:);
           m_train_target = svm.c_v_train_targets(:,mask == 1);
           m_valid_input = svm.c_m_train(mask == 0,:);
           m_valid_target = svm.c_v_train_targets(:,mask == 0);
        end
    end
    
end