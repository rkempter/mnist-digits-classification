classdef SMOClassifier < handle

    properties(GetAccess = 'private', SetAccess = 'private')
        
        % kernel matrix. [ c_v_col(1), c_v_col(2), ..., c_v_col(n) ]
        % kernel matrix has dimension n^2
        c_m_kernel
        
        % target is a row vector equal to [c_s_target_1, c_s_target_2, ...
        % v_target(n)]
        c_v_target
        
        % alpha is a row vector contains n different scalars c_s_alpha_i
        c_v_alpha
        
        % f function vector is a row vector with scalars equal to
        % f_i = sum_j^n c_v_alpha(j) * c_v_target(j) * c_m_kernel(i,j) -
        % c_v_tatget(i)
        c_v_f
        
        % Index sets are three rows for I_+, I_-, I_0, each having n
        % booleans. True if index is in set, False otherwise.
        c_m_I
        
        % bias parameter b
        c_s_bias = 0;
        
        % Hyperparameter C defining the box constraint
        c_s_C
        
        % Hyperparameter tau used for precision measure in equality
        c_s_tau
        
        % size of alpha
        c_s_n
        
        % boolean to check if figure should be shown;
        c_b_show_figure
        
        % termination constant
        c_s_termination = 1e-8
    end
    
    methods
        % SMOClassifier constructor
        function classifier = SMOClassifier(m_kernel, v_targets, s_C, s_tau, b_figure) 
            % defining class variables
            classifier.c_m_kernel = m_kernel;
            
            classifier.c_v_target = v_targets;
            
            classifier.c_s_C = s_C;
            
            classifier.c_s_tau = s_tau;
            
            classifier.c_b_show_figure = b_figure;
            
            % number of training points. v_targets has one row and n
            % columns.
            classifier.c_s_n = numel(v_targets);
            
            % Initialize c_v_alpha with zeros
            classifier.c_v_alpha = zeros(1, classifier.c_s_n);
            
            % Initialize the f-vector array with the targets
            classifier.c_v_f = -v_targets;
            
            % compute the index sets
            % I_+
            classifier.c_m_I(1,:) = (v_targets == 1);
            % I_-
            classifier.c_m_I(2,:) = (v_targets == -1);
            % I_0
            classifier.c_m_I(3,:) = zeros(1, classifier.c_s_n);
        end
        
        function s_nbr_iter = doTraining(classifier)
            
            s_nbr_iter = 0;
            
            if classifier.c_b_show_figure == 1
                fig1 = figure;
                set(fig1,'visible','off');
                % subplot 1
                sb_1 = subplot(2,1,1);
                title('SVM Criterion Evolution');
                hold all
               
                % subplot 2
                sb_2 = subplot(2,1,2);
                title('Termination Criterion Evolution');
                hold all
                
                s_last_crit = 0;
                s_last_diff = 0;
            end
            
            while 1
                
                [b_low, b_up] = classifier.computeB();
                
                [s_i, s_j] = classifier.getMostViolatedPair(b_low, b_up);
                
                % Check if termination conditions are fulfilled. If true,
                % stop training.
                if (s_i == -1 && s_j == -1) || s_nbr_iter >= 10000000
                    
                    classifier.c_s_bias = (b_low + b_up) / 2;
                    break;
                end
                
                % plot every 20 iterations
                if mod(s_nbr_iter,20) == 0 && s_nbr_iter > 0 && classifier.c_b_show_figure
                    % plot termination criterion
                    s_last_diff = classifier.plotTerminationCriterion(b_low, b_up, s_nbr_iter, sb_2, s_last_diff);
                    
                    % plot svm criterion
                    s_last_crit = classifier.plotSVMCriterion(sb_1, s_nbr_iter, s_last_crit);
                end
                
                % s_sigma is element of {-1, 1}
                s_sigma = classifier.c_v_target(s_i) * classifier.c_v_target(s_j);
                
                % store old alphas
                v_alpha_tilde = [classifier.c_v_alpha(s_i), classifier.c_v_alpha(s_j)];
                
                % compute omega
                s_omega = v_alpha_tilde(1) + s_sigma * v_alpha_tilde(2);
                
                [L, H] = classifier.computePossibleRange(s_sigma, s_omega);
                v_rangeBoundries = [L, H];
                
                % theta
                s_theta = classifier.c_m_kernel(s_i, s_i) + classifier.c_m_kernel(s_j, s_j) - 2 * classifier.c_m_kernel(s_i, s_j);
                
                if s_theta > 10^-15
                    % do normal minimization
                    s_alpha_j_new = classifier.computeMin(s_theta, v_rangeBoundries, v_alpha_tilde, s_i, s_j);
                else
                    % do special case minimzation
                    s_alpha_j_new = classifier.computeMinSpecial(v_alpha_tilde, v_rangeBoundries, s_omega, s_sigma, s_i, s_j);
                end
                
                % compute new alpha_i
                s_alpha_i_new = s_omega - s_sigma * s_alpha_j_new;
                 
                % do update of alpha
                classifier.c_v_alpha(s_i) = s_alpha_i_new;
                classifier.c_v_alpha(s_j) = s_alpha_j_new;
                
                % update f_i:
                classifier.c_v_f = classifier.c_v_f + ((s_alpha_i_new - v_alpha_tilde(1)) * classifier.c_v_target(s_i) * classifier.c_m_kernel(:, s_i))';
                classifier.c_v_f = classifier.c_v_f + ((s_alpha_j_new - v_alpha_tilde(2)) * classifier.c_v_target(s_j) * classifier.c_m_kernel(:, s_j))';
                
                % update index set (only the specific columns at i, resp. j
                % are updated.
                v_index_set_i = classifier.updateIndexSet(s_alpha_i_new, classifier.c_v_target(s_i));
                v_index_set_j = classifier.updateIndexSet(s_alpha_j_new, classifier.c_v_target(s_j));
                classifier.c_m_I(:, s_i) = v_index_set_i;
                classifier.c_m_I(:, s_j) = v_index_set_j;
                
                % increment iteration number
                s_nbr_iter = s_nbr_iter + 1;
            end
        
            % show figure after training
            if classifier.c_b_show_figure == 1
                set(fig1,'visible','on');
            end
        end
        
        % Compute training error
        function [s_nbr_error, s_nbr_correct] = getTrainingError(classifier)
            % todo: check if already trained!
            
            v_output = sign(classifier.c_m_kernel * (classifier.c_v_alpha .* classifier.c_v_target)'- classifier.c_s_bias);
            
            v_check = v_output .* classifier.c_v_target';
           
            s_nbr_error = sum(v_check < 0);
            s_nbr_correct = sum(v_check == 1);
        end
        
        % Use the trained classifier to classify a set of input vectors.
        % Returns the number of wrong and correctly classified patterns
        function [s_nbr_error, s_nbr_correct] = classify(classifier, m_input, v_target, m_training)
           
            % need to compute new kernel
            v_input_norm = sum(m_input .* m_input,2);
            v_train_norm = sum(m_training .* m_training,2);
            
            s_size_input = size(v_input_norm,1);
            s_size_train = size(v_train_norm,1);
            
            m_A = (1/2) * ones(s_size_input, 1) * v_train_norm' + (1/2) * v_input_norm * ones(1, s_size_train) - m_input * m_training';
            
            m_K = exp(-classifier.c_s_tau * m_A);
            
            v_output = sign(m_K * (classifier.c_v_alpha .* classifier.c_v_target)'- classifier.c_s_bias);
            
            v_check = v_output .* v_target';
           
            s_nbr_error = sum(v_check < 0);
            s_nbr_correct = sum(v_check == 1);
        end
    end
    
    methods(Access='private')
        
        % compute termination criterion for plot
        function s_b_diff = plotTerminationCriterion(classifier, s_b_low, s_b_up, s_iter, subplot, s_last)
            
            s_b_diff = abs(s_b_low - s_b_up);
            
            if s_b_low <= s_b_up + 2*classifier.c_s_termination
                s_b_diff = 0;
            end
            
            % plot termination criterion on subplot
            plot(subplot, [s_iter-20, s_iter], [s_last, s_b_diff], 'Color', 'b');
        end
        
        % The SVM criterion should go up with every iteration.
        % MAXIMIZATION.
        function s_svm_crit = plotSVMCriterion(classifier, subplot, s_iter, s_last)
            % compute SVM criterion
            s_svm_crit = sum(classifier.c_v_alpha) - 1 / 2 * classifier.c_v_alpha * diag(classifier.c_v_target) * classifier.c_m_kernel * diag(classifier.c_v_target) * classifier.c_v_alpha';
            
            % plot the criterion on subplot
            plot(subplot, [s_iter-20, s_iter], [s_last, s_svm_crit], 'Color', 'b');
        end
        
        % computes the new index set of a specific index element. Returns a
        % column-vector with elements [i_plus, i_negative, i_0]
        function col = updateIndexSet(classifier, new_alpha, target) 
           i_plus = (target * new_alpha == 0 && target == 1) || (target * new_alpha == -classifier.c_s_C);
           i_minus = (target * new_alpha == 0 && target == -1) || (target * new_alpha == classifier.c_s_C);
           i_0 = (new_alpha > 0 && new_alpha < classifier.c_s_C);
           col = [i_plus; i_minus; i_0];
        end
        
        % compute the minimum possible new alpha_j. Clip to range if
        % necessary
        function s_alpha_new = computeMin(classifier, s_theta, v_boundries, v_alpha_tilde, s_i, s_j)
           s_alpha_new = v_alpha_tilde(2) + classifier.c_v_target(s_j) * (classifier.c_v_f(s_i) - classifier.c_v_f(s_j)) / s_theta;
           
           if s_alpha_new < v_boundries(1)
               s_alpha_new = v_boundries(1);
           elseif s_alpha_new > v_boundries(2)
               s_alpha_new = v_boundries(2);
           end
        
        end
        
        % if theta is smaller than 10^-15, we need to compute the new alpha
        % in a different way because of possible numerical errors (division
        % by 0).
        function s_alpha_new = computeMinSpecial(classifier, v_alpha_tilde, v_boundries, s_omega, s_sigma, s_i, s_j)
           
            % anonymous method to compute L_i or H_i
            f_alpha_i = @(x) s_omega - s_sigma * x;
            
            % anonymous method for computation of v:
            f_v = @(x) classifier.c_v_f(x) + classifier.c_v_target(x) - v_alpha_tilde(1) * classifier.c_v_target(s_i) * classifier.c_m_kernel(s_i, x) - v_alpha_tilde(2) * classifier.c_v_target(s_j) * classifier.c_m_kernel(x, s_j);
            
            % Anonymous method for phi computation. Phi is evaluated at 
            % both L and H.
            f_phi = @(x) 1 / 2 * (classifier.c_m_kernel(s_i, s_i) * f_alpha_i(x)^2 + classifier.c_m_kernel(s_j, s_j) * x^2) + s_sigma * classifier.c_m_kernel(s_i, s_j) * f_alpha_i(x) * x + classifier.c_v_target(s_i) * f_alpha_i(x) * f_v(s_i) + classifier.c_v_target(s_j) * x * f_v(s_j) - f_alpha_i(x) - x;
         
            s_phi_L = f_phi(v_boundries(1));
            s_phi_H = f_phi(v_boundries(2));
            
            if s_phi_L < s_phi_H
                s_alpha_new = v_boundries(1);
            else
                s_alpha_new = v_boundries(2);
            end
        end
        
        % method computes b_low and b_up.
        function [b_low, b_up] = computeB(classifier)
            % get I_up and I_low
            v_I_up = classifier.c_m_I(1,:) | classifier.c_m_I(3,:);
            v_I_low = classifier.c_m_I(2,:) | classifier.c_m_I(3,:);
           
            % compute b_low and b_up
            b_up = min(classifier.c_v_f(v_I_up));
            b_low = max(classifier.c_v_f(v_I_low));
        end
        
        % returns the index pair that are most violated
        % In case the termination condition is fulfilled, this method
        % returns [-1, -1]
        function [s_i, s_j] = getMostViolatedPair(classifier, b_low, b_up)
            % check termination condition. if true, return [-1, -1]
            % otherwise, return indices of location of b_low and b_up
            if b_low <= b_up + 2*classifier.c_s_termination
                s_i = -1;
                s_j = -1;
            else
                s_i = find(classifier.c_v_f == b_low, 1);
                s_j = find(classifier.c_v_f == b_up, 1);
            end
        end
       
        % Method computes the possible range boundries for alpha_j while
        % alpha_i is element of [0, C]. Alpha_j is element of [L, H]
        function [L, H] = computePossibleRange(classifier, s_sigma, s_omega)
           L = max(0, s_sigma * s_omega - (1 + s_sigma) / 2 * classifier.c_s_C);
           H = min(classifier.c_s_C, s_sigma * s_omega + (1-s_sigma) / 2 * classifier.c_s_C);
        end     
    end

end