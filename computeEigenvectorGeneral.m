function [eigvec,FctValSeq] = computeEigenvectorGeneral(W,start,normalized,crit,verbose,deg)
% Computes a nonconstant nonlinear eigenvector using the inverse power method
% (IPM) for nonlinear eigenproblems, as described in the paper
%
% M. Hein and T. Bühler
% An Inverse Power Method for Nonlinear Eigenproblems with Applications in 1-Spectral Clustering and Sparse PCA
% In Advances in Neural Information Processing Systems 23 (NIPS 2010)
% Available online at http://arxiv.org/abs/1012.0774
%
% Usage:
%   [eigvec,FctValSeq] = computeEigenvectorGeneral(W,start,normalized,crit,verbose,deg)
%
% Input:
%   W           - Sparse symmetric weight matrix.
%   start       - Start vector. Use multiple runs with random initialization.
%   normalized  - True/false for normalized/unnormalized 1-spectral clustering.
%   crit        - 1: Solve eigenproblem associated with Rcut/Ncut
%                 2: Solve eigenproblem associated with RCC/NCC     
%                 3: Solve eigenproblem associated with SVE/NVE 
%   verbose     - If true, log info to console.
%   deg         - Degrees of vertices as column vector. Default is 
%                 full(sum(W,2)) in normalized case and ones(size(W,1),1) 
%                 in unnormalized case. Will be ignored if normalized=false.
%
% Output:
%   eigvec      - Final eigenvector.
%   FctValSeq   - Sequence of functional values in each iteration.
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Copyright 2010-2020 Thomas Bühler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    if (nargin<5)
        verbose = true;
    end
    if (nargin<6)
        if (normalized)
            deg = full(sum(W,2));
        else
            deg = ones(size(W,1),1);
        end
    else
       deg = full(deg); 
    end
        
    assert(crit==1 || crit==2 || crit==3, 'Wrong usage. crit has to be between 1 and 3');
    assert(isnumeric(W) && issparse(W),'Wrong usage. W should be sparse and numeric.');

    [ix,jx,wval] = find(W);
    maxiter = 100;          % maximum number of outer iterations
    counter = 0;
    fold = start;
    fold = performCentering(fold, normalized, deg);
    if (crit==3); fold = fold/norm(fold,1); end
    [FctValOld, subgrad_old] = evaluateFunctional(fold, wval, ix, jx, normalized, deg, crit, W);
    FctValSeq = FctValOld;
    FctValRatio = 0.0;

    W2 = triu(W,1);         % diagonal part plays no role in inner problem
    L = 2*max(sum(W.^2));   % upper bound on Lipschitz constant
    alphaold = zeros(nnz(W2),1); 

    % print value of outer objective and cut values at starting point
    if(verbose)
        if (crit<3)
            [~, cut, cheeger] = createClustersGeneral(fold, W, normalized, -1, crit, deg);
            fprintf('......... Init - Functional: %.14g - CutBest: %.14g - CheegerBest: %.14g\n', ...
                    FctValOld, cut, cheeger);
        else 
            [~, vertex_exp] = opt_thresh_vertex_expansion(fold, W, normalized);
            fprintf('......... Init - Functional: %.14g - Vertex Expansion: %.14g\n', ...
                    FctValOld, vertex_exp);
        end
    end

    maxiter_inner = 40;     % number of inner iterations in first try
    maxiter_bound = 5120;   % maximum number of inner iterations

    % main loop
    while(counter<maxiter)
        
        [fnew, subgrad_new, FctValNew, counter, diffFunction, alphaold, maxiter_inner, maxiter_bound] = solveInnerProblem(...
            FctValOld, subgrad_old, fold, wval, ix, jx, normalized, deg, crit, maxiter, counter, verbose, W, W2, ...
            FctValRatio, alphaold, L, maxiter_inner, maxiter_bound);

        % if functional has decreased, update iterates, otherwise we will 
        % continue optimizing the inner problem 
        if(FctValNew<FctValOld)
            FctValRatio = FctValNew/FctValOld; %used by ncut/ncc
            FctValOld = FctValNew;
            FctValSeq = [FctValSeq,FctValNew];
            fold = fnew;
            subgrad_old = subgrad_new;
        else
            FctValRatio = 1.0;
        end

        % print value of outer objective and current cut values
        if (verbose)
            if (crit<3)
                [~, cut, cheeger] = createClustersGeneral(fold,W,normalized,-1,crit,deg);
                fprintf('......... Iter: %d - Functional: %.14g - CutBest: %.14g - CheegerBest: %.14g - DiffF: %.14g\n', ...
                        counter, FctValOld, cut, cheeger, diffFunction);
            else 
                [~, vertex_exp] = opt_thresh_vertex_expansion(fold, W, normalized);
                fprintf('......... Iter: %d - Functional: %.14g - Vertex Expansion: %.14g - DiffF: %.14g\n', ...
                        counter, FctValOld, vertex_exp, diffFunction);
            end
        end
    end

   % print final objective
   if(verbose)
       if (crit<3)
           [~, cut, cheeger] = createClustersGeneral(fold, W, normalized, -1, crit, deg);
           fprintf('......... Final result: Functional: %.16g  - Cut: %.14g - Cheeger Cut : %.14g \n', ...
                   FctValOld, cut, cheeger);
       else 
           [~, vertex_exp] = opt_thresh_vertex_expansion(fold, W, normalized);
           fprintf('......... Final result: Functional: %.16g  - Vertex Expansion: %.14g \n', ...
                   FctValOld, vertex_exp);
       end
   end
   eigvec = fold;
end


function [fnew, subgrad_new, FctValNew, counter, diffFunction, alphaold, maxiter_inner, maxiter_bound] = solveInnerProblem(...
             FctValOld, subgrad_old, fold, wval, ix, jx, normalized, deg, crit, maxiter, counter, verbose, W, W2, ...
             FctValRatio, alphaold, L, maxiter_inner, maxiter_bound)

    % solve inner problem
    if (crit==3)
        start = randn(length(fold),1);%fold;
        params.c2 = -FctValOld*subgrad_old;
        params.W = W;
        eps1 = 1E-3;
        obj_subg = @(x,params) (obj_subg_vertex_exp(x,params));
        % [f_new, Obj, niter] = ip_cutting_plane(start, params, eps1, 'linprog', obj_subg);
        [fnew, Obj, niter] = ip_bundle_level(start, params, eps1, 'linprog', 'fista_mex', verbose, obj_subg);
    else
        epsilon = 1E-14; 
        [fnew, alphaold, Obj, niter] = mex_solve_inner_problem(W2, FctValOld*full(subgrad_old), ...
                                                     FctValRatio*alphaold, maxiter_inner, epsilon, L);
    end

    % if the objective is zero, fold was an eigenvector and we can stop
    if ((crit==3 && Obj>-1E-11) || (crit<3 && Obj==0))
        fnew = fold;
        counter = maxiter;
    else
        counter = counter+1;
    end

    % print value of inner objective 
    if(verbose)
        fprintf('......... Inner Problem - Final Obj: %.14g - Number of Iterations: %d\n', Obj, niter);
        if (crit<3)
            f3 = fnew/norm(fnew); 
            sval3 = wval.*abs(f3(ix)-f3(jx)); 
            Obj2 = 0.5*sum(sval3)-FctValOld*subgrad_old'*f3;
            fprintf('......... Original Obj: %.16g - Zeros: %d - Balance: %.15g \n', ...
                    Obj2, sum(fold==0), sum(sign(fold)) );
        end
    end

    % subtract median or weighted median and normalize
    fnew = performCentering(fnew, normalized, deg);
    fnew = fnew/norm(fnew,1);
    diffFunction = min(norm(fnew-fold), norm(fnew+fold));

    % compute current functional value and subgradient of denominator
    [FctValNew, subgrad_new] = evaluateFunctional(fnew, wval, ix, jx, normalized, deg, crit, W);

    if (crit<3)
        % update bounds for maximum inner iterations
        [maxiter_bound, maxiter_inner] = update_maxiter(FctValNew, FctValOld, maxiter_bound, ...
                                                        maxiter_inner, counter, verbose);
        if (maxiter_inner>maxiter_bound)
            counter = maxiter;
            if (verbose)
                fprintf('......... Reached maximum number of inner iterations: %d\n', maxiter_bound); 
            end
        end
    end
end

% update bounds for inner iterations. if the functional value has not yet 
% decreased, the current maxiter_inner is doubled. the algorithm will then 
% try again until the bound maxiter_bound is reached. this bound has its
% highest value in the beginning and is reduced at 5th and 10th iteration.
function [maxiter_bound, maxiter_inner] = update_maxiter(FctValNew, FctValOld, maxiter_bound, maxiter_inner, counter, verbose)
    if(FctValNew<FctValOld)
        if (counter>10)
            maxiter_bound = 640;  % reduce the maximum number of iterations 
        elseif (counter>5)
            maxiter_bound = 1280; % reduce the maximum number of iterations 
        end
        maxiter_inner = min(maxiter_inner,maxiter_bound);  
    else
        maxiter_inner = maxiter_inner*2;
        if(verbose)
            fprintf('......... Functional has not decreased. Old: %.14g - New: %.14g. Setting inner iterations to %d.\n', ...
                    FctValOld, FctValNew, maxiter_inner);
        end
    end
end

% subtract median/weighted median
function fnew = performCentering(fnew, normalized, deg)
    if (normalized)
       fnew = fnew - weighted_median(fnew,deg);
    else
       fnew = fnew - median(fnew);
    end
end

% evaluate the objective function
function [FctVal, subgrad] = evaluateFunctional(f, wval, ix, jx, normalized, deg, crit, W)
    if (crit<3)
        sval = wval.*abs(f(ix)-f(jx));
        subgrad = computeSubGradient(f, normalized, deg, crit);
        FctVal = 0.5*sum(sval) / (f'*subgrad);
    else
        num = length(f);
        [~,sort_ind] = sort(f);
        Wsort = W(sort_ind,sort_ind);
        Wtril = tril(Wsort,-1);
        Wtriu = triu(Wsort,1);
        RCk_sort = mex_thresholds_vertex_cut(Wtril,Wtriu);
        RCk_sort_shift = [RCk_sort(2:end); 0];
        subg = zeros(num,1);
        subg(sort_ind) = RCk_sort-RCk_sort_shift;
        subgrad = computeSubGradient(f, normalized, deg, 2);
        if (~normalized)
           FctVal = (subg'*f)/norm(f,1);
        else
           FctVal = (subg'*f)/(deg'*abs(f));
        end
    end
end

% compute the subgradient of the denominator
% (the same for cheeger cut and vertex exp). 
function subgrad = computeSubGradient(fold, normalized, deg, crit)
    if (crit==1) 
        % compute the subgradient associated with NCut/Rcut problem
        if (normalized)
            [~, ind] = sort(fold);
            VolV = sum(deg);
            deg_sort = deg(ind);
            vec = zeros(size(fold,1),1);
            vec(ind) = deg_sort.*(2 * cumsum(deg_sort) - VolV - deg_sort);
            subgrad = vec/VolV;
        else
            [~, ind] = sort(fold);
            num = size(fold,1);
            vec = zeros(num,1);
            vec(ind) = 2*(1:num)-num-1;
            subgrad = vec/num;
        end    
    elseif (crit==2) 
        % compute the subgradient associated with NCC/RCC problem
        % note that here we assume that fold has been centered before
        if (normalized)
            ixNull = find(fold==0);
            Null = sum(deg(ixNull));
            subgrad = deg.*sign(fold);
            if(Null>0)
                diffPosNeg = sum(deg(fold>0)) - sum(deg(fold<0));
                subgrad(ixNull) = -deg(ixNull) * diffPosNeg/Null;
            end
        else
            ixNull = find(fold==0);
            Null = length(ixNull);
            subgrad = sign(fold);
            if(Null>0)
                diffPosNeg = sum(fold>0) - sum(fold<0);
                subgrad(ixNull) = -diffPosNeg/Null;
            end
        end
    end
end
