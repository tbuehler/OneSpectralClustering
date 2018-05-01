function [eigvec,FctValSeq]=computeEigenvectorGeneral(W,start,normalized,crit,verbose,deg)
% Computes a nonconstant eigenvector of the 1-Laplacian using the 
% nonlinear inverse power method.
%
% Usage:
%   [eigvec,FctValSeq]= computeEigenvectorGeneral(W,start,normalized,crit,verbose,deg)
%
% Input:
%   W           - Sparse symmetric weight matrix.
%   start       - Start vector. Use multiple runs with random initialization.
%   normalized  - True/false for normalized/unnormalized 1-spectral clustering.
%   crit        - 1: Solve eigenproblem associated with Rcut/Ncut
%                 2: Solve eigenproblem associated with RCC/NCC     
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
% Copyright 2010-18 Thomas Bühler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    if (nargin<5)
        verbose = true;
    end
    if nargin<6
        if (normalized)
            deg = full(sum(W,2));
        else
            deg = ones(size(W,1),1);
        end
    else
       deg = full(deg); 
    end
        
    assert(crit==1 || crit ==2, 'Wrong usage. crit has to be 1 (Rcut/Ncut) or 2 (RCC/NCC)');
    assert(isnumeric(W) && issparse(W),'Wrong usage. W should be sparse and numeric.');

    [ix,jx,wval] = find(W);
    W2 = triu(W,1);         % diagonal part plays no role in inner problem
    L = 2*max(sum(W.^2));   % upper bound on Lipschitz constant

    maxiter_inner = 40;     % number of inner iterations in first try
    maxiter_bound = 5120;   % maximum number of inner iterations
    maxiter_outer = 100;    % maximum number of outer iterations
    epsilon = 1E-14; 

    counter = 0;
    alphaold = zeros(length(ix),1); 
    fold = start;
    fold = performCentering(fold, normalized, deg);
    [FctValOld, subgrad_old] = evaluateFunctional(fold, wval, ix, jx, normalized, deg, crit);
    FctValRatio = 0.0;
    FctValSeq = [FctValOld];
    
    % print value of outer objective and cut values at starting point
    if(verbose)
        [ac,cut,cheeger] = createClustersGeneral(fold,W,normalized,-1,crit,deg);
        fprintf('......... Init - Functional: %.14g - CutBest: %.14g - CheegerBest: %.14g\n', FctValOld, cut, cheeger);
    end

    % main loop
    while(counter<maxiter_outer && maxiter_inner<=maxiter_bound)
        
        % solve inner problem
        [fnew,alphaold,Obj,niter] = solveInnerProblem(W2,FctValOld*full(subgrad_old),FctValRatio*alphaold,maxiter_inner,epsilon,L);
        
        % if the objective is zero, fold was an eigenvector and we can stop
        if (Obj==0) 
            fnew = fold;
            counter = maxiter_outer;
        end
        
        % print value of primal and dual inner objective 
        if(verbose)
            fprintf('......... Inner Problem - Final Obj: %.16g  - Number of Iterations: %d \n', Obj, niter);
            f3 = fnew/norm(fnew); 
            sval3 = wval.*abs(f3(ix)-f3(jx)); 
            Obj2 = 0.5*sum(sval3)-FctValOld*subgrad_old'*f3;
            fprintf('......... Original Obj: %.16g - Zeros: %d - Balance: %.15g \n', Obj2, sum(fold==0), sum(sign(fold)) );
        end
        
        % subtract median or weighted median and normalize
        fnew = performCentering(fnew, normalized, deg);
        fnew = fnew/norm(fnew,1);
        diffFunction = min(norm(fnew-fold),norm(fnew+fold));
        counter = counter+1;
                 
        % compute current functional value and subgradient of denominator
        [FctValNew, subgrad_new] = evaluateFunctional(fnew, wval, ix, jx, normalized, deg, crit);
  	    
        % update bounds for maximum inner iterations
        [maxiter_bound, maxiter_inner] = update_maxiter(FctValNew,FctValOld, maxiter_bound, maxiter_inner,counter, verbose);
        
        % if functional has decreased, update iterates, otherwise we will 
        % continue optimizing the inner problem 
        if(FctValNew<FctValOld)
            FctValRatio = FctValNew/FctValOld;
            FctValOld = FctValNew;
            FctValSeq = [FctValSeq,FctValNew];
            fold = fnew;
            subgrad_old = subgrad_new;
        else
            FctValRatio = 1.0;
        end
        
        % print value of outer objective and current cut values
        if (verbose)
            if(maxiter_inner>maxiter_bound)
                fprintf('......... Reached maximum number of inner iterations: %d\n', maxiter_bound); 
            end
            [ac,cut,cheeger] = createClustersGeneral(fold,W,normalized,-1,crit,deg);
            fprintf('......... Iter: %d - Functional: %.14g - CutBest: %.14g - CheegerBest: %.14g - DiffF: %.14g\n', counter, FctValOld, cut, cheeger, diffFunction);
        end
    end

    % print final objective
    if(verbose)
        [ac,cut,cheeger] = createClustersGeneral(fold,W,normalized,-1,crit,deg);
        fprintf('......... Final result: Functional: %.16g  - Cut: %.14g - Cheeger Cut : %.14g \n', FctValOld, cut, cheeger);
    end
    eigvec=fold;
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
        if(verbose)
            fprintf('......... Functional has not decreased. Old: %.14g - New: %.14g. Increasing number of inner iterations.\n', FctValOld, FctValNew);
        end
        maxiter_inner = maxiter_inner*2;
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
function [FctVal, subgrad] = evaluateFunctional(f, wval, ix, jx, normalized, deg, crit)
    sval = wval.*abs(f(ix)-f(jx));
    subgrad = computeSubGradient(f, normalized, deg, crit);
    FctVal = 0.5*sum(sval) /(f'*subgrad);
end

% compute the subgradient of the denominator
function subgrad = computeSubGradient(fold, normalized, deg, crit)        
    if (crit==1) 
        % compute the subgradient associated with NCut/Rcut problem
        if (normalized)
            [fsort, ind] = sort(fold);
            VolV= sum(deg);
            deg_sort=deg(ind);
            vec=zeros(size(fold,1),1);
            vec(ind) = deg_sort.*(2 * cumsum(deg_sort)- VolV -deg_sort);
            subgrad=vec/VolV;
        else
            [fsort, ind] = sort(fold);
            num = size(fold,1);
            vec=zeros(num,1);
            vec(ind) = 2*(1:num)-num-1;
            subgrad=vec/num;
        end    
    elseif (crit==2) 
        % compute the subgradient associated with NCC/RCC problem
        % note that here we assume that fold has been centered before
        if (normalized)
            ixNull = find(fold==0);
            Null = sum(deg(ixNull));
            subgrad = deg.*sign(fold);
            if(Null>0)
                diffPosNeg = sum(deg(fold>0))-sum(deg(fold<0));
                subgrad(ixNull) = -deg(ixNull)* diffPosNeg/Null;
            end
        else
            ixNull = find(fold==0);
            Null = length(ixNull);
            subgrad = sign(fold);
            if(Null>0)
                diffPosNeg = sum(fold>0)-sum(fold<0);
                subgrad(ixNull) = -diffPosNeg/Null;
            end
        end
    end
end
