function [eigvec,FctValSeq] = computeEigenvectorGeneral_vc(W,start,normalized,verbose,deg,vertex_cut)
% Computes a nonconstant eigenvector of the 1-Laplacian using the
% nonlinear inverse power method.
%
% Usage:
%   [eigvec,FctValSeq] = computeEigenvectorGeneral_vc(W,start,normalized,verbose,deg)
%
% Input:
%   W           - Sparse symmetric weight matrix.
%   start       - Start vector. Use multiple runs with random initialization.
%   normalized  - True/false for normalized/unnormalized 1-spectral clustering.
%   deg         - Degrees of vertices as column vector. Default is 
%                 full(sum(W,2)) in normalized case and ones(size(W,1),1) 
%                 in unnormalized case. Will be ignored if normalized=false.
%
% Output:
%   eigvec      - Final eigenvector.
%   FctValSeq   - Sequence of functional values in each iteration.
%
% (C)2010-14 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    if (nargin<4)
        verbose = true;
    end
    if (nargin<5)
        if (normalized)
            deg = full(sum(W,2));
        else
            deg = ones(size(W,1),1);
        end
    else
       deg = full(deg); 
    end

    assert(isnumeric(W) && issparse(W),'Wrong usage. W should be sparse and numeric.');

    [ix,jx,wval] = find(W);
    maxiter = 100;

    counter = 0;

    FctValOld = inf;
    FctValSeq = [];
    fnew = start;

    while(counter<maxiter)

        % Subtract median/weighted median
        fnew = performCentering(fnew, normalized, deg);

        % compute functional (assumes that weighted median/median is zero).
        FctVal = evaluateFunctional(fnew, wval, ix, jx, normalized, deg, W, vertex_cut);

        fold = fnew;
        FctValSeq = [FctValSeq,FctVal];
        FctValOld = FctVal;

        if(verbose)
            if( ~vertex_cut)
                [ac,cut,cheeger] = createClustersGeneral(fold, W, normalized, -1, 2, deg);
            else 
                [ac, cheeger] = opt_thresh_vertex_expansion(x, params, normalized);
            end
            disp(['****Iter: ',num2str(counter),' - Functional: ',num2str(FctVal,'%1.16f'),'- CheegerBest: ',num2str(cheeger,'%1.14f')']);
            disp([' ']);
        end

        % compute subgradient of denominator (the same for cheeger cut and
        % vertex exp). assumes fold has median/weighted median zero
        subgrad = computeSubGradient(fold, normalized, deg);


        % Solve inner problem
        if (vertex_cut)
            start = randn(length(fold),1);%fold;
            params.c2 = -FctVal*subgrad;
            params.W = W;
            eps1 = 1E-3;
            obj_subg = @(x,params) (obj_subg_vertex_exp(x,params));
            verbose = true;
        else
            start = fold;
            params.c2 = -FctVal*subgrad;
            params.wval = wval;
            params.ix = ix;
            params.jx = jx;
            eps1 = 1E-3;
            obj_subg = @(x,params) (obj_subg_cheeger(x,params));
            verbose = false;
        end
        
        
        [fnew,Obj,cur_delta,it,toc1] = ip_bundle(start,params,eps1,'bundle_level',verbose,obj_subg);
               
        verbose = false;

        if(verbose)
            disp(['Objective Bundle method:',num2str(Obj),' time: ', num2str(toc1), ' it: ', num2str(it) ]);
        end

        if (Obj>-1E-11)
            fnew = fold;
            counter = maxiter; %% stopping criterion
        end
 

        %diffFunction = min(norm(fnew-fold),norm(fnew+fold));
        counter = counter+1;
    end


    % Subtract median
    fnew = performCentering(fnew, normalized, deg);

    % compute most recent Functional value (assumes that fnew has median 0)
    FctVal = evaluateFunctional(fnew, wval, ix, jx, normalized, deg, W, vertex_cut);

    if(FctVal<FctValOld)
        fold = fnew;
        FctValSeq = [FctValSeq,FctVal];
    else
        FctVal = FctValOld;
    end

   % print final objective
   if(verbose)
        if( ~vertex_cut)
            [ac,cut,cheeger] = createClustersGeneral(fold, W, normalized, -1, 2, deg);
        else 
            [ac, cheeger] = opt_thresh_vertex_expansion(x, params, normalized);
            cut = cheeger;
        end
        fprintf('......... Final result: Functional: %.16g  - Cut: %.14g - Cheeger Cut : %.14g \n', FctVal, cut, cheeger);
    end
    eigvec = fold;
end

% subtract median/weighted median
function fnew = performCentering(fnew, normalized, deg)
    if (normalized)
       fnew = fnew - weighted_median(fnew,deg);
    else
       fnew = fnew - median(fnew);
    end
    fnew = fnew/norm(fnew,1);
end

% evaluate the objective function
% assumes fold has (weighted) median 0.
function FctVal = evaluateFunctional(f, wval, ix, jx, normalized, deg, W, vertex_cut)
    if (~vertex_cut)
        sval = wval.*abs(f(ix)-f(jx));
        R = 0.5*sum(sval);
    else
        num = length(f);
        params.W = W;
        [RCk_sort,sort_ind] = thresholds_vertex_cut_fast(f,params);
        RCk_sort_shift = [RCk_sort(2:end); 0];

        subg = zeros(num,1);
        subg(sort_ind) = RCk_sort-RCk_sort_shift;

        R = subg'*f;
    end
    if (~normalized)
        FctVal = R/norm(f,1);
    else
        FctVal = R/(deg'*abs(f));
    end
end

% compute the subgradient of the denominator
% (the same for cheeger cut and vertex exp). 
% assumes fold has median/weighted median zero
function subgrad = computeSubGradient(fold, normalized, deg)
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
            subgrad(ixNull) = -diffPosNeg/length(ixNull);
        end
    end
end
