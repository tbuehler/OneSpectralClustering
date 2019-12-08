function [fold,FctValOuter]=computeEigenvectorGeneral_vc(W,fold,normalized,verbose,deg,vertex_cut)
% Computes a nonconstant eigenvector of the 1-Laplacian using the
% nonlinear inverse power method.
%
% Usage:
%   [fold,FctValOuter]=
%   computeEigenvectorGeneral_vc(W,fold,normalized,verbose,deg)
%
% Input:
%   W           - Sparse symmetric weight matrix.
%   fold        - Start vector. Use multiple runs with random initialization.
%   normalized  - True/false for normalized/unnormalized 1-spectral clustering.
%
%   deg         - Degrees of vertices as column vector. Default is
%                 full(sum(W,2)) in normalized case and ones(size(W,1),1)
%                 in unnormalized case. Will be ignored if normalized=false.
%
% Output:
%	fold        - Final eigenvector.
%   FctValOuter - Values of the functional in each iteration.
%
% (C)2010-14 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    if(nargin<4)
        verbose=true;
    end
    if nargin<5
        if (normalized)
            deg=full(sum(W,2));
        else
            deg=ones(size(W,1),1);
        end
    else
        deg=full(deg);
    end


    assert(isnumeric(W) && issparse(W),'Wrong usage. W should be sparse and numeric.');

    [ix,jx,wval]=find(W);
    maxiterations = 100;

    counter=0;

    FctValOld=inf;
    FctValOuter=[];
    fnew=fold;

    while(counter<maxiterations)

        % Subtract median/weighted median
        fnew=subtract_median(fnew,normalized, deg);

        % compute functional (assumes that weighted median/median is zero).
        FctVal=Functional(fnew,normalized,deg,wval,ix,jx,W,vertex_cut);


        fold = fnew;
        FctValOuter=[FctValOuter,FctVal];
        FctValOld = FctVal;

        if(verbose)
            if( ~vertex_cut)
                [ac,cut,cheeger]=createClustersGeneral(fold,W,normalized,-1,2,deg);
            else 
                [ac, cheeger] = opt_thresh_vertex_expansion(x,params,normalized);
            end
            disp(['****Iter: ',num2str(counter),' - Functional: ',num2str(FctVal,'%1.16f'),'- CheegerBest: ',num2str(cheeger,'%1.14f')']);
            disp([' ']);
        end

        % compute subgradient of denominator (the same for cheeger cut and
        % vertex exp). assumes fold has median/weighted median zero
        vec=subgradient_denom(fold,normalized,deg);


        % Solve inner problem
        if (vertex_cut)
            start=randn(length(fold),1);%fold;
            params.c2=-FctVal*vec;
            params.W=W;
            eps1=1E-3;
            obj_subg= @(x,params) (obj_subg_vertex_exp(x,params));
            verbose=true;
        else
            start=fold;
            params.c2=-FctVal*vec;
            params.wval=wval;
            params.ix=ix;
            params.jx=jx;
            eps1=1E-3;
            obj_subg= @(x,params) (obj_subg_cheeger(x,params));
            verbose=false;
        end
        
        
        [fnew,Obj,cur_delta,it,toc1] = ip_bundle(start,params,eps1,'bundle_level',verbose,obj_subg);
               
        verbose=false;

        if(verbose)
            disp(['Objective Bundle method:',num2str(Obj),' time: ', num2str(toc1), ' it: ', num2str(it) ]);
        end

        if (Obj>-1E-11)
            fnew=fold;
            counter=maxiterations; %% stopping criterion
        end
 

        %diffFunction = min(norm(fnew-fold),norm(fnew+fold));
        counter=counter+1;
    end


    % Subtract median
    fnew=subtract_median(fnew,normalized, deg);

    % compute most recent Functional value (assumes that fnew has median 0)
    FctVal=Functional(fnew,normalized,deg,wval,ix,jx,W,vertex_cut);

    if(FctVal<FctValOld)
        fold = fnew;
        FctValOuter=[FctValOuter,FctVal];
    else
        FctVal=FctValOld;
    end



   if(verbose)
        if( ~vertex_cut)
            [ac,cut,cheeger]=createClustersGeneral(fold,W,normalized,-1,2,deg);
        else 
            [ac, cheeger] = opt_thresh_vertex_expansion(x,params,normalized);
        end
        disp(['****Iter: ',num2str(counter),' - Functional: ',num2str(FctVal,'%1.16f'),'- CheegerBest: ',num2str(cheeger,'%1.14f')']);
        disp([' ']);
    end

end


% Subtract median/weighted median
function fnew=subtract_median(fnew,normalized,deg)

    if (~normalized)
        fnew = fnew - median(fnew);
    else
        fnew = fnew - weighted_median(fnew,deg);
    end
    fnew = fnew/norm(fnew,1);

end


% compute subgradient of denominator (the same for cheeger cut and
% vertex exp). assumes fold has median/weighted median zero
function vec=subgradient_denom(fold,normalized,deg)

   
    % make sure we have <vec,1>=0
    if (~normalized)
        ixNull=find(fold==0); %ixNull=ixNull(randperm(length(ixNull)));
        Pos=sum(fold>0);
        Neg=sum(fold<0);
        Null=length(ixNull);
        fcur=sign(fold);
        if(Null>0)
            diffPosNeg=Pos-Neg;
            fcur(ixNull)=-diffPosNeg/length(ixNull);
        end
        vec = fcur;
    else
        ixNull=find(fold==0); %ixNull=ixNull(randperm(length(ixNull)));
        Pos=sum(deg(fold>0));
        Neg=sum(deg(fold<0));
        Null=sum(deg(ixNull));
        fcur=deg.*sign(fold);
        if(Null>0)
            diffPosNeg=Pos-Neg;
            fcur(ixNull)= -deg(ixNull)* diffPosNeg/Null;
        end
        vec = fcur;
    end

end

% compute functional for cheeger. assumes fold has (weighted) median 0.
function FctVal= Functional(fnew,normalized,deg,wval,ix,jx,W,vertex_cut)


 

    if (~vertex_cut)
        sval = wval.*abs(fnew(ix)-fnew(jx));
        R=0.5*sum(sval);
    else
        num=length(fnew);
        params.W=W;
        [RCk_sort,sort_ind]=thresholds_vertex_cut_fast(fnew,params);
        RCk_sort_shift=[RCk_sort(2:end); 0];

        subg=zeros(num,1);
        subg(sort_ind)=RCk_sort-RCk_sort_shift;

        R=subg'*fnew;
    end
    if (~normalized)
        FctVal = R/norm(fnew,1);
    else
        FctVal = R/(deg'*abs(fnew));
    end

end
