function [RCk_sort,sort_ind] = thresholds_vertex_cut_fast(x,params)
% Performs optimal thresholding of the numerator of the unconstrained set ratio 
% for vertex expansion
%
% (C)2012-14 Thomas Buehler
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de


    A=params.W;

    [~,sort_ind]=sort(x);
    A=A(sort_ind,sort_ind);

    Atril=tril(A,-1);
    Atriu=triu(A,1);
   
    RCk_sort=mex_thresholds_vertex_cut(Atril,Atriu); % this does not use the actual values of W

    
end
