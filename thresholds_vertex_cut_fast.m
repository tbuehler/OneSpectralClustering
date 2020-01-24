function [RCk_sort,sort_ind] = thresholds_vertex_cut_fast(x, W)
% Performs optimal thresholding of the numerator of the unconstrained set ratio 
% for vertex expansion
%
% (C)2012-14 Thomas Buehler
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de

    [~,sort_ind] = sort(x);
    W = W(sort_ind,sort_ind);

    Wtril = tril(W,-1);
    Wtriu = triu(W,1);
   
    RCk_sort = mex_thresholds_vertex_cut(Wtril,Wtriu); % this does not use the actual values of W
end
