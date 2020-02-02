function [obj,subg] = obj_subg_vertex_exp(x,params)
% Performs objective and subgradient for the symmetric vertex expansion.
%
% (C)2012-14 Thomas Buehler
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de

    c2 = params.c2;

    num = length(x);
    [RCk_sort,sort_ind] = thresholds_vertex_cut_fast(x, params.W);
   
    RCk_sort_shift = [RCk_sort(2:end); 0];

    subg = zeros(num,1);
    subg(sort_ind) = RCk_sort-RCk_sort_shift;

    subg = subg+c2;
    
    obj = subg'*x;
end
