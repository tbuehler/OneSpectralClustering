function [clusters, vertex_exp] = opt_thresh_vertex_expansion(x, W, normalized)
% Performs optimal thresholding of the unconstrained set ratio for vertex
% expansion
%
% Input:
% x                 The vector.
% params            Contains weight matrix etc.
%
% Output:
% clusters          Thresholded vector f yielding the best objective.
% vertex_exp        Vertex expansion value of resulting clustering.
%
%
% (C)2012-14 Thomas Buehler
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    % perform optimal thresholding of numerator
    [RCk_sort,sort_ind] = thresholds_vertex_cut_fast(x, W);
    
    % denominator
    num = length(x);
    
    if (normalized)
        deg = full(sum(W));
        deg = deg(sort_ind);
        
        vol_compl = cumsum(deg);
        vol = sum(deg)-vol_compl;
        
        vol = vol(1:num-1);
        vol_compl = vol_compl(1:num-1);
        denom = min(vol,vol_compl)';
    else
        card = num-1:-1:1;
        card_compl = 1:1:num-1;
        denom = min(card,card_compl)';
    end
    
    % remove first entry (corresponds to empty set)
    RCk_sort = RCk_sort(2:end);
    sort_ind = sort_ind(2:end);
    
    % find best objective
    objective = RCk_sort./denom;
    ix = find(objective==min(objective),1);
    vertex_exp = objective(ix);
    clusters = zeros(num,1);
    clusters(sort_ind(ix:end)) = 1;
end

