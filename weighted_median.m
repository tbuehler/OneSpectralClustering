function [c,index] =  weighted_median(f,deg)
% Computes the weighted median corresponding to weights deg of the vector f
%
% Usage: [c,index] =  weighted_median(f,deg)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% (C)2010-11 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    [sf,ix]=sort(f);
    cumdeg = cumsum(deg(ix));
    totdeg = cumdeg(end);

    [f_unique, ix2] = unique(sf,'last'); % ix2 gives index of last occurence
    cumdeg_unique=cumdeg(ix2);
    cumdeg_unique_shifted=[0;cumdeg_unique(1:end-1)];
    deg_unique= cumdeg_unique - cumdeg_unique_shifted;

    alpha = cumdeg_unique_shifted + cumdeg_unique - totdeg;
    alpha=alpha./deg_unique;

    ix3= find(abs(alpha)==min(abs(alpha)));
    
    index = ix(ix2(ix3(1)));
    c = f(index);
    
end
