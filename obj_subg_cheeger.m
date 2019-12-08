function [obj,subg] = obj_subg_cheeger(x,params)
% Performs objective and subgradient for the cheeger cut objective.
%
% (C)2012-14 Thomas Buehler
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de

    c2=params.c2;
    wval=params.wval;
    ix=params.ix;
    jx=params.jx;

    subg= sparse(ix, ones(length(ix),1), wval.*sign(x(ix)-x(jx)), length(x),1) + c2;


    % because of one-homogeneity
    obj=x'*subg;
end
