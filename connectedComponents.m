function [comp,connected,sizes]=connectedComponents(W)
% Returns all connected components of a graph.
%
% Usage: [comp,connected,sizes]=connectedComponents(W)
%
% Input: 
%   W:      - Sparse symmetric weight matrix, size nxn
%
% Output:
%   comp:      - indicator vector of connected components of size nx1
%   connected: - 1 if graph has only one connected component
%   sizes:     - size of each connected component
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% (C)2010-18 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    l=1;
    [isCon, first_comp] = isConnected(W);
    sizes(l)=sum(first_comp);
    connected=isCon;

    comp=first_comp;
    while ~isCon
        ind=find(comp==0);
        Wpart=W(ind,ind);
        [isCon, first_comp] = isConnected(Wpart);
        l=l+1;
        sizes(l)=sum(first_comp);
        comp(ind)=l*first_comp;
    end
end

function [connected,components]=isConnected(W)
% Checks whether a graph is connected.
    A = W>0; % adjacency matrix
    alreadyseen = zeros(size(W,1),1);
    currentCandidates=1;

    while ~isempty(currentCandidates)
        candidates= (sum(A(:,currentCandidates),2)>0);
        alreadyseen(currentCandidates)=1;
        currentCandidates=find(candidates-alreadyseen>0);
    end

    connected = sum(alreadyseen)==size(W,2);
    components=alreadyseen;
end

