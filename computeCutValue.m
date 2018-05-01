function [cutpart1,cutpart2] = computeCutValue(clusters,W,normalized,deg)
% Computes the components in the Ratio/Normalized Cut and Ratio/Normalized Cheeger Cut expression. 
%
% Usage: [cutpart1,cutpart2] = computeCutValue(clusters,W,normalized)
%
% One then has Ratio/Normalized Cut = cutpart1 + cutpart2
% and Ratio/Normalized Cheeger Cut = max(cutpart1,cutpart2)
%
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% (C)2010-18 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    W3= sum(W(clusters==1,:),1);
    cut=full(sum(W3(clusters~=1),2));
    
    if(cut==0)
        cutpart1=0;
        cutpart2=0;
    else
        if (~normalized)
            sizeA = sum(clusters==1);
            sizeB = size(clusters,1)-sizeA;

            cutpart1=cut/sizeA;
            cutpart2=cut/sizeB;
        else
            degA=deg(clusters==1);
            volA=sum(degA);
            
            degB=deg(clusters~=1);
            volB=sum(degB);
  
            cutpart1=cut/volA;
            cutpart2=cut/volB;
        end
    end
    
end