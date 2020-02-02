function [f_new2, obj, it, toc1] = ip_bundle(start,params,eps1,strategy,verbose,obj_subg)
% Solves the inner problem in the IPM using cutting-plane or bundle level method
%
% (C)2010 Thomas Buehler 
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    % check that input is correct
    assert(isfield(params,'c2'));

    %solve problem
    switch strategy
        case 'cutting_plane'
            t1 = tic;
            [f_new2, obj, it] = ip_cutting_plane(start,params,eps1,'linprog',obj_subg);
            toc1 = toc(t1);
    %     case 'bundle_simple'
    %         t1=tic;
    %         [f_new2,obj,cur_delta,it]=ip_bundle_bell(start,params,eps1,false);
    %         toc1=toc(t1);
        case 'bundle_level'
            t1 = tic;
            [f_new2, obj, it] = ip_bundle_level(start,params,eps1,'linprog','fista_mex',verbose,obj_subg);
            toc1 = toc(t1);
        otherwise
            error('Unknown strategy');
    end
end
