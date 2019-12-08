function [f_new2,obj,cur_delta,it,toc1] = ip_cheeger_bundle(start,params,eps1,strategy,verbose,obj_subg)
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
        
        t1=tic;
        [f_new2,obj,cur_delta,it]=ip_cutting_plane(start,params,eps1,'mosek',obj_subg);
        toc1=toc(t1);
        
%     case 'bundle_simple'
%
%         t1=tic;
%         [f_new2,obj,cur_delta,it]=ip_bundle_bell(start,params,eps1,false);
%         toc1=toc(t1);
        
    case 'bundle_level'
        
        t1= tic;
        [f_new2,obj,cur_delta,it]=ip_bundle_level(start,params,eps1,'linprog','fista_mex',verbose,obj_subg);
        toc1=toc(t1);
        
    otherwise
        error('Unknown strategy');
end


%    %% ABL METHOD L2 case
%     tic;
%     f_new=abs(start(ind_rest)/norm(start(ind_rest)));
%     disp(max(f_new)*(c)+  0.5*sum(wval_rest.*abs(f_new(ix_rest)-f_new(jx_rest))) - gamma * f_new'* 2* vec);
%     [f_new3,obj,cur_delta3,it3]=ip_ABL_L2(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest);
%
%
%
%     %[f_new2,obj,cur_delta,it]=ip_ABL_Linf(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest);
%
%     inner_obj_ABL_all(l)=obj;
%     times_ABL_all(l)=toc;
%     iter_ABL_all(l)=it;
%     disp(['Objective ABL:',num2str(inner_obj_ABL_all(l)),' time: ', num2str(times_ABL_all(l))]);
%
%
%
%     %% ABL METHOD Linf case
%     tic;
%     f_new=abs(start(ind_rest)/norm(start(ind_rest),inf));
%     disp(max(f_new)*(c)+  0.5*sum(wval_rest.*abs(f_new(ix_rest)-f_new(jx_rest))) - gamma * f_new'* 2* vec);
%     [f_new3,obj,cur_delta3,it3]=ip_ABL_Linf(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest,'barrier','barrier');
%
%
%     %[f_new2,obj,cur_delta,it]=ip_ABL_Linf(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest);
%
%     inner_obj_ABL_Linf_all(l)=obj;
%     times_ABL_Linf_all(l)=toc;
%     iter_ABL_Linf_all(l)=it3;
%     disp(['Objective ABL (Linf case):',num2str(inner_obj_ABL_Linf_all(l)),' time: ', num2str(times_ABL_Linf_all(l))]);
%
%
%      %% ABL METHOD Linf case
%     tic;
%     f_new=abs(start(ind_rest)/norm(start(ind_rest),inf));
%     disp(max(f_new)*(c)+  0.5*sum(wval_rest.*abs(f_new(ix_rest)-f_new(jx_rest))) - gamma * f_new'* 2* vec);
%     [f_new3,obj,cur_delta3,it3]=ip_ABL_Linf(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest,'cvx','cvx');
%
%
%     %[f_new2,obj,cur_delta,it]=ip_ABL_Linf(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest);
%
%     inner_obj_ABL_Linf_all_cvx(l)=obj;
%     times_ABL_Linf_all_cvx(l)=toc;
%     iter_ABL_Linf_all_cvx(l)=it3;
%     disp(['Objective ABL (Linf case, cvx):',num2str(inner_obj_ABL_Linf_all_cvx(l)),' time: ', num2str(times_ABL_Linf_all_cvx(l))]);
%
%
%
%     %% APL METHOD Linf case
%     tic;
%     f_new=abs(start(ind_rest)/norm(start(ind_rest)));
%     disp(max(f_new)*(c)+  0.5*sum(wval_rest.*abs(f_new(ix_rest)-f_new(jx_rest))) - gamma * f_new'* 2* vec);
%     [f_new3,obj,cur_delta3,it]=ip_APL_Linf(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest);
%
%
%      inner_obj_APL_Linf_all(l)=obj;
%      times_APL_Linf_all(l)=toc;
%      iter_APL_Linf_all(l)=it;
%      disp(['Objective APL (Linf case):',num2str(inner_obj_APL_Linf_all(l)),' time: ', num2str(times_APL_Linf_all(l))]);
%
%
%
%     %% APL METHOD L2 case
%     tic;
%     f_new=abs(start(ind_rest)/norm(start(ind_rest)));
%     disp(max(f_new)*(c)+  0.5*sum(wval_rest.*abs(f_new(ix_rest)-f_new(jx_rest))) - gamma * f_new'* 2* vec);
%     [f_new3,obj,cur_delta3,it]=ip_APL_L2(c,-gamma*2*vec,0.5,1E-4,start(ind_rest),wval_rest,ix_rest,jx_rest);
%
%
%      inner_obj_APL_L2_all(l)=obj;
%      times_APL_L2_all(l)=toc;
%      iter_APL_L2_all(l)=it;
%      disp(['Objective APL (L2 case):',num2str(inner_obj_APL_L2_all(l)),' time: ', num2str(times_APL_L2_all(l))]);




end
