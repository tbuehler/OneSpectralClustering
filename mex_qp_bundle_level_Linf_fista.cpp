// Solves the quadratic problem in the bundle-level method using FISTA
//
// (C)2010 Thomas Buehler 
// Machine Learning Group, Saarland University, Germany
// http://www.ml.uni-saarland.de

#include <math.h>
#include "mex.h"
#include "matrix.h"
#include <time.h>
#include "float.h"


void compute_vproj(double *,double*,double*,double *,double *,int, int );
void compute_objective(double *,double ,double *,double *,double ,double,double *,double & ,double &,int,int,double * );
void compute_A_vproj(double * ,double * ,double * , int , int);


void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
  // Test number of parameters.
  if (nlhs != 7 || nrhs != 7) {
     mexWarnMsgTxt("Usage: [vproj,primal_obj,dual_obj,alpha, it,vproj_best,primal_obj_best]=mex_qp_bundle_level_Linf_fista(z,A,At,lev,eps1,L,alpha)");
    return;
  }
 
  // get input vector
  double * z = mxGetPr(prhs[0]);        // vector z
  int dim_z=mxGetM(prhs[0]);
  
  // get matrix of subgradients
  double * A = mxGetPr(prhs[1]);        // matrix of subgradients
  int size_bundle = mxGetM(prhs[1]);    // size of the bundle
  int dim = mxGetN(prhs[1]);            // dimension
  
  // get transposed matrix of subgradients (not sure if this is necessary, let's see)
  double * At = mxGetPr(prhs[2]);        // matrix of subgradients
  int size_bundle_t = mxGetN(prhs[2]);    // size of the bundle
  int dim_t = mxGetM(prhs[2]);            // dimension
  
  // get other params
  double lev= mxGetScalar(prhs[3]);
  double eps1=mxGetScalar(prhs[4]);
  double L= mxGetScalar(prhs[5]);
  double step=1/L;
  
  // get starting vector
  double * alpha_start = mxGetPr(prhs[6]);
  int dim_alph = mxGetM(prhs[6]);
  
  // some output
  bool debug_mode=false;
  if (debug_mode) {
    mexPrintf("dim of z: %d  size of bundle: %d dim of bundle %d\n",dim_z,size_bundle,dim);
    mexPrintf("size of bundle2: %d dim of bundle2 %d\n",size_bundle_t,dim_t);
    mexPrintf("level: %.4g  eps1: %.4g   L : %.4g\n",lev,eps1,L);
    mexPrintf("dimension of alpha: %d  \n",dim_alph);
  }
  
  // check that dimensions are correct
  if (dim!=dim_z || dim_t!=dim_z || size_bundle_t!=size_bundle
          || dim_alph!= size_bundle) {
      mexWarnMsgTxt("Dimensions mismatch.");
      return;
  }

  // stepsizes in fista, iteration number
  double tnew,told,factor;
  int it;
  
  // initialization;
  tnew=1;
  told=1;
  it=0;
  int itmax=1000;
  
  // i don't want to verwrite my starting values since this is a possible source of bugs
  double *alph = new double[dim_alph];
  for (int i=0;i<dim_alph;i++) alph[i]=alpha_start[i];
  
  // create additional FISTA variable
  double* betat    = new double[dim_alph];
  double* betat_new = new double[dim_alph];
  double * dummy;
  
  for (int i=0;i<dim_alph;i++) betat[i]=alpha_start[i];
  
  // this stays constant
  double norm_z=0;
  for (int i=0;i< dim;i++){
      norm_z+=z[i]*z[i];
  }
  norm_z=0.5*norm_z;
  
  // for feasibility
  double eps2=0.001;
  
  // compute current primal variable
  double * v= new double[dim];
  double * vproj = new double[dim];
  

  compute_vproj(z,A,alph,v,vproj,dim,dim_alph);
   
  // compute objectives
  double  primal_obj=0;
  double  dual_obj=0;
  
  
  double * Avproj=new double[dim_alph];
  compute_A_vproj(At,vproj,Avproj,dim,dim_alph);
 
  compute_objective(vproj,norm_z,v,alph,lev,eps2,z,primal_obj,dual_obj,dim,dim_alph,Avproj);
  
  double gap = primal_obj-dual_obj;
  
 
  // current best objective and primal variable
  double * vproj_best= new double[dim];
  for (int i=0;i<dim;i++){
    vproj_best[i]= vproj[i];
  }

  
  double primal_obj_best=primal_obj;
  
  
  if (debug_mode) {
    mexPrintf("iter= %d primal_obj_best=%.4g primal_obj =%.4g dual obj =%.4g gap = %.4g\n", it,primal_obj_best, primal_obj,dual_obj,gap);
  }
  /*
  int l1,l2;
  for (l1=0;l1<dim;l1++) {
     for (l2=0; l2<size_bundle;l2++) {
         mexPrintf("Column %d Row %d: Entry %.5g\n",l1+1,l2+1, A[l1*size_bundle+l2]);
     }
  }
  
   *
   *
   *
 */
  
  
  bool converged=false;
  while (!converged && it<itmax) {
      
      it=it+1;
      
      // update beta_new;
      for (int i=0;i<dim_alph;i++) {
          betat_new[i]=alph[i]+ step* (Avproj[i] -lev);
          if (betat_new[i]<0) betat_new[i]=0;
      }
   
      
      // update stepsizes
      tnew = (1 + sqrt(1+4*told*told))/2;
      factor = (told-1)/tnew; 
      
      // update alpha
      for (int i=0;i<dim_alph;i++) {
          alph[i]=betat_new[i] + factor*(betat_new[i]-betat[i]);
          
      }
     
      // update variables
      told=tnew;
      dummy = betat;
      betat= betat_new;
      betat_new=dummy;
      
      
      
      // compute objective etc
      compute_vproj(z,A,alph,v,vproj,dim,dim_alph);
      
      
   //   for (int i=0;i<10;i++) {
   //       mexPrintf("%.3g\n",vproj[i]);
   //   }
      compute_A_vproj(At,vproj,Avproj,dim,dim_alph);
  //for (int i=0;i<dim_alph;i++) {
   //       mexPrintf("%.3g\n",Avproj[i]);
    //  }
      
      
      compute_objective(vproj,norm_z,v,alph,lev,eps2,z,primal_obj,dual_obj,dim,dim_alph,Avproj);
  
     // mexPrintf("%.5g %.5g\n",primal_obj,dual_obj);
      
      gap = primal_obj-dual_obj;
      
      
      // store best objective
      if (primal_obj<primal_obj_best) {
           for (int i=0;i<dim;i++){
            vproj_best[i]= vproj[i];
            }
           primal_obj_best=primal_obj;
      }
      
      
      
      // check if converged
      if (fabs(gap)<eps1) converged=true;
      
           
      // display stuff
      if (debug_mode) {
      mexPrintf("iter= %d primal_obj_best=%.4g primal_obj =%.4g dual obj =%.4g fabs(gap) =%.4g gap = %.4g eps1=%.4g converged=%d\n", it,primal_obj_best, primal_obj,dual_obj,fabs(gap),gap,eps1,converged);
      }
     //   return;
      
  }
  
  // final iteration
  //mexPrintf("iter= %d primal_obj_best=%.4g primal_obj =%.4g dual obj =%.4g fabs(gap) =%.4g gap = %.4g eps1=%.4g converged=%d\n", it,primal_obj_best, primal_obj,dual_obj,fabs(gap),gap,eps1,converged);
    
  
  
  
  
  
  
  
  
  
  
  
  
  

  // create output        
  plhs[0] = mxCreateDoubleMatrix(dim,1,mxREAL);         // vproj
  plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);           // primal obj
  plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);           // dual obj
  plhs[3] = mxCreateDoubleMatrix(dim_alph,1,mxREAL);   // alpha
  plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);           // it
  plhs[5] = mxCreateDoubleMatrix(dim,1,mxREAL);         // vproj best
  plhs[6] = mxCreateDoubleMatrix(1,1,mxREAL);           // primal obj best
  
  // pointers to output
  double* vproj_out = mxGetPr(plhs[0]);
  double* primal_obj_out = mxGetPr(plhs[1]);
  double* dual_obj_out = mxGetPr(plhs[2]);
  double* alpha_out = mxGetPr(plhs[3]);
  double* iter_out = mxGetPr(plhs[4]);
  double* vproj_best_out = mxGetPr(plhs[5]); 
  double* primal_obj_best_out = mxGetPr(plhs[6]);
    
  
  // copy output
  for (int i=0;i<dim;i++) vproj_out[i]=vproj[i];
  *primal_obj_out= primal_obj;
  *dual_obj_out = dual_obj;
  for (int i=0;i<dim_alph;i++) alpha_out[i]=alph[i];
  *iter_out=it;
  for (int i=0;i<dim;i++) vproj_best_out[i]=vproj_best[i];
  *primal_obj_best_out = primal_obj_best;
  
  // free memory
  delete[] v; delete[] vproj;delete[] vproj_best; delete[] betat; delete[] betat_new; delete[] Avproj; delete[] alph;
  return;
  
  
  
  
  /*
   
 
  
  
  	//start = clock();
	//ende = clock();
	//time2+=(float) (ende-start);
    
    //primalVal2=0;
  
  
  


  clock_t start,ende;
  float time1=0,time2=0, time3=0,time4=0;


	dummyPointer=pval; pval=pvalold; pvalold=dummyPointer;
 */
  
  
}
  
  
  

	
     

    
void compute_A_vproj(double * At,double * vproj,double * Avproj, int dim, int dim_alpha){
    
    
    double sum;
    // compute this as   vproj' *A' 
      for (int i=0;i< dim_alpha;i++) { // for every column = dimension of alpha
        sum=0;
        for (int j=0;j< dim;j++) {
            sum+=vproj[j]*At[i*dim + j];
        }
        Avproj[i]=sum;
    }
}
    
    
    
    
    
    


    


// computes the primal variable
void compute_vproj(double *z,double *A,double *alpha,double *v, double *vproj,int dim, int dim_alpha){

    double sum;
    
    // compute this as   z'-alpha' *A 
    for (int i=0;i< dim;i++) { // for every column = dimension of v
        sum=0;
        for (int j=0;j< dim_alpha;j++) {
            sum+=alpha[j]*A[i*dim_alpha + j];
        }
        v[i]=z[i]-sum;
    }
    
    // now do projection
    for (int i=0;i<dim;i++) {
        if (v[i]>=1)
            vproj[i]=1;
        else if (v[i]<=-1)
            vproj[i]=-1;
        else vproj[i]=v[i];
    }
    
    return;
 }
   
// computes primal and dual objectives
void  compute_objective(double *vproj,double norm_z,double *v,double *alpha,double lev,double eps2,double *z,double & primal_obj,double & dual_obj,int dim,int dim_alpha, double * Avproj){
    
    
    double norm_vproj=0;
    for (int i=0;i<dim;i++) {
        norm_vproj+= vproj[i]*vproj[i];
    }
    norm_vproj=0.5*norm_vproj;
    
    
    double sum_alpha=0;
    for (int i=0;i< dim_alpha; i++) {
        sum_alpha+= alpha[i];
    }
    
    
    double sum_proj=0;
    for (int i=0; i<dim; i++) {
        if (v[i]>1) sum_proj+=v[i]-1;
        else if (v[i]<-1) sum_proj-= (v[i]+1);
    }
    
    
    dual_obj=-norm_vproj + norm_z - lev* sum_alpha -sum_proj;
    
    
    /*
    double max_Avproj=Avproj[0];
    for (int i=1;i<dim_alpha;i++) {
        if (Avproj[i]> max_Avproj)
            max_Avproj=Avproj[i];
    }
    
    if (max_Avproj>lev + eps2 * fabs(lev))
        primal_obj= DBL_max;
    else {
        prod_vproj_z;
        for (int i=0;i<dim;i++) {
            prod_vproj_z += vproj[i]*z[i]; }
       primal_obj= norm_vproj + norm_z - prod_vproj_z;
    }*/
    
    
    bool feasible= true;
    double bound = lev+eps2*fabs(lev);
    //mexPrintf("%.5g\n",bound);
    int i=0;
    while (feasible && i<dim_alpha) {
        //mexPrintf("%.5g\n",Avproj[i]);
        if(Avproj[i]>bound)
            feasible=false;
        i++;
    }
    
    if (feasible) {
        double prod_vproj_z=0;
        for (int i=0;i<dim;i++) {
            prod_vproj_z += vproj[i]*z[i]; }
        primal_obj= norm_vproj + norm_z - prod_vproj_z;
    }
    else {
        primal_obj= DBL_MAX;
    }
           
 
}
  

