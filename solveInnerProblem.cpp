// Solves the inner problem in the nonlinear inverse power method 
// for nonconstant eigenvectors of the graph 1-Laplacian.
//
// (C)2010 Matthias Hein and Thomas Buehler 
// Machine Learning Group, Saarland University, Germany
// http://www.ml.uni-saarland.de

#include <math.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
  // Test number of parameters.
  if (nrhs != 6 || nlhs != 4) {
    mexWarnMsgTxt("Usage: [X,rval,Obj,iter]=solveInnerProblem(W,Y,rval,MAXITER,EPS,FourTimesMaxSumSquaredWeights)");
    return;
  }
  
  // get important parameters
  int rows = (int)mxGetM(prhs[0]); // number of rows of W
  int cols = (int)mxGetN(prhs[0]); // number of columns of W (should be the same)
  int len  = (int)mxGetM(prhs[1]); // the desired output
  int lenrval = (int)mxGetM(prhs[2]); // rval

  if(!mxIsSparse(prhs[0])) { mexWarnMsgTxt("Matrix is not sparse");}
  
  if(rows!=cols){
    mexWarnMsgTxt("Sparse matrix is not square");
    return;
  }

  if(rows!=len){
    mexWarnMsgTxt("Length of the vector is not the same as the number of the rows of the sparse matrix");
    return;
  }
    
  // Create output array and compute values
  double* sr = mxGetPr(prhs[0]);     // get values
  mwIndex* irs = mxGetIr(prhs[0]);   // get row
  mwIndex* jcs = mxGetJc(prhs[0]);   // get columns
  
  double* Xobs = mxGetPr(prhs[1]);
  double* rval = mxGetPr(prhs[2]);     // get values

  int MAXITER = (int) mxGetScalar(prhs[3]); 
  double EPS = mxGetScalar(prhs[4]); 
  double MaxSumSquaredWeights = mxGetScalar(prhs[5]); 
  //mexPrintf("Elements: %f\n",MaxSumSquaredWeights);

  if(MaxSumSquaredWeights<=0){
	  mexWarnMsgTxt("Lipschitz constant has to be positive");
    return;
  }

  double *X;      /* output matrix */
  plhs[0] = mxCreateDoubleMatrix(len,1,mxREAL); /* create the output vector */
  plhs[1] = mxCreateDoubleMatrix(lenrval,1,mxREAL); /* create the output dual variable */
  plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL); /* create the output objective value */
  plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL); /* create the final iteration value */
  
  //plhs[1]= (mxArray *)prhs[2];
  //plhs[2]= (mxArray *)prhs[3];

  X = mxGetPr(plhs[0]);
  double* Z = mxGetPr(plhs[1]);
  double* OutputObj = mxGetPr(plhs[2]);
  double* FinalIter = mxGetPr(plhs[3]);

  int counter=0,i,j,start,mid,end,iter=0;
  double tnew=1; double told=1,alpha,beta,factor;
  double dummy,normD,normDiff,relativeChange,Fval;
  double* dummyPointer;

  double* D =new double[len];
  double* Dold =new double[len];
  for(i=0; i<len; i++) { D[i]=0; Dold[i]=0;}

  double* pval    = new double[lenrval];
  double* pvalold = new double[lenrval];
  for(i=0; i<lenrval; i++) { pval[i]=0; }
  for(i=0; i<lenrval; i++) { pvalold[i]=0; }

  
  //MaxSumSquaredWeights=max(sum(W.^2,2));
  /*double MaxSumSquaredWeights=0;  
  for(j=0; j<cols; j++) 
  {   
	dummy=0;
	for(i=0; i<jcs[j+1]-jcs[j]; i++) {  dummy+=sr[j]*sr[j]; }
	if(dummy>MaxSumSquaredWeights) { MaxSumSquaredWeights=dummy; }
  }
  MaxSumSquaredWeights=4*MaxSumSquaredWeights;*/
  Fval=EPS+1;
  while(iter<MAXITER && Fval > EPS)
  {
     //if (iter % 100 == 0){ mexPrintf("iter=%d tnew = %.5g\n",iter,tnew);}
    // exchange D and Dold
	dummyPointer=D; D=Dold; Dold=dummyPointer;

	//mexPrintf("Exchanged D \n");

	// exchange pval and pvalold
	dummyPointer=pval; pval=pvalold; pvalold=dummyPointer;

	//mexPrintf("Exchanged Pointer \n");

	// exchange tnew and told
	told=tnew;
    
	// initialize X with zeros 
    for(i=0; i<len; i++) { X[i]=0; }

	//mexPrintf("Initialized X, %i \n",X[0]);
  
    //sval = lambda*wval.*rval;
    //X = -sparse(jx,1,sval);
    dummy=0; counter=0;
    for(j=0; j<cols; j++) 
    {   
	   for(i=0; i<jcs[j+1]-jcs[j]; i++)
	   {  
         dummy = sr[counter]*rval[counter];
		 //mexPrintf("Computed dummy, %f\n",dummy);
         X[j] -= dummy;
	     X[irs[counter]] += dummy;
         counter++;
	   }
	} 
	//mexPrintf("Computed X, %f %f %f %f %f \n",X[0],X[1],X[2],X[3],X[4]);
    //D = Xobs-X; 
    normD = 0;  normDiff=0;
    for(i=0; i<len; i++) { D[i]=Xobs[i]-X[i]; normD+=D[i]*D[i]; normDiff+=(D[i]-Dold[i])*(D[i]-Dold[i]);}
  
    //pval = rval + uval.*(D(ix)-D(jx));
	 // pval=pval./max(abs(pval),1);
    counter=0;
	tnew = (1 + sqrt(1+4*told*told))/2;
    factor = (told-1)/tnew;
    for(j=0; j<cols; j++) 
    {   
       alpha=D[j];
	   for(i=0; i<jcs[j+1]-jcs[j]; i++)
	   {  
	      // update of pval
		  beta=rval[counter] + sr[counter]*( D[irs[counter]] - alpha)/MaxSumSquaredWeights;
		  // projection onto l_inf-cube 
		  if(beta>1) beta=1;
	      else if(beta<-1) beta=-1;
		  pval[counter]=beta;
		  // update of rval
          rval[counter] = beta + factor*(beta-pvalold[counter]);
		  counter++;
	   }	  
    }

    //tkp1=(1+sqrt(1+4*tk^2))/2;
    //rval = pval + (tk-1)/(tkp1)*(pval-pvalold);
	/*tnew = (1 + sqrt(1+4*told*told))/2;
    for(j=0; j<jcs[len]; j++) 
    {
	  alpha=pval[j];
	  if(alpha>1) { pval[j]=1; alpha=1;  }
	  else if(alpha<-1) { pval[j]=-1; alpha=-1;}
      rval[j] = alpha + (told-1)/tnew*(alpha-pvalold[j]);
    }*/
	//mexPrintf("Comp pval: %f %f %f\n",pval[0],pval[1],pval[len-1]);
	//mexPrintf("Comp rval: %f %f %f\n",rval[0],rval[1],rval[len-1]);
  
    relativeChange = sqrt(normDiff/normD);

    Fval = sqrt(normD);

    //mexPrintf("Iteration: %i, Fval: %1.15f, RelativeChange %1.15f\n",iter,Fval,relativeChange);
	//if(iter<10 || iter % 10==0)
	// mexPrintf("Iteration: %i, Fval: %1.15f, RelativeChange %1.15f\n",iter,Fval,relativeChange);
	iter++;
 }
 //mexPrintf("FINAL: Iterations %i, Fval: %1.15f, RelativeChange %1.15f\n",iter,Fval,relativeChange);
 for(i=0; i<len; i++) { X[i]=D[i];}
 for(i=0; i<lenrval; i++) { Z[i]=rval[i];}
 OutputObj[0]=Fval;
 FinalIter[0]=iter;
   //mexPrintf("iter=%d tnew = %.5g\n",iter,tnew);
 delete D; delete Dold; delete pvalold; delete pval;
	  
	

    // loop over columns
    /*for(i=0; i<jcs[j+1]-jcs[j]; i++) {     // loop over rows
      if(irs[counter]==j)
	  {
        sr[counter]+=vecvals[j];
	  }
	  counter++;
    }*/
    /*start=counter;  
	end=counter+jcs[j+1]-jcs[j]; 
	mid = floor(0.5*(start+end));
    while(irs[mid]!=j)
	{
		if(irs[mid]>j) { end=mid; mid = floor(0.5*(start+end)); }
		else           {start=mid; mid = floor(0.5*(start+end)); }
		//mexPrintf("Start: %i, Mid: %i, End %i \n",start,mid,end);
	}
	sr[mid]+=vecvals[j];
	counter=counter+jcs[j+1]-jcs[j];
  }*/
}
