// Solves the inner problem in the nonlinear inverse power method 
// for nonconstant eigenvectors of the graph 1-Laplacian.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Copyright 2010-15 Thomas Bühler and Matthias Hein
// Machine Learning Group, Saarland University, Germany
// http://www.ml.uni-saarland.de
#include <math.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
  
  // Test number of parameters.
  if (nrhs != 6 || nlhs != 4) {
    mexWarnMsgTxt("Usage: [X,alphaout,OutputObj,FinalIter]=solveInnerProblem(W,Y,alpha,MAXITER,EPS,FourTimesMaxSumSquaredWeights)");
    return;
  }
  
  // get important parameters
  int rows = (int)mxGetM(prhs[0]);      // number of rows of W
  int cols = (int)mxGetN(prhs[0]);      // number of columns of W (should be the same)
  int len  = (int)mxGetM(prhs[1]);      // length of Y
  int lenalpha = (int)mxGetM(prhs[2]);   // length of alpha

  // check if input has correct format
  if(!mxIsSparse(prhs[0])) { 
    mexWarnMsgTxt("Matrix is not sparse");
    return;
  }
  
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
  mwIndex* irs = mxGetIr(prhs[0]);   // get rows
  mwIndex* jcs = mxGetJc(prhs[0]);   // get columns
  
  double* Y = mxGetPr(prhs[1]);
  double* alpha = mxGetPr(prhs[2]);     // get values

  int MAXITER = (int) mxGetScalar(prhs[3]); 
  double EPS = mxGetScalar(prhs[4]); 
  
  // get Lipschitz constant
  double MaxSumSquaredWeights = mxGetScalar(prhs[5]); 
  
  if(MaxSumSquaredWeights<=0){
    mexWarnMsgTxt("Lipschitz constant has to be positive");
    return;
  }

  // allocate memory for output
  plhs[0] = mxCreateDoubleMatrix(len,1,mxREAL);     // create the output vector 
  plhs[1] = mxCreateDoubleMatrix(lenalpha,1,mxREAL); // create the output dual variable 
  plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);       // create the output objective value
  plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);       // create the final iteration value 
  
  // create pointers to output
  double* X = mxGetPr(plhs[0]);
  double* alphaout = mxGetPr(plhs[1]);
  double* OutputObj = mxGetPr(plhs[2]);
  double* FinalIter = mxGetPr(plhs[3]);

  // some helpers
  int counter=0,i,j,iter=0;
  double tnew=1,told=1,Dcur,betacur,factor;
  double dummy,normD,normDiff,relativeChange,Fval;
  double* dummyPointer;

  double* D =new double[len];
  double* Dold =new double[len];
  for(i=0; i<len; i++) { 
    D[i]=0; 
    Dold[i]=0;
  }

  double* beta    = new double[lenalpha];
  double* betaold = new double[lenalpha];
  for(i=0; i<lenalpha; i++) { 
    beta[i]=0;
    betaold[i]=0;
  }

  // main loop
  Fval=EPS+1;
  while(iter<MAXITER && Fval > EPS)
  {
    // exchange D and Dold
    dummyPointer=D; D=Dold; Dold=dummyPointer;

    // exchange beta and betaold
    dummyPointer=beta; beta=betaold; betaold=dummyPointer;

    // exchange tnew and told
    told=tnew;
    
    // initialize X with zeros 
    for(i=0; i<len; i++) { X[i]=0; }

    //sval = lambda*wval.*alpha;
    //X = -sparse(jx,1,sval);
    dummy=0; counter=0;
    for(j=0; j<cols; j++) 
    {   
        for(i=0; i<jcs[j+1]-jcs[j]; i++)
        {  
            dummy = sr[counter]*alpha[counter];
            X[j] -= dummy;
            X[irs[counter]] += dummy;
            counter++;
        }
    } 
    //D = Y-X; 
    normD = 0;  normDiff=0;
    for(i=0; i<len; i++) { 
        D[i]=Y[i]-X[i]; 
        normD+=D[i]*D[i]; 
        normDiff+=(D[i]-Dold[i])*(D[i]-Dold[i]);
    }
  
    //beta = alpha + uval.*(D(ix)-D(jx));
    // beta=beta./max(abs(beta),1);
    counter=0;
    tnew = (1 + sqrt(1+4*told*told))/2;
    factor = (told-1)/tnew;
    for(j=0; j<cols; j++) 
    {   
       Dcur=D[j];
       for(i=0; i<jcs[j+1]-jcs[j]; i++)
       {  
            // update of beta
            betacur=alpha[counter] + sr[counter]*( D[irs[counter]] - Dcur)/MaxSumSquaredWeights;
            // projection onto l_inf-cube 
            if(betacur>1) betacur=1;
            else if(betacur<-1) betacur=-1;
            beta[counter]=betacur;
            // update of alpha
            alpha[counter] = betacur + factor*(betacur-betaold[counter]);
            counter++;
       }	  
    }

    relativeChange = sqrt(normDiff/normD);
    Fval = sqrt(normD);
    iter++;
 }
 
 // set output
 for(i=0; i<len; i++) { X[i]=D[i];}
 for(i=0; i<lenalpha; i++) { alphaout[i]=alpha[i];}
 OutputObj[0]=Fval;
 FinalIter[0]=iter;

 delete[] D; delete[] Dold; delete[] betaold; delete[] beta;
  
}
