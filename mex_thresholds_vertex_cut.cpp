// Computes the vaues of the vertex cut for each threshold
//
// (C)2010 Thomas Buehler 
// Machine Learning Group, Saarland University, Germany
// http://www.ml.uni-saarland.de

#include <math.h>
#include "mex.h"
#include "matrix.h"
#include <time.h>
#include "float.h"


void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    // Test number of parameters.
    if (nlhs != 1 || nrhs != 2) {
        mexWarnMsgTxt("Usage: RCk_sort=mex_thresholds_vertex_cut(Atril,Atriu)");
        return;
    }
    
    
    // get matrix lower
    mwIndex * ir = mxGetIr(prhs[0]);
    mwIndex * jc = mxGetJc(prhs[0]);
    mwSize rowsl=mxGetM(prhs[0]);
    mwSize colsl=mxGetN(prhs[0]);
    mwSize nnzl=mxGetNzmax(prhs[0]);
    
    
    // get matrix upper
    mwIndex * iru = mxGetIr(prhs[1]);
    mwIndex * jcu = mxGetJc(prhs[1]);
    mwSize rowsu=mxGetM(prhs[1]);
    mwSize colsu=mxGetN(prhs[1]);
    mwSize nnzu=mxGetNzmax(prhs[1]);
    
    // some output
    bool debug_mode=false;
    if (debug_mode) {
        mexPrintf("dim of Atril: %d  %d number of nonzeros %d\n",rowsl, colsl, nnzl);
        mexPrintf("dim of Atriu: %d  %d number of nonzeros %d\n",rowsu, colsu, nnzu);
    }
    
    // make some checks
    if (rowsl!= colsl || rowsu!=colsu || rowsl!=rowsu) {
        mexWarnMsgTxt("Dimensions mismatch.");
        return;
    }
    
    
    mwSize num=rowsl;
    double sum;
    
    double * part1 = new double[num];
    double * part2 = new double[num];
    double * M = new double[num];
    
    // initialize M
    for (mwIndex i=0;i<num;i++) {
        M[i]=0;
    }
    part1[0]=0;
    
    // go through cols j=1 to (end -1)
    // correpsonding to k=2 to end
    for (mwIndex j =0;j<num-1;j++) {
        // go through corresponding row entries
        for (mwIndex i=jc[j];i<jc[j+1];i++) {
            M[ir[i]]=1; // the actual value of Atril is unimportant
        }
        
        // sum up for part1
        sum=0;
        for (mwIndex i=j+1;i<num;i++) {
            sum+=M[i];
        }
        part1[j+1]=sum;
    }
    
    
    // do the same for part2
    // initialize M
    for (mwIndex i=0;i<num;i++) {
        M[i]=0;
    }
    
    
    // go through cols j=num to 1
    // correpsonding to k=2 to end
    mwIndex curcol;
    for (mwIndex j =0;j<num;j++) {
        
        curcol=num-1-j;
        // go through corresponding row entries
        for (mwIndex i=jcu[curcol];i< jcu[curcol+1] ;i++) {  //jcu has num+1 entries
            M[iru[i]]=1;
        }
        
        // sum up for part
        sum=0;
        for (mwIndex i=0;i<curcol;i++) {
            sum+=M[i];
        }
        part2[curcol]=sum;
    }
    
    
    
    
    // create output
    plhs[0] = mxCreateDoubleMatrix(num,1,mxREAL);          // RCk
    
    // pointers to output
    double* RCk_sort = mxGetPr(plhs[0]);
    
    // copy output
    for (mwIndex i=0;i<num;i++) RCk_sort[i]=part1[i]+part2[i];
    
    // free memory
    delete [] M; delete [] part1; delete [] part2;
    return;
    
    
}


