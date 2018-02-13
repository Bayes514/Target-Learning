
#include "tan_ensemble.h"
#include "utils.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>
using namespace std;

tan_ensemble::tan_ensemble() : 
trainingIsFinished_(false)
{
}

tan_ensemble::tan_ensemble(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false)
{
    name_ = "tan_ensemble";
}

tan_ensemble::~tan_ensemble(void)
{
}

void tan_ensemble::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    //safeAlloc(parents, noCatAtts_);
    parents_.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        parents_[a] = NOPARENT;
    }

    xxyDist_.reset(is);
}

void tan_ensemble::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void tan_ensemble::initialisePass()
{
    assert(trainingIsFinished_ == false);
    
}

void tan_ensemble::train(const instance &inst)
{
    xxyDist_.update(inst);
}



void tan_ensemble::finalisePass()
{
    assert(trainingIsFinished_ == false);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(xxyDist_, cmi);


    CategoricalAttribute firstAtt = 0;

    parents_[firstAtt] = NOPARENT;

    float *maxWeight;
    CategoricalAttribute *bestParentSoFar;
    CategoricalAttribute candidate = firstAtt;
    std::set<CategoricalAttribute> notInTree;

    safeAlloc(maxWeight, noCatAtts_);
    safeAlloc(bestParentSoFar, noCatAtts_);//bestParentSoFar=new int[noCatAtts_]

    maxWeight[firstAtt] = -std::numeric_limits<float>::max();
    
    
    for (CategoricalAttribute a = 1; a < noCatAtts_; a++)
    {
        maxWeight[a] = cmi[firstAtt][a];
        if (cmi[firstAtt][a] > maxWeight[candidate])
            candidate = a;
        bestParentSoFar[a] = firstAtt;
        notInTree.insert(a);
    }
    
   
    while (!notInTree.empty())
    {
       
        const CategoricalAttribute current = candidate;
        parents_[current] = bestParentSoFar[current];
        notInTree.erase(current);

        if (!notInTree.empty())
        {
            candidate = *notInTree.begin();
            for (std::set<CategoricalAttribute>::const_iterator it =
                    notInTree.begin(); it != notInTree.end(); it++)
            {
                
                if (maxWeight[*it] < cmi[current][*it])
                {
                    maxWeight[*it] = cmi[current][*it];
                    bestParentSoFar[*it] = current;
                }
                
               
                if (maxWeight[*it] > maxWeight[candidate])
                    candidate = *it;
            }
        }
    }

  
    delete[] bestParentSoFar;
    delete[] maxWeight;

    
    trainingIsFinished_ = true;
}


bool tan_ensemble::trainingIsFinished()
{
    return trainingIsFinished_;
}

void tan_ensemble::classify(const instance &inst, std::vector<double> &classDist)
{
   
    crosstab<float> cmiLocal = crosstab<float>(noCatAtts_);
    getCondMutualInfloc(xxyDist_, cmiLocal, inst);

    vector<CategoricalAttribute> parents_loc;
    parents_loc.assign(noCatAtts_,NOPARENT);
    
    CategoricalAttribute firstAtt = 0;

    parents_loc[firstAtt] = NOPARENT;

    float *maxWeight;
    CategoricalAttribute *bestParentSoFar;
    CategoricalAttribute candidate = firstAtt;
    std::set<CategoricalAttribute> notInTree;

    safeAlloc(maxWeight, noCatAtts_);
    safeAlloc(bestParentSoFar, noCatAtts_);//bestParentSoFar=new int[noCatAtts_]

    maxWeight[firstAtt] = -std::numeric_limits<float>::max();
    
   
    for (CategoricalAttribute a = 1; a < noCatAtts_; a++)
    {
        maxWeight[a] = cmiLocal[firstAtt][a];
        if (cmiLocal[firstAtt][a] > maxWeight[candidate])
            candidate = a;
        bestParentSoFar[a] = firstAtt;
        notInTree.insert(a);
    }
    
    
    while (!notInTree.empty())
    {
       
        const CategoricalAttribute current = candidate;
        parents_loc[current] = bestParentSoFar[current];
        notInTree.erase(current);

        if (!notInTree.empty())
        {
            candidate = *notInTree.begin();
            for (std::set<CategoricalAttribute>::const_iterator it =
                    notInTree.begin(); it != notInTree.end(); it++)
            {
               
                if (maxWeight[*it] < cmiLocal[current][*it])
                {
                    maxWeight[*it] = cmiLocal[current][*it];
                    bestParentSoFar[*it] = current;
                }
                
               
                if (maxWeight[*it] > maxWeight[candidate])
                    candidate = *it;
            }
        }
    }

   
    delete[] bestParentSoFar;
    delete[] maxWeight;

        
/*        
 * p(y)= [∑i δ(y,ū_i)+(1/m)]/(N+1)  {i=1,2,...,N}
 *y is the ture class lable and ū_i is Corresponding estimate. m is the count of y may values.
 *N is the total sample count.
 */

/*              
 * p(x_j)= [∑i δ(x_j,ā_ij)+1]/(N+1)  {i=1,2,...,N}
 *x_j is a sample  and ā_ij is Corresponding estimate.
 *N is the total sample count.
 */
 /*  
 * p(x_j,y)= [∑i δ[<x_j,y>,<ā_ij,ū_i>]+(1+m)]/(N+1)  {i=1,2,...,N}
 *y is the ture class lable and ū_i is estimate.
 *x_j is a sample  and ā_ij is Corresponding estimate.
 *m is the count of y may values.
 *N is the  total sample count.
 */
    
    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] = xxyDist_.xyCounts.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent_loc = parents_loc[x1];

        if (parent_loc == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
            }
        } else
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent_loc,
                        inst.getCatVal(parent_loc), y);
            }
        }
    }
    normalise(classDist);
    
   
    std::vector<double> classDistGeneral(noClasses_,0.0);
     for (CatValue y = 0; y < noClasses_; y++)
    {
        classDistGeneral[y] = xxyDist_.xyCounts.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent = parents_[x1];

        if (parent == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDistGeneral[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
            }
        } else
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDistGeneral[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
            }
        }
    }
    normalise(classDistGeneral);
    
     for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] += classDistGeneral[y];
        classDist[y]=classDist[y]/2;
    }
    normalise(classDist);
}

