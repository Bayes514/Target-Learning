/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 **
 ** This program is free software: you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation, either version 3 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program. If not, see <http://www.gnu.org/licenses/>.
 **
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

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
    //learner::initialisePass (pass_);
    //	dist->clear();
    //	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
    //		parents_[a] = NOPARENT;
    //	}
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
//    for(int i=0;i<noCatAtts_;i++)
//    {
//        for(int j=0;j<noCatAtts_;j++)
//            cout<<cmi[i][j]<<" ";
//        cout<<endl;
//        
//    }    
    // find the maximum spanning tree

    CategoricalAttribute firstAtt = 0;

    parents_[firstAtt] = NOPARENT;

    float *maxWeight;
    CategoricalAttribute *bestParentSoFar;
    CategoricalAttribute candidate = firstAtt;
    std::set<CategoricalAttribute> notInTree;

    safeAlloc(maxWeight, noCatAtts_);//其实就是 maxWeight=new float[noCatAtts_]
    safeAlloc(bestParentSoFar, noCatAtts_);//bestParentSoFar=new int[noCatAtts_]

    maxWeight[firstAtt] = -std::numeric_limits<float>::max();
    
    
    for (CategoricalAttribute a = 1; a < noCatAtts_; a++)
    {
        maxWeight[a] = cmi[firstAtt][a];
        if (cmi[firstAtt][a] > maxWeight[candidate])
            candidate = a;
        bestParentSoFar[a] = firstAtt;//目前各个属性的父结点设置为0
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

    //for (attribute a = 0; a < meta->noAttributes; a++) {
    //  delete []mi[a];
    //}
    //delete []mi;
    delete[] bestParentSoFar;
    delete[] maxWeight;

    
    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool tan_ensemble::trainingIsFinished()
{
    return trainingIsFinished_;
}

void tan_ensemble::classify(const instance &inst, std::vector<double> &classDist)
{
   
    crosstab<float> cmiLocal = crosstab<float>(noCatAtts_);
    getCondMutualInfloc(xxyDist_, cmiLocal, inst);
//    for(int i=0;i<noCatAtts_;i++)
//    {
//        for(int j=0;j<noCatAtts_;j++)
//            cout<<cmi[i][j]<<" ";
//        cout<<endl;
//        
//    }    
    // find the maximum spanning tree
   
    vector<CategoricalAttribute> parents_loc;
    parents_loc.assign(noCatAtts_,NOPARENT);
    
    CategoricalAttribute firstAtt = 0;

    parents_loc[firstAtt] = NOPARENT;

    float *maxWeight;
    CategoricalAttribute *bestParentSoFar;
    CategoricalAttribute candidate = firstAtt;
    std::set<CategoricalAttribute> notInTree;

    safeAlloc(maxWeight, noCatAtts_);//其实就是 maxWeight=new float[noCatAtts_]
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

    //for (attribute a = 0; a < meta->noAttributes; a++) {
    //  delete []mi[a];
    //}
    //delete []mi;
    delete[] bestParentSoFar;
    delete[] maxWeight;

        
    
    
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

