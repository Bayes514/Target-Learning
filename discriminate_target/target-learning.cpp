/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 ** Implements Sahami's k-dependence Bayesian classifier
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
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>
#include<iostream>

#include "target-learning.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

target-learning::target-learning() : pass_(1)
{
}

target-learning::target-learning(char*const*& argv, char*const* end) : pass_(1)
{
    name_ = "localkdb";
    union_kdb_localkdb = true;//更改truefalse选择是否结合，默认为结合
    // defaults
    k_ = 1;

    // get arguments
    while (argv != end)
    {
        if (*argv[0] != '-')
        {
            break;
        } else if (argv[0][1] == 'k')
        {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else if (streq(argv[0] + 1, "un"))//加上-un就是全局和局部结合的
        {
            union_kdb_localkdb = true;
        } else
        {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

target-learning::~target-learning(void)
{
}

void target-learning::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass
{
public:

    miCmpClass(std::vector<float> *m)
    {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b)
    {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};

void target-learning::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions


    parentsGeneral.resize(noCatAtts);

    for (CategoricalAttribute a = 0; a < noCatAtts; a++)
    {
        parentsGeneral[a].clear();

    }

    /*初始化各数据结构空间*/
    dist.reset(is);

    trainingIsFinished_ = false;

}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void target-learning::train(const instance &inst)
{
    dist.update(inst);


}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void target-learning::initialisePass()
{
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void target-learning::finalisePass()
{
    assert(trainingIsFinished_ == false);
    //  printf("finalisePass\n");

    // calculate the mutual information from the xy distribution
    std::vector<float> mi;
    getMutualInformation(dist.xxyCounts.xyCounts, mi);

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist.xxyCounts, cmi);

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        order.push_back(a);
    }

    // assign the parents
    if (!order.empty())
    {
        miCmpClass cmp(&mi);

        std::sort(order.begin(), order.end(), cmp);

        // proper KDB assignment of parents
        for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++)
        {
            parentsGeneral[*it].push_back(order[0]);
            for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++)
            {
                // make parents into the top k attributes on mi that precede *it in order
                if (parentsGeneral[*it].size() < k_)
                {
                    // create space for another parent
                    // set it initially to the new parent.
                    // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                    parentsGeneral[*it].push_back(*it2);
                }
                for (unsigned int i = 0; i < parentsGeneral[*it].size(); i++)
                {
                    if (cmi[*it2][*it] > cmi[parentsGeneral[*it][i]][*it])
                    {
                        // move lower value parents down in order
                        for (unsigned int j = parentsGeneral[*it].size() - 1; j > i; j--)
                        {
                            parentsGeneral[*it][j] = parentsGeneral[*it][j - 1];
                        }
                        // insert the new att
                        parentsGeneral[*it][i] = *it2;
                        break;
                    }
                }
            }

            if (verbosity >= 2)
            {
                printf("%s parents: ", instanceStream_->getCatAttName(*it));
                for (unsigned int i = 0; i < parentsGeneral[*it].size(); i++)
                {
                    printf("%s ", instanceStream_->getCatAttName(parentsGeneral[*it][i]));
                }
                putchar('\n');
            }
        }
    }
    order.clear();


    trainingIsFinished_ = true;

}

/// true iff no more passes are required. updated by finalisePass()

bool target-learning::trainingIsFinished()
{
    return trainingIsFinished_;

}

void target-learning::classify(const instance& inst, std::vector<double> &posteriorDist)
{
    //printf("classify\n");


    //local代码开始


    //std::vector<CategoricalAttribute> LocalParentsVector; //保存通过local得到的父节点
    std::vector<std::vector<std::vector<CategoricalAttribute> > > LocalParentsVector; ///保存通过local得到的父节点
    for(CatValue y=0;y<noClasses_;y++){ ///初始化 LocalParentsVector
        std::vector< std::vector<CategoricalAttribute> > localparents;
        localparents.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++){
            localparents[a].resize(2);
            localparents[a][0]=NOPARENT;
            localparents[a][1]=NOPARENT;
        }
        LocalParentsVector.push_back(localparents);
    }

    //printf("buildvector1\n");
    std::vector<std::vector<CategoricalAttribute> > OrderLocalVector; ///排序序列
    ///建立结构
    for(CatValue y=0;y<noClasses_;y++){
        ///对属性进行排序，根据I(xi;y)从大到小
        std::vector<CategoricalAttribute> orderLocal;   ///保存排好的序列
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++){
            orderLocal.push_back(a);
        }
        std::vector<float> mi_loc;

        getMIxiylockdb2(dist, mi_loc, inst,y);//计算相应公式值
        //getMIxiyloc2(xxyDist_, mi_loc, inst,y);//计算相应公式值
        /*for(int i=0;i<noCatAtts_;i++){
            std::cout<<mi_loc[i]<<std::endl;
        }*/
        miCmpClass cmp(&mi_loc);    /// creates a comparator for mi_loc
        std::sort(orderLocal.begin(), orderLocal.end(), cmp);   ///降序排列
        OrderLocalVector.push_back(orderLocal); ///将类标签y产生的ordervector保存到总的OrderLocalVector

        ///开始建立父子关系

        std::vector<std::vector<CategoricalAttribute> > &localparents = LocalParentsVector[y];   ///建立引用标识
        //float maxMI = -std::numeric_limits<float>::max();
        float maxMI =0.0;
        CategoricalAttribute bestfather = NOPARENT;
        CategoricalAttribute secbestfather = NOPARENT;
        for(CategoricalAttribute child=1;child<noCatAtts_;child++){
            //maxMI = -std::numeric_limits<float>::max();  ///如果互信息小于0 就不加边
            maxMI = 0.0;
            bestfather = NOPARENT;
            secbestfather = NOPARENT;
            for(CategoricalAttribute father=0;father<child;father++){
                CategoricalAttribute xi = orderLocal[child];
                CategoricalAttribute xj = orderLocal[father];

                //if(mi_loc[xi]<0)break;//判断是否小于0

                CatValue vi = inst.getCatVal(xi);
                CatValue vj = inst.getCatVal(xj);

                float tempMI = Ixixjykdb2(dist,xi,vi,xj,vj,y);     //这里计算公式值
                //float tempMI = Ixixjy2(xxyDist_,xi,vi,xj,vj,y);     //这里计算公式值
                if(tempMI>maxMI){       //这里判断父节点互信息大于0，不加边&&(mi_loc[xj]>0)

                    bestfather = xj;
                    maxMI = tempMI;
                }
            }
            //printf("找到第一个父节点\n");
            //加循环找次大
            maxMI=0.0;
            for(CategoricalAttribute father=0;father<child;father++){
                CategoricalAttribute xi = orderLocal[child];
                CategoricalAttribute xj = orderLocal[father];

                //if(mi_loc[xi]<0)break;//判断是否小于0

                //判断是否小于2
                CatValue vi = inst.getCatVal(xi);
                CatValue vj = inst.getCatVal(xj);

                float tempMI = Ixixjykdb2(dist,xi,vi,xj,vj,y);     //这里计算公式值
                //float tempMI = Ixixjy2(xxyDist_,xi,vi,xj,vj,y);     //这里计算公式值
                if((tempMI>maxMI)&&(xj!=bestfather)){       //这里判断父节点互信息小于0，不加边&&(mi_loc[xj]>0)

                    secbestfather = xj;
                    maxMI = tempMI;
                }

            }
            //printf("找到第二个父节点\n");
            localparents[orderLocal[child]][0] = bestfather;
            localparents[orderLocal[child]][1] = secbestfather;
            //std::cout<<LocalParentsVector[y][orderLocal[child]]<<std::endl;
        }
    }


    ///计算classDist
    std::vector<double> localclassDist; ///保存通过local得到的类标签概率分布
    localclassDist.resize(noClasses_);  ///初始化
     for (CatValue y = 0; y < noClasses_; y++){
        localclassDist[y] = dist.xxyCounts.xyCounts.p(y);
    }

    for(CatValue y=0;y<noClasses_;y++){

        /*std::vector<float> mi_loc;
        getMIxiylockdb2(dist, mi_loc, inst,y);//计算相应公式值，用于判断是否小于0，小于0则该点孤立*/

        std::vector<std::vector<CategoricalAttribute> > &localparents = LocalParentsVector[y];   ///建立引用标识
        std::vector<CategoricalAttribute> &orderLocal = OrderLocalVector[y];    ///建立引用标识
        //std::cout<<LocalParentsVector[y][0]<<std::endl;
        for(CategoricalAttribute orderindex = 0;orderindex<noCatAtts_;orderindex++){
            ///orderindex为orderLocal下标
            //printf("计算local联合概率\n");
            const CategoricalAttribute x1 = orderLocal[orderindex];
            const CategoricalAttribute bparent = localparents[x1][0];
            const CategoricalAttribute cparent = localparents[x1][1];

            /*if(bparent!=NOPARENT)
                std::cout<<"1    "<<bparent<<std::endl;
            else std::cout<<"1noparent    "<<std::endl;
            if(cparent!=NOPARENT)
                std::cout<<"2    "<<cparent<<std::endl;
            else std::cout<<"2noparent    "<<std::endl;*/
            //std::cout<<"y  "<<y<<"   "<<mi_loc[x1]<<std::endl;

            /*if(mi_loc[x1]<0){
                localclassDist[y] *=dist.xxyCounts.xyCounts.p(x1,inst.getCatVal(x1));
            }else*/
            if((bparent == NOPARENT)&&(cparent==NOPARENT)){
                localclassDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
            }else if((bparent != NOPARENT)&&(cparent==NOPARENT)){
                localclassDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), bparent,
                        inst.getCatVal(bparent), y);
            }else
            {
                localclassDist[y] *=dist.p(x1,inst.getCatVal(x1),bparent,inst.getCatVal(bparent),cparent,inst.getCatVal(cparent),y);
            }


        }

    }



    normalise(localclassDist);//归一化localkdb
    /*for (int classno = 0; classno < noClasses_; classno++)
        {
           std::cout<<localclassDist[classno]<<std::endl;

        }*/
    if (union_kdb_localkdb == true)
    {
        //全局parentsGeneral的联合概率
        std::vector<double> posteriorDistGeneral;
        posteriorDistGeneral.assign(noClasses_, 0);

        for (CatValue y = 0; y < noClasses_; y++)
        {
            posteriorDistGeneral[y] = dist.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);
        }
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                if (parentsGeneral[x1].size() == 0)
                {
                    // printf("PARent=0  \n");
                    posteriorDistGeneral[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parentsGeneral[x1].size() == 1)
                {
                    //  printf("PARent=1  \n");
                    const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parentsGeneral[x1][0], inst.getCatVal(parentsGeneral[x1][0]));
                    if (totalCount1 == 0)
                    {
                        posteriorDistGeneral[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else
                    {
                        posteriorDistGeneral[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parentsGeneral[x1][0], inst.getCatVal(parentsGeneral[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parentsGeneral[x1].size() == 2)
                {
                    // printf("PARent=2  \n");
                     // count for instances x1=v1, x2=v2 看看第1个和第2个父结点是否都在训练集中有取值
                    const InstanceCount totalCount1 = dist.xxyCounts.getCount(parentsGeneral[x1][0], inst.getCatVal(parentsGeneral[x1][0]), parentsGeneral[x1][1], inst.getCatVal(parentsGeneral[x1][1]));
                    if (totalCount1 == 0)
                    {
                        const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parentsGeneral[x1][0], inst.getCatVal(parentsGeneral[x1][0]));
                        if (totalCount2 == 0)
                        {
                            posteriorDistGeneral[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else
                        {
                            posteriorDistGeneral[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parentsGeneral[x1][0], inst.getCatVal(parentsGeneral[x1][0]), y);
                        }
                    } else
                    {
                        posteriorDistGeneral[y] *= dist.p(x1, inst.getCatVal(x1), parentsGeneral[x1][0], inst.getCatVal(parentsGeneral[x1][0]), parentsGeneral[x1][1], inst.getCatVal(parentsGeneral[x1][1]), y);
                    }
                }
            }
        }
        normalise(posteriorDistGeneral);//归一化kdb

        //联合概率结合
        for (int classno = 0; classno < noClasses_; classno++)
        {
            //posteriorDist[classno] =localclassDist[classno];//单独local
            posteriorDist[classno] =(localclassDist[classno]+posteriorDistGeneral[classno])/2;//两个集成
        }

    }
}



