#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxyDist.h"
#include "xxxxyDist.h"
#include "yDist.h"

class k3db_ensemble :  public IncrementalLearner
{
public:
  k3db_ensemble();
  k3db_ensemble(char*const*& argv, char*const* end);
  ~k3db_ensemble(void);

  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c);

  virtual void classify(const instance &inst, std::vector<double> &classDist);

protected:
  unsigned int pass_;                                        ///< the number of passes for the learner
  unsigned int k_;                                           ///< the maximum number of parents
  unsigned int noCatAtts_;                                   ///< the number of categorical attributes.
  unsigned int noClasses_;                                   ///< the number of classes
  bool union_kdb_localkdb; 
  
  xxyDist dist_;                                             // used in the first pass
  xxxxyDist dist_1; 
  yDist classDist_;                                          // used in the second pass and for classification
  yDist classDist_1;   
  std::vector<CategoricalAttribute> order_;
  std::vector<distributionTree> dTree_;                      // used in the second pass and for classification
  std::vector<distributionTree> dTree_1;  
  std::vector<std::vector<CategoricalAttribute> > parents_;
  std::vector<std::vector<CategoricalAttribute> > parents_1;
  //std::vector<std::vector<CategoricalAttribute> > parents_2;
  
  bool trainingIsFinished_;
  InstanceStream* instanceStream_;
};