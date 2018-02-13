#pragma once
#ifndef LOCALKDB_H
#define LOCALKDB_H
#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxyDist.h"
#include "xxxyDist.h"
#include "yDist.h"

class target-learning :  public IncrementalLearner
{
public:
  target-learning();
  target-learning(char*const*& argv, char*const* end);
  ~target-learning(void);

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
  xxxyDist dist; 
  std::vector<std::vector<CategoricalAttribute> > parentsGeneral;
  bool trainingIsFinished_;
  InstanceStream* instanceStream_;
};
#endif