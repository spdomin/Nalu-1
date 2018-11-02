/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef SurfaceForceAndMomentMLWallFunctionAlgorithm_h
#define SurfaceForceAndMomentMLWallFunctionAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class SurfaceForceAndMomentMLWallFunctionAlgorithm : public Algorithm
{
public:

  SurfaceForceAndMomentMLWallFunctionAlgorithm(
    Realm &realm,
    stk::mesh::PartVector &partVec,
    const std::string &outputFileName,
    const int &frequency_,
    const std::vector<double > &parameters,
    const bool &useShifted);
  ~SurfaceForceAndMomentMLWallFunctionAlgorithm();

  void execute();

  void pre_work();

  void cross_product(
    double *force, double *cross, double *rad);

  const std::string &outputFileName_;
  const int &frequency_;
  const std::vector<double > &parameters_;
  const bool useShifted_;

  VectorFieldType *coordinates_;
  ScalarFieldType *pressure_;
  VectorFieldType *pressureForce_;
  ScalarFieldType *tauWall_;
  VectorFieldType *vectorTauWall_;
  ScalarFieldType *yplus_;
  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  GenericFieldType *wallNormalDistanceBip_;
  GenericFieldType *vectorTauWallBip_;
  GenericFieldType *exposedAreaVec_;
  ScalarFieldType *assembledArea_;

  const int w_;
};

} // namespace nalu
} // namespace Sierra

#endif
