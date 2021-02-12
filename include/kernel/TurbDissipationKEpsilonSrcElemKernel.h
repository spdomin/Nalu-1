/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TURBDISSIPATIONKEPSILONSRCELEMKERNEL_H
#define TURBDISSIPATIONKEPSILONSRCELEMKERNEL_H

#include "Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

template <typename AlgTraits>
class TurbDissipationKEpsilonSrcElemKernel : public Kernel
{
public:
  TurbDissipationKEpsilonSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    const bool);

  virtual ~TurbDissipationKEpsilonSrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  TurbDissipationKEpsilonSrcElemKernel() = delete;

  ScalarFieldType* tkeNp1_{nullptr};
  ScalarFieldType* epsNp1_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};
  VectorFieldType* velocityNp1_{nullptr};
  ScalarFieldType* tvisc_{nullptr};
  VectorFieldType* coordinates_{nullptr};

  const bool lumpedMass_;
  const bool shiftedGradOp_;
  const double cEpsOne_;
  const double cEpsTwo_;
  const double tkeProdLimitRatio_;
  const double includeDivU_;
  const double twoThirds_;

  /// Integration point to node mapping
  const int* ipNodeMap_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace sierra

#endif /* TURBDISSIPATIONKEPSILONSRCELEMKERNEL_H */
