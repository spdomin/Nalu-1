/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TemperatureMassElemKernel_h
#define TemperatureMassElemKernel_h

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class TimeIntegrator;
class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** CMM (BDF2/BE) for scalar equation
 */
template<typename AlgTraits>
class TemperatureMassElemKernel: public Kernel
{
public:
  TemperatureMassElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&,
    const bool);

  virtual ~TemperatureMassElemKernel();

  /** Perform pre-timestep work for the computational kernel
   */
  virtual void setup(const TimeIntegrator&);

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  TemperatureMassElemKernel() = delete;

  ScalarFieldType *density_{nullptr};
  ScalarFieldType *specificHeat_{nullptr};
  ScalarFieldType *temperatureNm1_{nullptr};
  ScalarFieldType *temperatureN_{nullptr};
  ScalarFieldType *temperatureNp1_{nullptr};
  VectorFieldType *coordinates_{nullptr};

  const bool lumpedMass_;

  /// Integration point to node mapping
  const int* ipNodeMap_;

  double dt_{0.0};
  double gamma1_{0.0};
  double gamma2_{0.0};
  double gamma3_{0.0};
  
  /// Shape functions
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_ {"view_shape_func"};
};

}  // nalu
}  // sierra

#endif /* SCALARMASSELEMKERNEL_H */
