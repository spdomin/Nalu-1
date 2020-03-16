/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TemperatureNSOElemKernel_h
#define TemperatureNSOElemKernel_h

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** NSO for momentum equation
 *
 */
template<typename AlgTraits>
class TemperatureNSOElemKernel : public Kernel
{
public:
  TemperatureNSOElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    VectorFieldType*,
    double,
    ElemDataRequests&);

  virtual ~TemperatureNSOElemKernel() {}

  virtual void setup(const TimeIntegrator&);

  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  TemperatureNSOElemKernel() = delete;

  ScalarFieldType *temperatureNm1_{nullptr};
  ScalarFieldType *temperatureN_{nullptr};
  ScalarFieldType *temperatureNp1_{nullptr};
  ScalarFieldType *densityNm1_{nullptr};
  ScalarFieldType *densityN_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  ScalarFieldType *specificHeat_{nullptr};
  ScalarFieldType *thermalCond_{nullptr};
  VectorFieldType *GjT_{nullptr};
  VectorFieldType *velocityRTM_{nullptr};
  VectorFieldType *coordinates_{nullptr};
  
  const int *lrscv_;

  const double altResFac_;
  const double om_altResFac_;
 
  double dt_{0.0};
  double gamma1_{0.0};
  double gamma2_{0.0};
  double gamma3_{0.0};
  const double Cupw_{0.1};
  const bool shiftedGradOp_;
  const double small_{1.0e-16};
  const double gUpFac_;
  const double gLwFac_;
  const double nonCons_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

}  // nalu
}  // sierra

#endif /* TemperatureNSOElemKernel_h */
