/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/TemperatureMassElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "TimeIntegrator.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
TemperatureMassElemKernel<AlgTraits>::TemperatureMassElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* temperature,
  ScalarFieldType* density,
  ScalarFieldType* specificHeat,
  ElemDataRequests& dataPreReqs,
  const bool lumpedMass)
  : Kernel(),
    density_(density),
    specificHeat_(specificHeat),
    lumpedMass_(lumpedMass),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  
  temperatureN_ = &(temperature->field_of_state(stk::mesh::StateN));
  temperatureNp1_ = &(temperature->field_of_state(stk::mesh::StateNP1));
  if (temperature->number_of_states() == 2)
    temperatureNm1_ = temperatureN_;
  else
    temperatureNm1_ = &(temperature->field_of_state(stk::mesh::StateNM1));

  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  // compute shape function
  if ( lumpedMass_ )
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shifted_shape_fcn(ptr);}, v_shape_function_);
  else
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields and data
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*temperatureNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(*temperatureN_, 1);
  dataPreReqs.add_gathered_nodal_field(*temperatureNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);
  dataPreReqs.add_gathered_nodal_field(*specificHeat_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}

template<typename AlgTraits>
TemperatureMassElemKernel<AlgTraits>::~TemperatureMassElemKernel()
{}

template<typename AlgTraits>
void
TemperatureMassElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  dt_ = timeIntegrator.get_time_step();
  gamma1_ = timeIntegrator.get_gamma1();
  gamma2_ = timeIntegrator.get_gamma2();
  gamma3_ = timeIntegrator.get_gamma3(); // gamma3 may be zero
}

template<typename AlgTraits>
void
TemperatureMassElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_tNm1 = scratchViews.get_scratch_view_1D(
    *temperatureNm1_);
  SharedMemView<DoubleType*>& v_tN = scratchViews.get_scratch_view_1D(
    *temperatureN_);
  SharedMemView<DoubleType*>& v_tNp1 = scratchViews.get_scratch_view_1D(
    *temperatureNp1_);
  SharedMemView<DoubleType*>& v_rho = scratchViews.get_scratch_view_1D(
    *density_);
  SharedMemView<DoubleType*>& v_cp = scratchViews.get_scratch_view_1D(
    *specificHeat_);

  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  for ( int ip = 0; ip < AlgTraits::numScvIp_; ++ip ) {

    // nearest node to ip
    const int nearestNode = ipNodeMap_[ip];

    // zero out; scalar
    DoubleType tNm1Scv = 0.0;
    DoubleType tNScv = 0.0;
    DoubleType tNp1Scv = 0.0;
    DoubleType rhoScv = 0.0;
    DoubleType cpScv = 0.0;

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      // save off shape function
      const DoubleType r = v_shape_function_(ip,ic);

      // temperature
      tNm1Scv += r*v_tNm1(ic);
      tNScv += r*v_tN(ic);
      tNp1Scv += r*v_tNp1(ic);

      // props
      rhoScv += r*v_rho(ic);
      cpScv += r*v_cp(ic);
    }

    // assemble rhs
    const DoubleType scV = v_scv_volume(ip);
    rhs(nearestNode) -=
      rhoScv*cpScv*(gamma1_*tNp1Scv + gamma2_*tNScv + gamma3_*tNm1Scv)*scV/dt_;
    
    // manage LHS
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType lhsfac = v_shape_function_(ip,ic)*gamma1_*rhoScv*cpScv*scV/dt_;
      lhs(nearestNode,ic) += lhsfac;
    }
  }
}
  
INSTANTIATE_KERNEL(TemperatureMassElemKernel);

}  // nalu
}  // sierra
