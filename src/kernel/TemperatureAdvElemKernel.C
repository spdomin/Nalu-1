/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/TemperatureAdvElemKernel.h"
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
TemperatureAdvElemKernel<AlgTraits>::TemperatureAdvElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* temperature,
  ScalarFieldType* density,
  ScalarFieldType* specificHeat,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    density_(density),
    specificHeat_(specificHeat),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  
  temperatureNp1_ = &(temperature->field_of_state(stk::mesh::StateNP1));

  if ( solnOpts.does_mesh_move() )
    velocity_ = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm"); // not tested
  else
    velocity_ = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");

  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  // compute shape function
  get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);
  
  // fields and data
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*velocity_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*temperatureNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);
  dataPreReqs.add_gathered_nodal_field(*specificHeat_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
}

template<typename AlgTraits>
TemperatureAdvElemKernel<AlgTraits>::~TemperatureAdvElemKernel()
{}

template<typename AlgTraits>
void
TemperatureAdvElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_rhoCpUj[AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_uNp1 = scratchViews.get_scratch_view_2D(*velocity_);
  SharedMemView<DoubleType*>& v_tNp1 = scratchViews.get_scratch_view_1D(
    *temperatureNp1_);
  SharedMemView<DoubleType*>& v_rho = scratchViews.get_scratch_view_1D(
    *density_);
  SharedMemView<DoubleType*>& v_cp = scratchViews.get_scratch_view_1D(
    *specificHeat_);

  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  SharedMemView<DoubleType***>& v_dndx = scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv;

  for ( int ip = 0; ip < AlgTraits::numScvIp_; ++ip ) {

    // nearest node to ip
    const int nearestNode = ipNodeMap_[ip];

    // zero out
    for ( int j = 0; j < AlgTraits::nDim_; ++j )
      w_rhoCpUj[j] = 0.0;

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      // save off scaling factor
      const DoubleType rhoCpScaling = v_shape_function_(ip,ic)*v_rho(ic)*v_cp(ic);

      // compute advecting velocity scaling
      for ( int j = 0; j < AlgTraits::nDim_; ++j )
        w_rhoCpUj[j] += rhoCpScaling*v_uNp1(ic,j);
    }

    // assemble the residual
    DoubleType residual = 0.0;    
    const DoubleType scV = v_scv_volume(ip);

    // manage LHS
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      DoubleType lhsSum = 0.0;
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        lhsSum += v_dndx(ip,ic,j)*w_rhoCpUj[j];
      }
      lhs(nearestNode,ic) += lhsSum*scV;
      residual += lhsSum*scV*v_tNp1(ic);
    }

    // assemble rhs
    rhs(nearestNode) -= residual;
  }
}
  
INSTANTIATE_KERNEL(TemperatureAdvElemKernel);

}  // nalu
}  // sierra
