/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumMLWallFunctionElemKernel.h"
#include "master_element/MasterElement.h"
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

template<class BcAlgTraits>
MomentumMLWallFunctionElemKernel<BcAlgTraits>::MomentumMLWallFunctionElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    lhsFac_(0.0),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_)->ipNodeMap())
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  viscosity_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  exposedAreaVec_ = metaData.get_field<GenericFieldType>(metaData.side_rank(), "exposed_area_vector");
  vectorTauWall_ = metaData.get_field<GenericFieldType>(metaData.side_rank(), "vector_tau_wall");
  wallNormalDistanceBip_ = metaData.get_field<GenericFieldType>(metaData.side_rank(), "wall_normal_distance_bip");

  // add master elements
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_);
  dataPreReqs.add_cvfem_face_me(meFC);

  // compute and save shape function
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);
 
  // fields
  dataPreReqs.add_gathered_nodal_field(*viscosity_, 1);
  dataPreReqs.add_face_field(*exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  dataPreReqs.add_face_field(*vectorTauWall_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  dataPreReqs.add_face_field(*wallNormalDistanceBip_, BcAlgTraits::numFaceIp_);

  NaluEnv::self().naluOutputP0() << "ML momentum wall function in use; lhsFac_" << lhsFac_ << std::endl;
}

template<class BcAlgTraits>
MomentumMLWallFunctionElemKernel<BcAlgTraits>::~MomentumMLWallFunctionElemKernel()
{}

template<class BcAlgTraits>
void
MomentumMLWallFunctionElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
 
  SharedMemView<DoubleType**>& vf_exposedAreaVec = scratchViews.get_scratch_view_2D(*exposedAreaVec_);
  SharedMemView<DoubleType**>& vf_vTauW = scratchViews.get_scratch_view_2D(*vectorTauWall_);
  SharedMemView<DoubleType*>& vf_yp = scratchViews.get_scratch_view_1D(*wallNormalDistanceBip_);
  SharedMemView<DoubleType*>& v_viscosity = scratchViews.get_scratch_view_1D(*viscosity_);

  for ( int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip ) {
        
    const int nearestNode = ipNodeMap_[ip];
    
    // aMag
    DoubleType aMag = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      const DoubleType axj = vf_exposedAreaVec(ip,j);
      aMag += axj*axj;
    }
    aMag = stk::math::sqrt(aMag);
    
    // interpolate to bip
    DoubleType muBip = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      muBip += r*v_viscosity(ic);
    }

    // start the rhs/lhs assembly (optional approximate LHS)
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      const int indexR = nearestNode*BcAlgTraits::nDim_ + i;
      rhs(indexR) -= vf_vTauW(ip,i)*aMag;
      
      for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic)
        lhs(indexR,ic*BcAlgTraits::nDim_+i) += lhsFac_*muBip/vf_yp(ip)*vf_shape_function_(ip,ic)*aMag;
    }
  }  
}

INSTANTIATE_KERNEL_FACE(MomentumMLWallFunctionElemKernel);

}  // nalu
}  // sierra
