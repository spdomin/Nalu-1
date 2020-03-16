/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "nso/TemperatureNSOElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

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
TemperatureNSOElemKernel<AlgTraits>::TemperatureNSOElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* temperature,
  ScalarFieldType* density,
  ScalarFieldType* specificHeat,
  ScalarFieldType* thermalCond,
  VectorFieldType* GjT,
  double altResFac,  
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    specificHeat_(specificHeat),
    thermalCond_(thermalCond),
    GjT_(GjT),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    altResFac_(altResFac),
    om_altResFac_(1.0 - altResFac),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(temperature->name())),
    gUpFac_( (AlgTraits::nodesPerElement_ == 4 || AlgTraits::nodesPerElement_ == 8 ) ? 0.25 : 1.0),
    gLwFac_( (AlgTraits::nodesPerElement_ == 4 || AlgTraits::nodesPerElement_ == 8 ) ? 4.00 : 1.0),
    nonCons_(0.0)
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();

  NaluEnv::self().naluOutputP0() << "gUpFac_/gLwFac_ are: " << gUpFac_ << "/" << gLwFac_ << std::endl;

  // deal with state
  temperatureN_ = &(temperature->field_of_state(stk::mesh::StateN));
  temperatureNp1_ = &(temperature->field_of_state(stk::mesh::StateNP1));
  if (temperature->number_of_states() == 2)
    temperatureNm1_ = temperatureN_;
  else
    temperatureNm1_ = &(temperature->field_of_state(stk::mesh::StateNM1));

  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  if (density->number_of_states() == 2)
    densityNm1_ = densityN_;
  else
    densityNm1_ = &(density->field_of_state(stk::mesh::StateNM1));
  
  if (solnOpts.does_mesh_move())
    velocityRTM_ = metaData.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = metaData.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity");

  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*velocityRTM_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*GjT_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*temperatureNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(*temperatureN_, 1);
  dataPreReqs.add_gathered_nodal_field(*temperatureNp1_, 1);

  dataPreReqs.add_gathered_nodal_field(*densityNm1_,1);
  dataPreReqs.add_gathered_nodal_field(*densityN_,1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_,1);
  dataPreReqs.add_gathered_nodal_field(*specificHeat_,1);
  dataPreReqs.add_gathered_nodal_field(*thermalCond_,1);
 
  // master element data
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCS_GIJ, CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
TemperatureNSOElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  dt_ = timeIntegrator.get_time_step();
  gamma1_ = timeIntegrator.get_gamma1();
  gamma2_ = timeIntegrator.get_gamma2();
  gamma3_ = timeIntegrator.get_gamma3(); // gamma3 may be zero
}

template<typename AlgTraits>
void
TemperatureNSOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_dTdxScs    [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_rhoCpVrtmScs [AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_GjT = scratchViews.get_scratch_view_2D(*GjT_);
  SharedMemView<DoubleType**>& v_velocityRTM = scratchViews.get_scratch_view_2D(*velocityRTM_);
  SharedMemView<DoubleType*>& v_TNm1 = scratchViews.get_scratch_view_1D(*temperatureNm1_);
  SharedMemView<DoubleType*>& v_TN = scratchViews.get_scratch_view_1D(*temperatureN_);
  SharedMemView<DoubleType*>& v_TNp1 = scratchViews.get_scratch_view_1D(*temperatureNp1_);
  SharedMemView<DoubleType*>& v_rhoNm1 = scratchViews.get_scratch_view_1D(*densityNm1_);
  SharedMemView<DoubleType*>& v_rhoN = scratchViews.get_scratch_view_1D(*densityN_);
  SharedMemView<DoubleType*>& v_rhoNp1 = scratchViews.get_scratch_view_1D(*densityNp1_);
  SharedMemView<DoubleType*>& v_specificHeat = scratchViews.get_scratch_view_1D(*specificHeat_);
  SharedMemView<DoubleType*>& v_thermalCond = scratchViews.get_scratch_view_1D(*thermalCond_);

  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted
    : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;
  SharedMemView<DoubleType***>& v_gijUpper = scratchViews.get_me_views(CURRENT_COORDINATES).gijUpper;
  SharedMemView<DoubleType***>& v_gijLower = scratchViews.get_me_views(CURRENT_COORDINATES).gijLower;

  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // zero out; scalar
    DoubleType TNm1Scs = 0.0;
    DoubleType TNScs = 0.0;
    DoubleType TNp1Scs = 0.0;
    DoubleType rhoNm1Scs = 0.0;
    DoubleType rhoNScs = 0.0;
    DoubleType rhoNp1Scs = 0.0;
    DoubleType cpScs = 0.0;
    DoubleType kScs = 0.0;
    DoubleType dFdxDiff = 0.0;
    DoubleType dFhatdx = 0.0;
    DoubleType dFdxCont = 0.0;
    DoubleType dFdxAdvCons = 0.0;

    // zero out vector
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      w_dTdxScs[i] = 0.0;
      w_rhoCpVrtmScs[i] = 0.0;
    }

    // determine scs values of interest
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      // save off shape function
      const DoubleType r = v_shape_function_(ip,ic);

      // time term; scalar q
      TNm1Scs += r*v_TNm1(ic);
      TNScs += r*v_TN(ic);
      TNp1Scs += r*v_TNp1(ic);

      // time term, density
      rhoNm1Scs += r*v_rhoNm1(ic);
      rhoNScs += r*v_rhoN(ic);
      rhoNp1Scs += r*v_rhoNp1(ic);
      cpScs += r*v_specificHeat(ic);
      kScs += r*v_thermalCond(ic);

      // compute scs derivatives and flux derivative
      const DoubleType TIC = v_TNp1(ic);
      const DoubleType rhoIC = v_rhoNp1(ic);
      const DoubleType cpIC = v_specificHeat(ic);
      const DoubleType kIC = v_thermalCond(ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType dnj = v_dndx(ip,ic,j);
        const DoubleType vrtmj = v_velocityRTM(ic,j);
        w_dTdxScs[j] += TIC*dnj;
        w_rhoCpVrtmScs[j] += r*rhoIC*cpIC*vrtmj;
        dFdxDiff += kIC*v_GjT(ic,j)*dnj;
        dFhatdx += rhoIC*cpScs*vrtmj*dnj;
        dFdxCont += rhoIC*vrtmj*dnj;
        dFdxAdvCons += rhoIC*cpIC*vrtmj*TIC*dnj;
      }
    }

    DoubleType dFdxAdv = 0.0;
    for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
      dFdxAdv += w_rhoCpVrtmScs[j]*w_dTdxScs[j];
    }

    const DoubleType contEq = nonCons_*((rhoNp1Scs*gamma1_ + gamma2_*rhoNScs + gamma3_*rhoNm1Scs)/dt_ + dFdxCont);

    // compute residual for NSO; pde-based and alternative
    const DoubleType time = rhoNp1Scs*cpScs*(gamma1_*TNp1Scs + gamma2_*TNScs + gamma3_*TNm1Scs)/dt_;
   
    const DoubleType residualPde = time + dFdxAdv - dFdxDiff + cpScs*TNp1Scs*contEq;
    const DoubleType residualAlt = dFdxAdv + cpScs*TNp1Scs*contEq - (dFdxAdvCons - TNp1Scs*dFhatdx) ;
    
    const DoubleType residual = altResFac_*residualAlt + om_altResFac_*residualPde;
    
    // denominator for nu as well as terms for "upwind" nu
    DoubleType gUpperMagGradQ = small_;
    DoubleType rhoCpVrtmiGLowerRhoCpVrtmj = 0.0;
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      const DoubleType dqdxScsi = w_dTdxScs[i];
      const DoubleType rhoCpVrtmi = w_rhoCpVrtmScs[i];
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        gUpperMagGradQ += dqdxScsi*v_gijUpper(ip,i,j)*gUpFac_*w_dTdxScs[j];
        rhoCpVrtmiGLowerRhoCpVrtmj += rhoCpVrtmi*v_gijLower(ip,i,j)*gLwFac_*w_rhoCpVrtmScs[j];
      }
    }
    
    // non-Shakib
    const DoubleType nuResidual = stk::math::sqrt((residual*residual)/(gUpperMagGradQ));
    
    // construct nu from first-order-like approach; SNL-internal write-up (eq 209)
    // for now, only include advection as full set of terms is too diffuse
    const DoubleType nuFirstOrder = stk::math::sqrt(rhoCpVrtmiGLowerRhoCpVrtmj);
    
    // limit based on first order; Cupw_ is a fudge factor similar to Guermond's approach
    const DoubleType nu = stk::math::min(Cupw_*nuFirstOrder, nuResidual);

    DoubleType gijFac = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      // save of some variables
      const DoubleType TIC = v_TNp1(ic);

      // NSO diffusion-like term; -nu*gUpper*(dQ/dxj - Gjq)*ai (residual below)
      DoubleType lhsfac = 0.0;
      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
        const DoubleType axi = v_scs_areav(ip,i);
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
          const DoubleType fac = v_gijUpper(ip,i,j)*gUpFac_*v_dndx(ip,ic,j)*axi;
          gijFac += fac*TIC;
          lhsfac += -fac;
        }
      }
      
      lhs(il,ic) += nu*lhsfac;
      lhs(ir,ic) -= nu*lhsfac;
    }
    
    // residual; left and right
    const DoubleType residualNSO = -nu*gijFac;
    rhs(il) -= residualNSO;
    rhs(ir) += residualNSO;
  }
}

INSTANTIATE_KERNEL(TemperatureNSOElemKernel);

}  // nalu
}  // sierra
