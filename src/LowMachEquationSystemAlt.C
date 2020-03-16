/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "LowMachEquationSystemAlt.h"
#include "AlgorithmDriver.h"
#include "AssembleCourantReynoldsElemAlgorithm.h"
#include "AssembleMomentumElemWallFunctionSolverAlgorithm.h"
#include "AssembleMomentumElemWallFunctionProjectedSolverAlgorithm.h"
#include "AssembleNodalGradAlgorithmDriver.h"
#include "AssembleNodalGradElemAlgorithm.h"
#include "AssembleNodalGradBoundaryAlgorithm.h"
#include "AssembleNodalGradPBoundaryAlgorithm.h"
#include "AssembleNodalGradNonConformalAlgorithm.h"
#include "AssembleNodalGradUAlgorithmDriver.h"
#include "AssembleNodalGradUElemAlgorithm.h"
#include "AssembleNodalGradUBoundaryAlgorithm.h"
#include "AssembleNodalGradUNonConformalAlgorithm.h"
#include "AuxFunctionAlgorithm.h"
#include "ComputeDynamicPressureAlgorithm.h"
#include "ComputeMdotAlgorithmDriver.h"
#include "ComputeMdotInflowAlgorithm.h"
#include "ComputeMdotElemAlgorithm.h"
#include "ComputeMdotElemOpenAlgorithm.h"
#include "ComputeWallFrictionVelocityAlgorithm.h"
#include "ComputeWallFrictionVelocityProjectedAlgorithm.h"
#include "ConstantAuxFunction.h"
#include "CopyFieldAlgorithm.h"
#include "DirichletBC.h"
#include "EffectiveDiffFluxCoeffAlgorithm.h"
#include "Enums.h"
#include "EquationSystem.h"
#include "EquationSystems.h"
#include "FieldFunctions.h"
#include "LinearSolver.h"
#include "LinearSolvers.h"
#include "LinearSystem.h"
#include "master_element/MasterElement.h"
#include "NaluEnv.h"
#include "NaluParsing.h"
#include "ProjectedNodalGradientEquationSystem.h"
#include "PostProcessingData.h"
#include "Realm.h"
#include "Realms.h"
#include "SurfaceForceAndMomentAlgorithmDriver.h"
#include "SurfaceForceAndMomentAlgorithm.h"
#include "SurfaceForceAndMomentWallFunctionAlgorithm.h"
#include "SurfaceForceAndMomentWallFunctionProjectedAlgorithm.h"
#include "Simulation.h"
#include "SolutionOptions.h"
#include "SolverAlgorithmDriver.h"
#include "TurbViscKsgsAlgorithm.h"
#include "TurbViscSmagorinskyAlgorithm.h"
#include "TurbViscSSTAlgorithm.h"
#include "TurbViscWaleAlgorithm.h"
#include "FixPressureAtNodeAlgorithm.h"
#include "FixPressureAtNodeInfo.h"
#include "WallFunctionParamsAlgorithmDriver.h"

// template for kernels
#include "AlgTraits.h"
#include "kernel/KernelBuilder.h"
#include "kernel/KernelBuilderLog.h"

// kernels
#include "kernel/ContinuityAdvElemKernel.h"
#include "kernel/ContinuityMassElemKernel.h"
#include "kernel/MomentumAdvDiffElemKernel.h"
#include "kernel/MomentumActuatorSrcElemKernel.h"
#include "kernel/MomentumBuoyancyBoussinesqSrcElemKernel.h"
#include "kernel/MomentumBuoyancySrcElemKernel.h"
#include "kernel/MomentumMassElemKernel.h"
#include "kernel/MomentumUpwAdvDiffElemKernel.h"

// bc kernels
#include "kernel/ContinuityInflowElemKernel.h"
#include "kernel/ContinuityOpenElemKernel.h"
#include "kernel/MomentumOpenAdvDiffElemKernel.h"
#include "kernel/MomentumSymmetryElemKernel.h"
#include "kernel/MomentumWallFunctionElemKernel.h"

// nso
#include "nso/MomentumNSOElemKernel.h"
#include "nso/MomentumNSOKeElemKernel.h"
#include "nso/MomentumNSOSijElemKernel.h"
#include "nso/MomentumNSOGradElemSuppAlg.h"

// hybrid turbulence
#include "kernel/MomentumHybridTurbElemKernel.h"

// overset
#include "overset/UpdateOversetFringeAlgorithmDriver.h"

// user function
#include "user_functions/ConvectingTaylorVortexVelocityAuxFunction.h"
#include "user_functions/ConvectingTaylorVortexPressureAuxFunction.h"
#include "user_functions/TornadoAuxFunction.h"

#include "user_functions/WindEnergyAuxFunction.h"
#include "user_functions/WindEnergyTaylorVortexAuxFunction.h"
#include "user_functions/WindEnergyTaylorVortexPressureAuxFunction.h"

#include "user_functions/SteadyTaylorVortexVelocityAuxFunction.h"
#include "user_functions/SteadyTaylorVortexPressureAuxFunction.h"

#include "user_functions/VariableDensityVelocityAuxFunction.h"
#include "user_functions/VariableDensityPressureAuxFunction.h"

#include "user_functions/TaylorGreenPressureAuxFunction.h"
#include "user_functions/TaylorGreenVelocityAuxFunction.h"

#include "user_functions/BoussinesqNonIsoVelocityAuxFunction.h"

#include "user_functions/SinProfileChannelFlowVelocityAuxFunction.h"
#include "user_functions/SinProfilePipeFlowVelocityAuxFunction.h"

#include "user_functions/ChannelFlowPerturbedPlugVelocityAuxFunction.h"

#include "user_functions/BoundaryLayerPerturbationAuxFunction.h"

#include "user_functions/KovasznayVelocityAuxFunction.h"
#include "user_functions/KovasznayPressureAuxFunction.h"

#include "user_functions/OneTwoTenVelocityAuxFunction.h"

#include "user_functions/PowerlawPipeVelocityAuxFunction.h"
#include "user_functions/PowerlawVelocityAuxFunction.h"

#include "user_functions/PulseVelocityAuxFunction.h"

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/util/SortAndUnique.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/Comm.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

#include <utils/StkHelpers.h>

// basic c++
#include <vector>


namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// LowMachEquationSystemAlt - manage the low Mach equation system (uvw_p)
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
LowMachEquationSystemAlt::LowMachEquationSystemAlt(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "LowMachEOSWrap","low_mach_alt"),
    density_(NULL),
    viscosity_(NULL),
    dualNodalVolume_(NULL),
    surfaceForceAndMomentAlgDriver_(NULL),
    isInit_(true)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create momentum and pressure
  momentumEqSys_= new MomentumEquationSystemAlt(eqSystems);
  continuityEqSys_ = new ContinuityEquationSystemAlt(eqSystems);

  // inform realm
  realm_.hasFluids_ = true;

  // message to user
  if ( realm_.realmUsesEdges_ )
    throw std::runtime_error("LowMachEquationSystemAlt dies not support edge-based");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
LowMachEquationSystemAlt::~LowMachEquationSystemAlt()
{
  if ( NULL != surfaceForceAndMomentAlgDriver_ )
    delete surfaceForceAndMomentAlgDriver_;

  std::vector<Algorithm *>::iterator iim;
  for( iim=dynamicPressureAlg_.begin(); iim!=dynamicPressureAlg_.end(); ++iim )
    delete *iim;
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::initialize()
{
  // let equation systems that are owned some information
  momentumEqSys_->convergenceTolerance_ = convergenceTolerance_;
  continuityEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields_alt ---------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::register_nodal_fields_alt(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // add properties; denisty needs to be a restart field
  const int numStates = realm_.number_of_states();
  density_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "density", numStates));
  stk::mesh::put_field_on_mesh(*density_, *part, nullptr);
  realm_.augment_restart_variable_list("density");

  viscosity_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*viscosity_, *part, nullptr);

  // push to property list
  realm_.augment_property_map(DENSITY_ID, density_);
  realm_.augment_property_map(VISCOSITY_ID, viscosity_);

  // dual nodal volume (should push up...)
  dualNodalVolume_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume"));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, *part, nullptr);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &densityN = density_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &densityNp1, &densityN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }

  // register the fringe nodal field 
  if ( realm_.query_for_overset() && realm_.has_mesh_motion() ) {
    ScalarFieldType *fringeNode
      = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "fringe_node"));
    stk::mesh::put_field_on_mesh(*fringeNode, *part, nullptr);
  }
}

//--------------------------------------------------------------------------
//-------- register_element_fields_alt -------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::register_element_fields_alt(
  stk::mesh::Part *part,
  const stk::topology &theTopo)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract master element and get scs points
  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meSCS->numIntPoints_;
  GenericFieldType *massFlowRate = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs"));
  stk::mesh::put_field_on_mesh(*massFlowRate, *part, numScsIp, nullptr);

  // register the intersected elemental field
  if ( realm_.query_for_overset() ) {
    const int sizeOfElemField = 1;
    GenericFieldType *intersectedElement
      = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "intersected_element"));
    stk::mesh::put_field_on_mesh(*intersectedElement, *part, sizeOfElemField, nullptr);
  }

  // provide mean element Peclet and Courant fields; always...
  GenericFieldType *elemReynolds
    = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "element_reynolds"));
  stk::mesh::put_field_on_mesh(*elemReynolds, *part, 1, nullptr);
  GenericFieldType *elemCourant
    = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "element_courant"));
  stk::mesh::put_field_on_mesh(*elemCourant, *part, 1, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &theTopo,
  const OpenBoundaryConditionData &openBCData)
{
  // register boundary data
  stk::mesh::MetaData &metaData = realm_.meta_data();

  const int nDim = metaData.spatial_dimension();

  VectorFieldType *velocityBC = &(metaData.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "open_velocity_bc"));
  stk::mesh::put_field_on_mesh(*velocityBC, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;

  // extract the value for user specified velocity and save off the AuxFunction
  AuxFunction *theAuxFuncUbc = NULL;
  std::string velocityName = "velocity";

  UserDataType theDataType = get_bc_data_type(userData, velocityName);
  if ( CONSTANT_UD == theDataType ) {
    Velocity ux = userData.u_;
    std::vector<double> userSpecUbc(nDim);
    userSpecUbc[0] = ux.ux_;
    userSpecUbc[1] = ux.uy_;
    if ( nDim > 2)
      userSpecUbc[2] = ux.uz_;
    
    // new it
    theAuxFuncUbc = new ConstantAuxFunction(0, nDim, userSpecUbc);
  }
  else if ( FUNCTION_UD == theDataType ) {
    // extract the name and possible params
    std::string fcnName = get_bc_function_name(userData, velocityName);
    std::vector<double> theParams = get_bc_function_params(userData, velocityName);
    // switch on the name found...
    if ( fcnName == "power_law" ) {
      theAuxFuncUbc = new PowerlawVelocityAuxFunction(0,nDim,theParams);
    }
    else {
      throw std::runtime_error("Only power_law supported");
    }
  }
  else {
    throw std::runtime_error("Invalid Open Data Specification; must provide const or fcn for velocity");
  }

  // create the bc data alg
  AuxFunctionAlgorithm *auxAlgUbc
    = new AuxFunctionAlgorithm(realm_, part,
                               velocityBC, theAuxFuncUbc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlgUbc);

  // extract the value for user specified pressure and save off the AuxFunction
  ScalarFieldType *pressureBC
    = &(metaData.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_bc"));
  stk::mesh::put_field_on_mesh(*pressureBC, *part, nullptr);
  
  Pressure pSpec = userData.p_;
  std::vector<double> userSpecPbc(1);
  userSpecPbc[0] = pSpec.pressure_;
  
  // new it
  ConstantAuxFunction *theAuxFuncPbc = new ConstantAuxFunction(0, 1, userSpecPbc);
  
  // bc data alg
  AuxFunctionAlgorithm *auxAlgPbc
    = new AuxFunctionAlgorithm(realm_, part,
                               pressureBC, theAuxFuncPbc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlgPbc);

  // mdot at open bc; register field
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsBip = meFC->numIntPoints_;
  GenericFieldType *mdotBip 
    = &(metaData.declare_field<GenericFieldType>(static_cast<stk::topology::rank_t>(metaData.side_rank()), 
                                                 "open_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*mdotBip, *part, numScsBip, nullptr);

  // pbip; always register (initial value of zero)
  std::vector<double> zeroVec(numScsBip,0.0);
  GenericFieldType *pBip 
    = &(metaData.declare_field<GenericFieldType>(static_cast<stk::topology::rank_t>(metaData.side_rank()), 
                                                 "dynamic_pressure"));
  stk::mesh::put_field_on_mesh(*pBip, *part, numScsBip, zeroVec.data());
  
  // check for total bc to create an algorithm
  if ( userData.useTotalP_ ) {
    Algorithm * dynamicAlg = new ComputeDynamicPressureAlgorithm(realm_, part, realm_.realmUsesEdges_);
    dynamicPressureAlg_.push_back(dynamicAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_surface_pp_algorithm -----------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::register_surface_pp_algorithm(
  const PostProcessingData &theData,
  stk::mesh::PartVector &partVector)
{
  const std::string thePhysics = theData.physics_;

  // register nodal fields in common
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  VectorFieldType *pressureForce =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "pressure_force"));
  stk::mesh::put_field_on_mesh(*pressureForce, stk::mesh::selectUnion(partVector), meta_data.spatial_dimension(), nullptr);
  ScalarFieldType *tauWall =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "tau_wall"));
  stk::mesh::put_field_on_mesh(*tauWall, stk::mesh::selectUnion(partVector), nullptr);
  ScalarFieldType *yplus =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "yplus"));
  stk::mesh::put_field_on_mesh(*yplus, stk::mesh::selectUnion(partVector), nullptr);
 
  // force output for these variables
  realm_.augment_output_variable_list(pressureForce->name());
  realm_.augment_output_variable_list(tauWall->name());
  realm_.augment_output_variable_list(yplus->name());

  if ( thePhysics == "surface_force_and_moment" ) {
    ScalarFieldType *assembledArea =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment"));
    stk::mesh::put_field_on_mesh(*assembledArea, stk::mesh::selectUnion(partVector), nullptr);
    if ( NULL == surfaceForceAndMomentAlgDriver_ )
      surfaceForceAndMomentAlgDriver_ = new SurfaceForceAndMomentAlgorithmDriver(realm_);
    SurfaceForceAndMomentAlgorithm *ppAlg
      = new SurfaceForceAndMomentAlgorithm(
          realm_, partVector, theData.outputFileName_, theData.frequency_,
          theData.parameters_, realm_.realmUsesEdges_, assembledArea);
    surfaceForceAndMomentAlgDriver_->algVec_.push_back(ppAlg);
  }
  else if ( thePhysics == "surface_force_and_moment_wall_function_projected" ) {
    ScalarFieldType *assembledArea =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment_wfp"));
    stk::mesh::put_field_on_mesh(*assembledArea, stk::mesh::selectUnion(partVector), nullptr);
    if ( NULL == surfaceForceAndMomentAlgDriver_ )
      surfaceForceAndMomentAlgDriver_ = new SurfaceForceAndMomentAlgorithmDriver(realm_);
    SurfaceForceAndMomentWallFunctionProjectedAlgorithm *ppAlg
      = new SurfaceForceAndMomentWallFunctionProjectedAlgorithm(
          realm_, partVector, theData.outputFileName_, theData.frequency_,
          theData.parameters_, realm_.realmUsesEdges_, assembledArea, momentumEqSys_->pointInfoVec_, momentumEqSys_->wallFunctionGhosting_);
    surfaceForceAndMomentAlgDriver_->algVec_.push_back(ppAlg);
  }
  else {
    throw std::runtime_error("LowMachEquationSystemAlt::register_surface_pp_algorithm:Error() Unrecognized pp algorithm name");       
  }
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::register_initial_condition_fcn(
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> > &theParams)
{
  // extract nDim
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // iterate map and check for name
  const std::string dofName = "velocity";
  std::map<std::string, std::string>::const_iterator iterName
    = theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    
    // save off the field (np1 state)
    VectorFieldType *velocityNp1 = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
    
    // create a few Aux things
    AuxFunction *theAuxFunc = NULL;
    AuxFunctionAlgorithm *auxAlg = NULL;
    std::vector<double> fcnParams;

    // extract the params
    std::map<std::string, std::vector<double> >::const_iterator iterParams
      = theParams.find(dofName);
    if (iterParams != theParams.end()) {
      fcnParams = (*iterParams).second;	
    }

    // query function name and create aux
    if ( fcnName == "wind_energy_taylor_vortex") {
      // create the function
      theAuxFunc = new WindEnergyTaylorVortexAuxFunction(0,nDim,fcnParams);
    }
    else if ( fcnName == "boundary_layer_perturbation") {
      theAuxFunc = new BoundaryLayerPerturbationAuxFunction(0,nDim,fcnParams);
    }
    else if (fcnName == "kovasznay") {
      theAuxFunc = new KovasznayVelocityAuxFunction(0, nDim);
    }
    else if ( fcnName == "SteadyTaylorVortex" ) {
      theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensity" ) {      
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensityNonIso" ) {      
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "OneTwoTenVelocity" ) {      
      theAuxFunc = new OneTwoTenVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "convecting_taylor_vortex" ) {
      theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,nDim); 
    }
    else if ( fcnName == "TaylorGreen"  ) {
      theAuxFunc = new TaylorGreenVelocityAuxFunction(0,nDim); 
    }
    else if ( fcnName == "BoussinesqNonIso" ) {
      theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "SinProfileChannelFlow" ) {
      theAuxFunc = new SinProfileChannelFlowVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "SinProfilePipeFlow" ) {
      theAuxFunc = new SinProfilePipeFlowVelocityAuxFunction(0,nDim,fcnParams);
    }
    else if ( fcnName == "power_law" ) {
      theAuxFunc = new PowerlawVelocityAuxFunction(0,nDim,fcnParams);
    }
    else if ( fcnName == "ChannelFlowPerturbedPlug" ) {
      theAuxFunc = new ChannelFlowPerturbedPlugVelocityAuxFunction(0,nDim,fcnParams);
    }
    else {
      throw std::runtime_error("InitialCondFunction::non-supported velocity IC"); 
    }

    // create the algorithm
    auxAlg = new AuxFunctionAlgorithm(realm_, part,
                                      velocityNp1, theAuxFunc,
                                      stk::topology::NODE_RANK);
    
    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
  }
}

void
LowMachEquationSystemAlt::pre_iter_work()
{
  momentumEqSys_->pre_iter_work();
  continuityEqSys_->pre_iter_work();
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::solve_and_update()
{
  // wrap timing
  double timeA, timeB;
  if ( isInit_ ) {
    timeA = NaluEnv::self().nalu_time();
    compute_dynamic_pressure();
    continuityEqSys_->compute_projected_nodal_gradient();
    continuityEqSys_->computeMdotAlgDriver_->execute();
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerMisc_ += (timeB-timeA);
    isInit_ = false;
  }
  
  // compute tvisc
  momentumEqSys_->tviscAlgDriver_->execute();

  // compute effective viscosity
  momentumEqSys_->diffFluxCoeffAlgDriver_->execute();

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // momentum assemble, load_complete and solve
    momentumEqSys_->assemble_and_solve(momentumEqSys_->uTmp_);

    // update all of velocity
    timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *momentumEqSys_->uTmp_,
      1.0, momentumEqSys_->velocity_->field_of_state(stk::mesh::StateNP1),
      realm_.get_activate_aura());
    timeB = NaluEnv::self().nalu_time();
    momentumEqSys_->timerAssemble_ += (timeB-timeA);

    // compute velocity relative to mesh with new velocity
    realm_.compute_vrtm();

    // continuity assemble, load_complete and solve
    continuityEqSys_->assemble_and_solve(continuityEqSys_->pTmp_);

    // update pressure
    timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *continuityEqSys_->pTmp_,
      1.0, *continuityEqSys_->pressure_,
      realm_.get_activate_aura());
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerAssemble_ += (timeB-timeA);

    // compute mdot
    timeA = NaluEnv::self().nalu_time();
    continuityEqSys_->computeMdotAlgDriver_->execute();
    compute_dynamic_pressure();
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerMisc_ += (timeB-timeA);

    // project nodal velocity
    timeA = NaluEnv::self().nalu_time();
    project_nodal_velocity();
    timeB = NaluEnv::self().nalu_time();
    timerMisc_ += (timeB-timeA);

    // compute velocity relative to mesh with new velocity
    realm_.compute_vrtm();

    // velocity gradients based on current values;
    // note timing of this algorithm relative to initial_work
    // we use this approach to avoid two evals per
    // solve/update since dudx is required for tke
    // production
    timeA = NaluEnv::self().nalu_time();
    momentumEqSys_->compute_projected_nodal_gradient();
    timeB = NaluEnv::self().nalu_time();
    momentumEqSys_->timerMisc_ += (timeB-timeA);
  }

  // process CFL/Reynolds
  momentumEqSys_->cflReyAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- compute_dynamic_pressure ----------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::compute_dynamic_pressure()
{
  std::vector<Algorithm *>::iterator ii;
  for( ii=dynamicPressureAlg_.begin(); ii!=dynamicPressureAlg_.end(); ++ii ) {
    (*ii)->execute();
  }
}

//--------------------------------------------------------------------------
//-------- project_nodal_velocity ------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::project_nodal_velocity()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // time step
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double projTimeScale = dt/gamma1;

  const int nDim = meta_data.spatial_dimension();

  // field that we need
  VectorFieldType *velocity = momentumEqSys_->velocity_;
  VectorFieldType &velocityNp1 = velocity->field_of_state(stk::mesh::StateNP1);
  VectorFieldType *uTmp = momentumEqSys_->uTmp_;
  VectorFieldType *dpdx = continuityEqSys_->dpdx_;
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  //==========================================================
  // save off dpdx to uTmp (do it everywhere)
  //==========================================================
 
  // selector (everywhere dpdx lives) and node_buckets 
  stk::mesh::Selector s_nodes = stk::mesh::selectField(*dpdx);
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_nodes );
  
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * ut = stk::mesh::field_data(*uTmp, b);
    double * dp = stk::mesh::field_data(*dpdx, b);
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const int offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        ut[offSet+j] = dp[offSet+j];
      }
    }
  }

  //==========================================================
  // safe to update pressure gradient
  //==========================================================
  continuityEqSys_->compute_projected_nodal_gradient();

  //==========================================================
  // project u, u^n+1 = u^k+1 - dt/rho*(Gjp^N+1 - uTmp);
  //==========================================================
  
  // selector and node_buckets (only projected nodes)
  stk::mesh::Selector s_projected_nodes
    = (!stk::mesh::selectUnion(momentumEqSys_->notProjectedPart_)) &
    stk::mesh::selectField(*dpdx);
  stk::mesh::BucketVector const& p_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_projected_nodes );
  
  // process loop
  for ( stk::mesh::BucketVector::const_iterator ib = p_node_buckets.begin() ;
        ib != p_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * uNp1 = stk::mesh::field_data(velocityNp1, b);
    double * ut = stk::mesh::field_data(*uTmp, b);
    double * dp = stk::mesh::field_data(*dpdx, b);
    double * rho = stk::mesh::field_data(densityNp1, b);
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // Get scaling factor
      const double fac = projTimeScale/rho[k];
      
      // projection step
      const size_t offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        const double gdpx = dp[offSet+j] - ut[offSet+j];
        uNp1[offSet+j] -= fac*gdpx;
      }
    }
  }
}

void
LowMachEquationSystemAlt::predict_state()
{
  // Does Nothing
}

//--------------------------------------------------------------------------
//-------- post_converged_work ---------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystemAlt::post_converged_work()
{
  if (NULL != surfaceForceAndMomentAlgDriver_){
    surfaceForceAndMomentAlgDriver_->execute();
  }
  
  // output mass closure
  continuityEqSys_->computeMdotAlgDriver_->provide_output();
}

//==========================================================================
// Class Definition
//==========================================================================
// MomentumEquationSystemAlt - manages uvw pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumEquationSystemAlt::MomentumEquationSystemAlt(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "MomentumEQS","momentum"),
    managePNG_(realm_.get_consistent_mass_matrix_png("velocity")),
    velocity_(nullptr),
    dudx_(nullptr),
    coordinates_(nullptr),
    uTmp_(nullptr),
    visc_(nullptr),
    tvisc_(nullptr),
    evisc_(nullptr),
    assembleNodalGradAlgDriver_(new AssembleNodalGradUAlgorithmDriver(realm_, "dudx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_)),
    tviscAlgDriver_(new AlgorithmDriver(realm_)),
    cflReyAlgDriver_(new AlgorithmDriver(realm_)),
    wallFunctionParamsAlgDriver_(nullptr),
    wallFunctionGhosting_(nullptr),
    projectedNodalGradEqs_(nullptr),
    firstPNGResidual_(0.0)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("velocity");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MOMENTUM);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("velocity");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for velocity: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
     manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MomentumEquationSystemAlt::~MomentumEquationSystemAlt()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
  delete tviscAlgDriver_;
  delete cflReyAlgDriver_;

  if ( NULL != wallFunctionParamsAlgDriver_)
    delete wallFunctionParamsAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::initial_work()
{
  // call base class method (BDF2 state management, etc)
  EquationSystem::initial_work();

  // proceed with a bunch of initial work; wrap in timer
  const double timeA = NaluEnv::self().nalu_time();
  realm_.compute_vrtm();
  compute_projected_nodal_gradient();
  compute_wall_function_params();
  tviscAlgDriver_->execute();
  diffFluxCoeffAlgDriver_->execute();
  cflReyAlgDriver_->execute();

  const double timeB = NaluEnv::self().nalu_time();
  timerMisc_ += (timeB-timeA);
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields_alt ---------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_nodal_fields_alt(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  velocity_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity", numStates));
  stk::mesh::put_field_on_mesh(*velocity_, *part, nDim, nullptr);
  realm_.augment_restart_variable_list("velocity");

  dudx_ =  &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx"));
  stk::mesh::put_field_on_mesh(*dudx_, *part, nDim*nDim, nullptr);

  // delta solution for linear solver
  uTmp_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "uTmp"));
  stk::mesh::put_field_on_mesh(*uTmp_, *part, nDim, nullptr);

  coordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

  visc_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  if ( realm_.is_turbulent() ) {
    tvisc_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
    stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);
    evisc_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_u"));
    stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);
  }

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    VectorFieldType &velocityN = velocity_->field_of_state(stk::mesh::StateN);
    VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
    
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &velocityNp1, &velocityN,
                               0, nDim,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }

  // register specialty fields for PNG
  if (managePNG_ ) {
    // create temp vector field for duidx that will hold the active dudx
    VectorFieldType *duidx =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "duidx"));
    stk::mesh::put_field_on_mesh(*duidx, *part, nDim, nullptr);
  }

  // speciality source
  if ( NULL != realm_.actuator_ ) {
    VectorFieldType *actuatorSource 
      =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "actuator_source"));
    VectorFieldType *actuatorSourceLHS
      =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "actuator_source_lhs"));
    ScalarFieldType *g
      =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "g"));
    stk::mesh::put_field_on_mesh(*actuatorSource, *part, nullptr);
    stk::mesh::put_field_on_mesh(*actuatorSourceLHS, *part, nullptr);
    stk::mesh::put_field_on_mesh(*g, *part, nullptr);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm_alt ---------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_interior_algorithm_alt(
  stk::mesh::Part *part, 
  const std::vector<std::string> &terms)
{
  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  
  // non-solver CFL alg
  std::map<AlgorithmType, Algorithm *>::iterator it
    = cflReyAlgDriver_->algMap_.find(algType);
  if ( it == cflReyAlgDriver_->algMap_.end() ) {
    AssembleCourantReynoldsElemAlgorithm*theAlg
      = new AssembleCourantReynoldsElemAlgorithm(realm_, part);
    cflReyAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator itgu
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( itgu == assembleNodalGradAlgDriver_->algMap_.end() ) {
      AssembleNodalGradUElemAlgorithm *theAlg = new AssembleNodalGradUElemAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itgu->second->partVec_.push_back(part);
    }
  }

  // Homogeneous kernel implementation  
  stk::topology partTopo = part->topology();
  auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
  
  AssembleElemSolverAlgorithm* solverAlg = nullptr;
  bool solverAlgWasBuilt = false;
  
  std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_solver_alg
    (*this, *part, solverAlgMap, part->name());
  
  ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
  auto& activeKernels = solverAlg->activeKernels_;
  
  if (solverAlgWasBuilt) {
    
    build_topo_kernel_if_requested_alt<MomentumMassElemKernel>
      (partTopo, *this, activeKernels, "momentum_time_derivative", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);
    
    build_topo_kernel_if_requested_alt<MomentumMassElemKernel>
      (partTopo, *this, activeKernels, "lumped_momentum_time_derivative", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);
    
    build_topo_kernel_if_requested_alt<MomentumAdvDiffElemKernel>
      (partTopo, *this, activeKernels, "advection_diffusion", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_,
       realm_.is_turbulent()? evisc_ : visc_,
       dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumUpwAdvDiffElemKernel>
      (partTopo, *this, activeKernels, "upw_advection_diffusion", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, this, velocity_,
       realm_.is_turbulent()? evisc_ : visc_, dudx_,
       dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumActuatorSrcElemKernel>
      (partTopo, *this, activeKernels, "actuator", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);
    
    build_topo_kernel_if_requested_alt<MomentumActuatorSrcElemKernel>
      (partTopo, *this, activeKernels, "lumped_actuator", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);
    
    build_topo_kernel_if_requested_alt<MomentumBuoyancySrcElemKernel>
      (partTopo, *this, activeKernels, "buoyancy", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumBuoyancyBoussinesqSrcElemKernel>
      (partTopo, *this, activeKernels, "buoyancy_boussinesq", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumNSOElemKernel>
      (partTopo, *this, activeKernels, "NSO_2ND", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       0.0, 0.0, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumNSOElemKernel>
      (partTopo, *this, activeKernels, "NSO_2ND_ALT", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       0.0, 1.0, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumNSOKeElemKernel>
      (partTopo, *this, activeKernels, "NSO_2ND_KE", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_, 0.0, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumNSOSijElemKernel>
      (partTopo, *this, activeKernels, "NSO_2ND_SIJ", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumNSOElemKernel>
      (partTopo, *this, activeKernels, "NSO_4TH", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       1.0, 0.0, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumNSOElemKernel>
      (partTopo, *this, activeKernels, "NSO_4TH_ALT", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       1.0, 1.0, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<MomentumNSOKeElemKernel>
      (partTopo, *this, activeKernels, "NSO_4TH_KE", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_, 1.0, dataPreReqs);
  }

  // effective viscosity alg
  if ( realm_.is_turbulent() ) {
    std::map<AlgorithmType, Algorithm *>::iterator itev =
      diffFluxCoeffAlgDriver_->algMap_.find(algType);
    if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
      EffectiveDiffFluxCoeffAlgorithm *theAlg
        = new EffectiveDiffFluxCoeffAlgorithm(realm_, part, visc_, tvisc_, evisc_, 1.0, 1.0);
      diffFluxCoeffAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itev->second->partVec_.push_back(part);
    }

    // deal with tvisc better? - possibly should be on EqSysManager?
    std::map<AlgorithmType, Algorithm *>::iterator it_tv =
      tviscAlgDriver_->algMap_.find(algType);
    if ( it_tv == tviscAlgDriver_->algMap_.end() ) {
      Algorithm * theAlg = NULL;
      switch (realm_.solutionOptions_->turbulenceModel_ ) {
        case KSGS:
          theAlg = new TurbViscKsgsAlgorithm(realm_, part);
          break;
        case SMAGORINSKY:
          theAlg = new TurbViscSmagorinskyAlgorithm(realm_, part);
          break;
        case WALE:
          theAlg = new TurbViscWaleAlgorithm(realm_, part);
          break;
        case SST: case SST_DES:
          theAlg = new TurbViscSSTAlgorithm(realm_, part);
          break;
        default:
          throw std::runtime_error("non-supported turb model");
      }
      tviscAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it_tv->second->partVec_.push_back(part);
    }
  }

}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // push mesh part
  notProjectedPart_.push_back(part);

  // algorithm type
  const AlgorithmType algType = INFLOW;

  // velocity np1
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  // register boundary data; velocity_bc
  VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);
  
  // extract the value for user specified velocity and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  std::string velocityName = "velocity";
  UserDataType theDataType = get_bc_data_type(userData, velocityName);

  AuxFunction *theAuxFunc = NULL;
  if ( CONSTANT_UD == theDataType ) {
    Velocity ux = userData.u_;
    std::vector<double> userSpec(nDim);
    userSpec[0] = ux.ux_;
    userSpec[1] = ux.uy_;
    if ( nDim > 2)
      userSpec[2] = ux.uz_;
    
    // new it
    theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
    
  }
  else if ( FUNCTION_UD == theDataType ) {
    // extract the name/params
    std::string fcnName = get_bc_function_name(userData, velocityName);
    std::vector<double> theParams = get_bc_function_params(userData, velocityName);

    // switch on the name found...
    if ( fcnName == "convecting_taylor_vortex" ) {
      theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "SteadyTaylorVortex" ) {
      theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensity" ) {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensityNonIso" ) {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if (fcnName == "TaylorGreen" ) {
      theAuxFunc = new TaylorGreenVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "BoussinesqNonIso") {
      theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
    }
    else if ( fcnName == "kovasznay") {
      theAuxFunc = new KovasznayVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "power_law" ) {
      theAuxFunc = new PowerlawVelocityAuxFunction(0,nDim,theParams);
    }
    else if ( fcnName == "power_law_pipe" ) {
      theAuxFunc = new PowerlawPipeVelocityAuxFunction(0,nDim,theParams);
    }
    else if ( fcnName == "pulse" ) {
      theAuxFunc = new PulseVelocityAuxFunction(0,nDim,theParams);
    }
    else {
      throw std::runtime_error("MomentumEquationSystemAlt::register_inflow_bc: limited functions supported");
    }
  }
  else {
    throw std::runtime_error("MomentumEquationSystemAlt::register_inflow_bc: only constant and user function supported");
  }
  
  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
			       theBcField, theAuxFunc,
			       stk::topology::NODE_RANK);
  
  // how to populate the field?
  if ( userData.externalData_ ) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxAlg);
  }
  else {
    // put it on bcData
    bcDataAlg_.push_back(auxAlg);
  }

  // copy velocity_bc to velocity np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &velocityNp1,
                             0, nDim,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);
  
  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, theBcField, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg
      = new DirichletBC(realm_, this, part, &velocityNp1, theBcField, 0, nDim);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  }
  else {
    itd->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  // register boundary data; open_velocity_bc
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "open_velocity_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  Velocity ux = userData.u_;
  std::vector<double> userSpec(nDim);
  userSpec[0] = ux.ux_;
  userSpec[1] = ux.uy_;
  if ( nDim > 2)
    userSpec[2] = ux.uz_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver for momentum open
  auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
  
  stk::topology elemTopo = get_elem_topo(realm_, *part);
  
  AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
  bool solverAlgWasBuilt = false;
  
  std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
    = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "open");
  
  auto& activeKernels = faceElemSolverAlg->activeKernels_;
  
  if (solverAlgWasBuilt) {
    build_face_elem_topo_kernel_automatic<MomentumOpenAdvDiffElemKernel>
      (partTopo, elemTopo, *this, activeKernels, "momentum_open",
       realm_.meta_data(), *realm_.solutionOptions_, this,
       velocity_, dudx_, realm_.is_turbulent() ? evisc_ : visc_,
       faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);      
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const WallBoundaryConditionData &wallBCData)
{
  // find out if this is a wall function approach
  WallUserData userData = wallBCData.userData_;
  const bool anyWallFunctionActivated = userData.wallFunctionApproach_ || userData.wallFunctionProjectedApproach_;
   
  // push mesh part
  if ( !anyWallFunctionActivated )
    notProjectedPart_.push_back(part);

  if ( !realm_.solutionOptions_->useConsolidatedBcSolverAlg_ )
    throw std::runtime_error("MomentumEquationSystemAlt must use consolidated bcs");

  // np1 velocity
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();
  
  const std::string bcFieldName = anyWallFunctionActivated ? "wall_velocity_bc" : "velocity_bc";

  // register boundary data; velocity_bc
  VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, bcFieldName));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  AuxFunction *theAuxFunc = NULL;
  std::string velocityName = "velocity";

  if ( bc_data_specified(userData, velocityName) ) {

    UserDataType theDataType = get_bc_data_type(userData, velocityName);
    if ( CONSTANT_UD == theDataType ) {
      // constant data type specification
      Velocity ux = userData.u_;
      std::vector<double> userSpec(nDim);
      userSpec[0] = ux.ux_;
      userSpec[1] = ux.uy_;
      if ( nDim > 2)
        userSpec[2] = ux.uz_;
      theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
    }
    else if ( FUNCTION_UD == theDataType ) {
      // extract the name and parameters (double and string)
      std::string fcnName = get_bc_function_name(userData, velocityName);
      // switch on the name found...
      if ( fcnName == "tornado" ) {
        theAuxFunc = new TornadoAuxFunction(0,nDim);
      }
      else if ( fcnName == "wind_energy" ) {
        std::vector<std::string> theStringParams  = get_bc_function_string_params(userData, velocityName);
     	theAuxFunc = new WindEnergyAuxFunction(0,nDim, theStringParams, realm_);
      }
      else {
        throw std::runtime_error("Only wind_energy and tornado user functions supported");
      }
    }
  }
  else {
    throw std::runtime_error("Invalid Wall Data Specification; must provide const or fcn for velocity");
  }

  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);

  // check to see if this is an FSI interface to determine how we handle velocity population
  if ( userData.isFsiInterface_ ) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxAlg);
  }
  else {
    bcDataAlg_.push_back(auxAlg);
  }
  
  // copy velocity_bc to velocity np1
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
			     theBcField, &velocityNp1,
			     0, nDim,
			     stk::topology::NODE_RANK);

  // wall function activity will only set dof velocity np1 wall value as an IC
  if ( anyWallFunctionActivated )
    realm_.initCondAlg_.push_back(theCopyAlg);
  else
    bcDataMapAlg_.push_back(theCopyAlg);
    
  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    const AlgorithmType algTypePNG = anyWallFunctionActivated ? WALL_FCN : WALL;
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algTypePNG);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, theBcField, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algTypePNG] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // Dirichlet or wall function bc
  if ( anyWallFunctionActivated ) {

    // register fields; nodal
    ScalarFieldType *assembledWallArea =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_wf"));
    stk::mesh::put_field_on_mesh(*assembledWallArea, *part, nullptr);

    ScalarFieldType *assembledWallNormalDistance=  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_normal_distance"));
    stk::mesh::put_field_on_mesh(*assembledWallNormalDistance, *part, nullptr);

    // integration point; size it based on number of boundary integration points
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(partTopo);
    const int numScsBip = meFC->numIntPoints_;

    stk::topology::rank_t sideRank = static_cast<stk::topology::rank_t>(meta_data.side_rank());
    GenericFieldType *wallFrictionVelocityBip 
      =  &(meta_data.declare_field<GenericFieldType>(sideRank, "wall_friction_velocity_bip"));
    stk::mesh::put_field_on_mesh(*wallFrictionVelocityBip, *part, numScsBip, nullptr);

    GenericFieldType *wallNormalDistanceBip 
      =  &(meta_data.declare_field<GenericFieldType>(sideRank, "wall_normal_distance_bip"));
    stk::mesh::put_field_on_mesh(*wallNormalDistanceBip, *part, numScsBip, nullptr);

    // create wallFunctionParamsAlgDriver
    if ( NULL == wallFunctionParamsAlgDriver_) 
      wallFunctionParamsAlgDriver_ = new WallFunctionParamsAlgorithmDriver(realm_);
    
    const AlgorithmType wfAlgType = WALL_FCN;
    const AlgorithmType wfAlgProjectedType = WALL_FCN_PROJ;
    
    // create algorithm for utau, yp and assembled nodal wall area, and assembled wall normal distance
    if ( userData.wallFunctionApproach_ ) {
      std::map<AlgorithmType, Algorithm *>::iterator it_utau =
        wallFunctionParamsAlgDriver_->algMap_.find(wfAlgType);
      if ( it_utau == wallFunctionParamsAlgDriver_->algMap_.end() ) {
        ComputeWallFrictionVelocityAlgorithm *theUtauAlg =
          new ComputeWallFrictionVelocityAlgorithm(realm_, part, realm_.realmUsesEdges_);
        wallFunctionParamsAlgDriver_->algMap_[wfAlgType] = theUtauAlg;
      }
      else {
        it_utau->second->partVec_.push_back(part);
      }
    }
    else {
      // first extract projected distance
      const double projectedDistance = userData.projectedDistance_;
      std::map<AlgorithmType, Algorithm *>::iterator it_utau =
        wallFunctionParamsAlgDriver_->algMap_.find(wfAlgProjectedType);
      if ( it_utau == wallFunctionParamsAlgDriver_->algMap_.end() ) {
        ComputeWallFrictionVelocityProjectedAlgorithm *theUtauAlg =
          new ComputeWallFrictionVelocityProjectedAlgorithm(realm_, part, projectedDistance, realm_.realmUsesEdges_, 
                                                            pointInfoVec_, wallFunctionGhosting_);
        wallFunctionParamsAlgDriver_->algMap_[wfAlgProjectedType] = theUtauAlg;
      }
      else {
        // push back part and projected distance
        it_utau->second->partVec_.push_back(part);
        it_utau->second->set_data(projectedDistance);
      }
    }
  
    // create lhs/rhs algorithm; generalized for edge (nearest node usage) and element
    if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ && !userData.wallFunctionProjectedApproach_) {        
      // element-based uses consolidated approach fully
      auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
      
      AssembleElemSolverAlgorithm* solverAlg = nullptr;
      bool solverAlgWasBuilt = false;
      
      std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "wall_fcn");
      
      ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
      auto& activeKernels = solverAlg->activeKernels_;
      
      if (solverAlgWasBuilt) {
        build_face_topo_kernel_automatic<MomentumWallFunctionElemKernel>
          (partTopo, *this, activeKernels, "momentum_wall_function",
           realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);
        report_built_supp_alg_names();   
      }
    }
    else {
      // element-based using non-consolidated projected
      std::map<AlgorithmType, SolverAlgorithm *>::iterator it_wf =
          solverAlgDriver_->solverAlgMap_.find(wfAlgProjectedType);
      if ( it_wf == solverAlgDriver_->solverAlgMap_.end() ) {
        AssembleMomentumElemWallFunctionProjectedSolverAlgorithm *theAlg 
          = new AssembleMomentumElemWallFunctionProjectedSolverAlgorithm(realm_, part, this, realm_.realmUsesEdges_, pointInfoVec_, wallFunctionGhosting_);
        solverAlgDriver_->solverAlgMap_[wfAlgProjectedType] = theAlg;
      }
      else {
        it_wf->second->partVec_.push_back(part);
      }
    }
  }
  else {
    const AlgorithmType algType = WALL;
    
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
      DirichletBC *theAlg
        = new DirichletBC(realm_, this, part, &velocityNp1, theBcField, 0, nDim);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    }
    else {
      itd->second->partVec_.push_back(part);
    }
  }
  
  // specialty FSI
  if ( userData.isFsiInterface_ ) {
    // FIXME: need p^n+1/2; requires "old" pressure... need a utility to save it and compute it...
    NaluEnv::self().naluOutputP0() << "Warning: Second-order FSI requires p^n+1/2; BC is using p^n+1" << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const SymmetryBoundaryConditionData &/*symmetryBCData*/)
{
  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
  
  stk::topology elemTopo = get_elem_topo(realm_, *part);
  
  AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
  bool solverAlgWasBuilt = false;
  
  std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
    = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "symm");
  
  auto& activeKernels = faceElemSolverAlg->activeKernels_;
  
  if (solverAlgWasBuilt) {
    
    const stk::mesh::MetaData& metaData = realm_.meta_data();
    const std::string viscName = realm_.is_turbulent()
      ? "effective_viscosity_u" : "viscosity";
    
    build_face_elem_topo_kernel_automatic<MomentumSymmetryElemKernel>
      (partTopo, elemTopo, *this, activeKernels, "momentum_symmetry",
       metaData, *realm_.solutionOptions_,
       metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity"),
       metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName),
       faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_
       );    
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &theTopo)
{
  // skip for now
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::register_overset_bc()
{
  create_constraint_algorithm(velocity_);

  int nDim = realm_.meta_data().spatial_dimension();
  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);
  theAlg->fields_.push_back(std::unique_ptr<OversetFieldData>(new OversetFieldData(velocity_,1,nDim)));
  
  if ( realm_.has_mesh_motion() ) {
    UpdateOversetFringeAlgorithmDriver* theAlgPost = new UpdateOversetFringeAlgorithmDriver(realm_,false);
    // Perform fringe updates after all equation system solves (ideally on the post_time_step)
    equationSystems_.postIterAlgDriver_.push_back(theAlgPost);
    theAlgPost->fields_.push_back(std::unique_ptr<OversetFieldData>(new OversetFieldData(velocity_,1,nDim)));
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_MOMENTUM;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("velocity");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MOMENTUM);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // initialize new solver
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}


//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::predict_state()
{
  // copy state n to state np1
  VectorFieldType &uN = velocity_->field_of_state(stk::mesh::StateN);
  VectorFieldType &uNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), uN, uNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- compute_wall_function_params ------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::compute_wall_function_params()
{
  if (NULL != wallFunctionParamsAlgDriver_){
    wallFunctionParamsAlgDriver_->execute();
  }
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_U, "duidx", "qTmp", "pTmp", "PNGradUEQS");

    // turn off output
    projectedNodalGradEqs_->deactivate_output();
  }
  // fill the map for expected boundary condition names; recycle pTmp (ui copied in as needed)
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "pTmp");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "pTmp"); // might want wall_function velocity_bc?
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "pTmp");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "pTmp");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystemAlt::compute_projected_nodal_gradient()
{
  if ( !managePNG_ ) {
    const double timeA = -NaluEnv::self().nalu_time();
    assembleNodalGradAlgDriver_->execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  }
  else {
    // this option is more complex... Rather than solving a nDim*nDim system, we
    // copy each velocity component i to the expected dof for the PNG system; pTmp

    // extract fields
    ScalarFieldType *pTmp = realm_.meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp");
    VectorFieldType *duidx = realm_.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "duidx");

    const int nDim = realm_.meta_data().spatial_dimension();

    // manage norms here
    bool isFirst = realm_.currentNonlinearIteration_ == 1;
    if ( isFirst )
      firstPNGResidual_ = 0.0;

    double sumNonlinearResidual = 0.0;
    double sumLinearResidual = 0.0;
    int sumLinearIterations = 0;
    for ( int i = 0; i < nDim; ++i ) {
      // copy velocity, component i to pTmp
      field_index_copy(realm_.meta_data(), realm_.bulk_data(), *velocity_, i, *pTmp, 0,
        realm_.get_activate_aura());

      // copy active tensor, dudx to vector, duidx
      for ( int k = 0; k < nDim; ++k ) {
        field_index_copy(realm_.meta_data(), realm_.bulk_data(), *dudx_, i*nDim+k, *duidx, k,
          realm_.get_activate_aura());
      }

      projectedNodalGradEqs_->solve_and_update_external();

      // extract the solver history info
      const double nonlinearRes = projectedNodalGradEqs_->linsys_->nonLinearResidual();
      const double linearRes = projectedNodalGradEqs_->linsys_->linearResidual();
      const int linearIter = projectedNodalGradEqs_->linsys_->linearSolveIterations();

      // sum system norms for this iteration
      sumNonlinearResidual += nonlinearRes;
      sumLinearResidual += linearRes;
      sumLinearIterations += linearIter;

      // increment first nonlinear residual
      if ( isFirst )
        firstPNGResidual_ += nonlinearRes;

      // copy vector, duidx_k to tensor, dudx; this one might hurt as compared to a specialty loop..
      for ( int k = 0; k < nDim; ++k ) {
        field_index_copy(realm_.meta_data(), realm_.bulk_data(), *duidx, k, *dudx_, nDim*i+k,
          realm_.get_activate_aura());
      }
    }

    // output norms
    const double scaledNonLinearResidual = sumNonlinearResidual/std::max(std::numeric_limits<double>::epsilon(), firstPNGResidual_);
    std::string pngName = projectedNodalGradEqs_->linsys_->name();
    const int nameOffset = pngName.length()+8;
    NaluEnv::self().naluOutputP0()
        << std::setw(nameOffset) << std::right << pngName
        << std::setw(32-nameOffset)  << std::right << sumLinearIterations/(int)nDim
        << std::setw(18) << std::right << sumLinearResidual/(int)nDim
        << std::setw(15) << std::right << sumNonlinearResidual/(int)nDim
        << std::setw(14) << std::right << scaledNonLinearResidual << std::endl;

    // a bit covert, provide linsys with the new norm which is the sum of all norms
    projectedNodalGradEqs_->linsys_->setNonLinearResidual(sumNonlinearResidual);
  }
}

//==========================================================================
// Class Definition
//==========================================================================
// ContinuityEquationSystemAlt - manages p pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ContinuityEquationSystemAlt::ContinuityEquationSystemAlt(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "ContinuityEQS","continuity"),
    managePNG_(realm_.get_consistent_mass_matrix_png("pressure")),
    pressure_(NULL),
    dpdx_(NULL),
    massFlowRate_(NULL),
    coordinates_(NULL),
    pTmp_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "pressure", "dpdx")),
    computeMdotAlgDriver_(new ComputeMdotAlgorithmDriver(realm_)),
    projectedNodalGradEqs_(NULL)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("pressure");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_CONTINUITY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("pressure");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for pressure: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.equationSystems_.equationSystemVector_.push_back(this);
  
  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ContinuityEquationSystemAlt::~ContinuityEquationSystemAlt()
{
  delete assembleNodalGradAlgDriver_;
  delete computeMdotAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields_alt ---------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_nodal_fields_alt(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // register dof; set it as a restart variable
  pressure_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure"));
  stk::mesh::put_field_on_mesh(*pressure_, *part, nullptr);
  realm_.augment_restart_variable_list("pressure");

  dpdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx"));
  stk::mesh::put_field_on_mesh(*dpdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta with other split systems
  pTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*pTmp_, *part, nullptr);

  coordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm_alt ---------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_interior_algorithm_alt(
  stk::mesh::Part *part, 
  const std::vector<std::string> &terms)
{

  // non-solver, dpdx
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }


  // pure element-based scheme
  
  // mdot
  std::map<AlgorithmType, Algorithm *>::iterator itc =
    computeMdotAlgDriver_->algMap_.find(algType);
  if ( itc == computeMdotAlgDriver_->algMap_.end() ) {
    ComputeMdotElemAlgorithm *theAlg
      = new ComputeMdotElemAlgorithm(realm_, part, realm_.realmUsesEdges_);
    computeMdotAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itc->second->partVec_.push_back(part);
  }
  
  // Homogeneous kernel implementation  
  stk::topology partTopo = part->topology();
  auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
  
  AssembleElemSolverAlgorithm* solverAlg = nullptr;
  bool solverAlgWasBuilt = false;
  
  std::tie(solverAlg, solverAlgWasBuilt) 
    = build_or_add_part_to_solver_alg(*this, *part, solverAlgMap, part->name());
  
  ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
  auto& activeKernels = solverAlg->activeKernels_;
  
  if (solverAlgWasBuilt) {
    build_topo_kernel_if_requested_alt<ContinuityMassElemKernel>
      (partTopo, *this, activeKernels, "density_time_derivative", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);
    
    build_topo_kernel_if_requested_alt<ContinuityMassElemKernel>
      (partTopo, *this, activeKernels, "lumped_density_time_derivative", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);
    
    build_topo_kernel_if_requested_alt<ContinuityAdvElemKernel>
      (partTopo, *this, activeKernels, "advection", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);    
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  // register boundary data; cont_velocity_bc
  VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "cont_velocity_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);
  
  // extract the value for user specified velocity and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  std::string velocityName = "velocity";
  UserDataType theDataType = get_bc_data_type(userData, velocityName);
  
  AuxFunction *theAuxFunc = NULL;
  if ( CONSTANT_UD == theDataType ) {
    Velocity ux = userData.u_;
    std::vector<double> userSpec(nDim);
    userSpec[0] = ux.ux_;
    userSpec[1] = ux.uy_;
    if ( nDim > 2)
      userSpec[2] = ux.uz_;
    
    // new it
    theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);    
  }
  else if ( FUNCTION_UD == theDataType ) {
    // extract the name/params
    std::string fcnName = get_bc_function_name(userData, velocityName);
    std::vector<double> theParams = get_bc_function_params(userData, velocityName);

    // switch on the name found...
    if ( fcnName == "convecting_taylor_vortex" ) {
      theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "SteadyTaylorVortex" ) {
      theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensity" ) {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensityNonIso" ) {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "kovasznay") {
      theAuxFunc = new KovasznayVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "TaylorGreen") {
      theAuxFunc = new TaylorGreenVelocityAuxFunction(0, nDim);
    }
    else if ( fcnName == "BoussinesqNonIso") {
      theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
    }
    else if ( fcnName == "power_law" ) {
      theAuxFunc = new PowerlawVelocityAuxFunction(0,nDim,theParams);
    }
    else if ( fcnName == "power_law_pipe" ) {
      theAuxFunc = new PowerlawPipeVelocityAuxFunction(0,nDim,theParams);
    }
    else if ( fcnName == "pulse" ) {
      theAuxFunc = new PulseVelocityAuxFunction(0,nDim,theParams);
    }
    else {
      throw std::runtime_error("ContEquationSystem::register_inflow_bc: limited functions supported");
    }
  }
  else {
    throw std::runtime_error("ContEquationSystemAlt::register_inflow_bc: only constant and user function supported");
  }
  
  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  
  // how to populate the field?
  if ( userData.externalData_ ) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxAlg);
  }
  else {
    // put it on bcData
    bcDataAlg_.push_back(auxAlg);
  }

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // check to see if we are using shifted as inflow is shared
  const bool useShifted = realm_.get_cvfem_shifted_mdot();

  // non-solver inflow mdot - shared by both elem/edge
  std::map<AlgorithmType, Algorithm *>::iterator itmd =
    computeMdotAlgDriver_->algMap_.find(algType);
  if ( itmd == computeMdotAlgDriver_->algMap_.end() ) {
    ComputeMdotInflowAlgorithm *theAlg
      = new ComputeMdotInflowAlgorithm(realm_, part, useShifted);
    computeMdotAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itmd->second->partVec_.push_back(part);
  }
  
  // solver; lhs
  auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
  
  AssembleElemSolverAlgorithm* solverAlg = nullptr;
  bool solverAlgWasBuilt = false;
  
  std::tie(solverAlg, solverAlgWasBuilt) 
    = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "inflow");
  
  ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
  auto& activeKernels = solverAlg->activeKernels_;
  
  if (solverAlgWasBuilt) {
    build_face_topo_kernel_automatic<ContinuityInflowElemKernel>
      (partTopo, *this, activeKernels, "continuity_inflow",
       realm_.bulk_data(), *realm_.solutionOptions_, useShifted, dataPreReqs);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  const AlgorithmType algType = OPEN;

  // register boundary data
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  ScalarFieldType *pressureBC 
    = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_bc"));
    stk::mesh::put_field_on_mesh(*pressureBC, *part, nullptr);

  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradPBoundaryAlgorithm(realm_, part, pressureBC == NULL ? pressure_ : pressureBC, 
                                                  &dpdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // shared non-solver elem alg; compute open mdot
  std::map<AlgorithmType, Algorithm *>::iterator itm =
    computeMdotAlgDriver_->algMap_.find(algType);
  if ( itm == computeMdotAlgDriver_->algMap_.end() ) {
    ComputeMdotElemOpenAlgorithm *theAlg
      = new ComputeMdotElemOpenAlgorithm(realm_, part);
    computeMdotAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itm->second->partVec_.push_back(part);
  }
    
  // solver for continuity open
  auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
  
  stk::topology elemTopo = get_elem_topo(realm_, *part);
  
  AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
  bool solverAlgWasBuilt = false;
  
  std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
    = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "open");
  
  auto& activeKernels = faceElemSolverAlg->activeKernels_;
  
  if (solverAlgWasBuilt) {
    
    build_face_elem_topo_kernel_automatic<ContinuityOpenElemKernel>
      (partTopo, elemTopo, *this, activeKernels, "continuity_open",
       realm_.meta_data(), *realm_.solutionOptions_,
       faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);
    
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData &symmetryBCData)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &theTopo)
{
  // skip for now
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_overset_bc()
{
  create_constraint_algorithm(pressure_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  // manage pressure; variable density requires a pre-timestep evaluation of independent variables
  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(pressure_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::initialize()
{
  if (realm_.solutionOptions_->needPressureReference_) {
    const AlgorithmType algType = REF_PRESSURE;
    // Process parts if necessary
    realm_.solutionOptions_->fixPressureInfo_->create_part_vector(realm_.meta_data());
    stk::mesh::PartVector& pvec = realm_.solutionOptions_->fixPressureInfo_->partVec_;

    // The user could have provided just a Node ID instead of a part vector
    stk::mesh::Part* firstPart = pvec.size() > 0? pvec.at(0) : nullptr;

    auto it = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverDirichAlgMap_.end()) {
      FixPressureAtNodeAlgorithm* theAlg = new FixPressureAtNodeAlgorithm(
        realm_, firstPart, this);
      // populate the remaining parts if necessary
      for(size_t i=1; i < pvec.size(); i++)
        theAlg->partVec_.push_back( pvec[i]);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    } else {
      throw std::runtime_error("ContinuityEquationSystemAlt::initialize: logic error. Multiple initializations of FixPressureAtNodeAlgorithm.");
    }
  }

  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_CONTINUITY;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("pressure");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_CONTINUITY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::register_initial_condition_fcn(
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> >& theParams)
{
  // iterate map and check for name
  const std::string dofName = "pressure";
  std::map<std::string, std::string>::const_iterator iterName
    = theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    AuxFunction *theAuxFunc = NULL;
    if ( fcnName == "convecting_taylor_vortex" ) {
      // create the function
      theAuxFunc = new ConvectingTaylorVortexPressureAuxFunction();      
    }
    else if ( fcnName == "wind_energy_taylor_vortex") {
      // extract the params
      auto iterParams = theParams.find(dofName);
      std::vector<double> fcnParams = (iterParams != theParams.end()) ? (*iterParams).second : std::vector<double>();
      theAuxFunc = new WindEnergyTaylorVortexPressureAuxFunction(fcnParams);
    }
    else if ( fcnName == "SteadyTaylorVortex" ) {
      // create the function
      theAuxFunc = new SteadyTaylorVortexPressureAuxFunction();      
    }
    else if ( fcnName == "VariableDensity" ) {
      // create the function
      theAuxFunc = new VariableDensityPressureAuxFunction();      
    }
    else if ( fcnName == "VariableDensityNonIso" ) {
      // create the function
      theAuxFunc = new VariableDensityPressureAuxFunction();      
    }
    else if ( fcnName == "TaylorGreen") {
      // create the function
      theAuxFunc = new TaylorGreenPressureAuxFunction();      
    }
    else if ( fcnName == "kovasznay" ) {
      theAuxFunc = new KovasznayPressureAuxFunction();
    }
    else {
      throw std::runtime_error("ContinuityEquationSystemAlt::register_initial_condition_fcn: limited functions supported");
    }
    
    // create the algorithm
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
				 pressure_, theAuxFunc,
				 stk::topology::NODE_RANK);
    
    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
    
  }
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_ 
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_P, "dpdx", "qTmp", "pressure", "PNGradPEQS");
  }
  // fill the map for expected boundary condition names...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "pressure");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "pressure");
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "pressure_bc");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "pressure");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystemAlt::compute_projected_nodal_gradient()
{
  if ( !managePNG_ ) {
    const double timeA = -NaluEnv::self().nalu_time();
    assembleNodalGradAlgDriver_->execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  }
  else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

} // namespace nalu
} // namespace Sierra
