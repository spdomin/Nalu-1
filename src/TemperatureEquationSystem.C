/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "TemperatureEquationSystem.h"
#include "AlgorithmDriver.h"
#include "AssembleScalarElemSolverAlgorithm.h"
#include "AssembleScalarDiffNonConformalSolverAlgorithm.h"
#include "AssembleNodalGradAlgorithmDriver.h"
#include "AssembleNodalGradElemAlgorithm.h"
#include "AssembleNodalGradBoundaryAlgorithm.h"
#include "AssembleNodalGradNonConformalAlgorithm.h"
#include "AssembleNodeSolverAlgorithm.h"
#include "AuxFunctionAlgorithm.h"
#include "ConstantAuxFunction.h"
#include "CopyFieldAlgorithm.h"
#include "DirichletBC.h"
#include "EquationSystem.h"
#include "EquationSystems.h"
#include "Enums.h"
#include "FieldFunctions.h"
#include "LinearSolvers.h"
#include "LinearSolver.h"
#include "LinearSystem.h"
#include "NaluEnv.h"
#include "NaluParsing.h"
#include "ProjectedNodalGradientEquationSystem.h"
#include "Realm.h"
#include "Realms.h"
#include "HeatCondMassBackwardEulerNodeSuppAlg.h"
#include "HeatCondMassBDF2NodeSuppAlg.h"
#include "Simulation.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"
#include "SolverAlgorithmDriver.h"

// template for kernels
#include "AlgTraits.h"
#include "kernel/KernelBuilder.h"
#include "kernel/KernelBuilderLog.h"

// kernels
#include "kernel/TemperatureMassElemKernel.h"
#include "kernel/TemperatureAdvElemKernel.h"
#include "kernel/ScalarDiffElemKernel.h"
#include "nso/TemperatureNSOElemKernel.h"

// bc kernels - na

// props
#include "property_evaluator/ThermalConductivityFromPrandtlPropAlgorithm.h"

// nso
#include "nso/ScalarNSOElemKernel.h"

// user function - na

#include "overset/UpdateOversetFringeAlgorithmDriver.h"

// stk_util
#include <stk_util/parallel/Parallel.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// nalu utility
#include <utils/StkHelpers.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TemperatureEquationSystem - manages T pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TemperatureEquationSystem::TemperatureEquationSystem(
  EquationSystems& eqSystems,
  const double minT,
  const double maxT,
  const bool outputClippingDiag)
  : EquationSystem(eqSystems, "TemperatureEQS", "temperature"),
    minimumT_(minT),
    maximumT_(maxT),
    managePNG_(realm_.get_consistent_mass_matrix_png("temperature")),
    outputClippingDiag_(outputClippingDiag),
    temperature_(nullptr),
    dtdx_(nullptr),
    tTmp_(nullptr),
    density_(nullptr),
    specificHeat_(nullptr),
    thermalCond_(nullptr),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "temperature", "dtdx")),
    projectedNodalGradEqs_(nullptr),
    isInit_(true)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("temperature");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TEMPERATURE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("temperature");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for temperature: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // advertise as non isothermal
  realm_.isothermal_ = false;

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_projected_nodal_gradient(eqSystems);
  }

  // error if edge-based
  if ( realm_.realmUsesEdges_  )
    throw std::runtime_error("TemperatureEquationSystem::Error edge-based is not supported");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TemperatureEquationSystem::~TemperatureEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- populate_derived_quantities -------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::populate_derived_quantities()
{
  // placeholder
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields_alt ---------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::register_nodal_fields_alt(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  temperature_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature", numStates));
  stk::mesh::put_field_on_mesh(*temperature_, *part, nullptr);
  realm_.augment_restart_variable_list("temperature");

  dtdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dtdx"));
  stk::mesh::put_field_on_mesh(*dtdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta since this is a split system
  tTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*tTmp_, *part, nullptr);

  // add properties
  density_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "density", realm_.number_of_states()));
  stk::mesh::put_field_on_mesh(*density_, *part, nullptr);
  realm_.augment_property_map(DENSITY_ID, density_);
  
  specificHeat_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat"));
  stk::mesh::put_field_on_mesh(*specificHeat_, *part, nullptr);
  realm_.augment_property_map(SPEC_HEAT_ID, specificHeat_);
  
  thermalCond_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "thermal_conductivity"));
  stk::mesh::put_field_on_mesh(*thermalCond_, *part, nullptr);

  // check to see if Prandtl number was provided
  bool prProvided = false;
  const double providedPr = realm_.get_lam_prandtl("temperature", prProvided);
  if ( prProvided ) {
    // deal with viscosity first (declare, put, and prop)
    ScalarFieldType *viscosity =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
    stk::mesh::put_field_on_mesh(*viscosity, *part, nullptr);
    realm_.augment_property_map(VISCOSITY_ID, viscosity);

    // compute thermal conductivity using Pr; create and push back the algorithm
    NaluEnv::self().naluOutputP0() << "Laminar Prandtl provided; will compute Thermal conductivity based on this constant value" << std::endl;
    Algorithm *propAlg 
      = new ThermalConductivityFromPrandtlPropAlgorithm(realm_, part, thermalCond_, specificHeat_, viscosity, providedPr);
    propertyAlg_.push_back(propAlg);
  }
  else {
    // no Pr provided, simply augment property map and expect lambda to be provided in the input file
    realm_.augment_property_map(THERMAL_COND_ID, thermalCond_);
  }

  if ( realm_.is_turbulent() ) {
    throw std::runtime_error("TemperatureEquationSystem::Error turbulence is not supported");
  }

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &tempN = temperature_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &tempNp1 = temperature_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlgA
      = new CopyFieldAlgorithm(realm_, part,
                               &tempNp1, &tempN,
                               0, 1,
                               stk::topology::NODE_RANK);

    copyStateAlg_.push_back(theCopyAlgA);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm_alt ---------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::register_interior_algorithm_alt(
  stk::mesh::Part *part, 
  const std::vector<std::string> &terms)
{
  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &temperatureNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to projected nodal gradient; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &temperatureNp1, &dtdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    } 
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

    build_topo_kernel_if_requested_alt<TemperatureMassElemKernel>
      (partTopo, *this, activeKernels, "temperature_time_derivative", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, temperature_, density_, specificHeat_, dataPreReqs, false);

    build_topo_kernel_if_requested_alt<TemperatureMassElemKernel>
      (partTopo, *this, activeKernels, "lumped_temperature_time_derivative", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, temperature_, density_, specificHeat_, dataPreReqs, true);

    build_topo_kernel_if_requested_alt<TemperatureAdvElemKernel>
      (partTopo, *this, activeKernels, "advection", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, temperature_, density_, specificHeat_, dataPreReqs);

    build_topo_kernel_if_requested_alt<ScalarDiffElemKernel>
      (partTopo, *this, activeKernels, "diffusion", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, temperature_, thermalCond_, dataPreReqs);

    build_topo_kernel_if_requested_alt<TemperatureNSOElemKernel>
      (partTopo, *this, activeKernels, "nso", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, temperature_, density_, specificHeat_, thermalCond_, dtdx_, 0.0, dataPreReqs);
    
    build_topo_kernel_if_requested_alt<TemperatureNSOElemKernel>
      (partTopo, *this, activeKernels, "nso_alt", terms,
       realm_.bulk_data(), *realm_.solutionOptions_, temperature_, density_, specificHeat_, thermalCond_, dtdx_, 1.0, dataPreReqs);
  }

}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{
  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &temperatureNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; temperature_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified temperature and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  std::string temperatureName = "temperature";
  UserDataType theDataType = get_bc_data_type(userData, temperatureName);

  AuxFunction *theAuxFunc = NULL;
  if ( CONSTANT_UD == theDataType ) {
    Temperature temperature = userData.temperature_;
    std::vector<double> userSpec(1);
    userSpec[0] = temperature.temperature_;

    // new it
    theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
  }
  else {
    throw std::runtime_error("TemperatureEquationSystem::register_inflow_bc: only constant supported");
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

  // copy temperature_bc to temperature np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &temperatureNp1,
                             0, 1,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dtdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &temperatureNp1, &dtdxNone, edgeNodalGradient_);
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
      = new DirichletBC(realm_, this, part, &temperatureNp1, theBcField, 0, 1);
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
TemperatureEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData &openBCData)
{
  // algorithm type
  const AlgorithmType algType = OPEN;
  
  ScalarFieldType &temperatureNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dtdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &temperatureNp1, &dtdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // no current solver contributions for non-div temperature-form
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType &temperatureNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract the value for user specified temperature and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string temperatureName = "temperature";
  if ( bc_data_specified(userData, temperatureName) ) {

    // FIXME: Generalize for constant vs function

    // register boundary data; temperature_bc
    ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

    // extract data
    std::vector<double> userSpec(1);
    Temperature temperature = userData.temperature_;
    userSpec[0] = temperature.temperature_;

    // new it
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // copy temperature_bc to temperature np1...
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               theBcField, &temperatureNp1,
                               0, 1,
                               stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

    // Dirichlet bc
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
      DirichletBC *theAlg
        = new DirichletBC(realm_, this, part, &temperatureNp1, theBcField, 0, 1);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    }
    else {
      itd->second->partVec_.push_back(part);
    }
  }

  // non-solver; dtdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &temperatureNp1, &dtdxNone, edgeNodalGradient_);
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
TemperatureEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData &/*wallBCData*/)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType &temperatureNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dtdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &temperatureNp1, &dtdxNone, edgeNodalGradient_);
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
TemperatureEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{
  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType &tempNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dtdx; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tempNp1, &dtdxNone, edgeNodalGradient_);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
    else {
      // proceed with DG
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        AssembleNodalGradNonConformalAlgorithm *theAlg 
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, &tempNp1, &dtdxNone);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
  }
  
  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    AssembleScalarDiffNonConformalSolverAlgorithm *theAlg
      = new AssembleScalarDiffNonConformalSolverAlgorithm(realm_, part, this, temperature_, thermalCond_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    itsi->second->partVec_.push_back(part);
  }  
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(temperature_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(temperature_,1,1)));

  if ( realm_.has_mesh_motion() ) {
    UpdateOversetFringeAlgorithmDriver* theAlgPost = new UpdateOversetFringeAlgorithmDriver(realm_,false);
    // Perform fringe updates after all equation system solves (ideally on the post_time_step)
    equationSystems_.postIterAlgDriver_.push_back(theAlgPost);
    theAlgPost->fields_.push_back(std::unique_ptr<OversetFieldData>(new OversetFieldData(temperature_,1,1)));
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_TEMPERATURE;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("temperature");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TEMPERATURE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> > &/*theParams*/)
{
  // na
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::solve_and_update()
{

  // compute dt/dx
  if ( isInit_ ) {
    compute_projected_nodal_gradient();
    isInit_ = false;
  }

  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // mixture fraction assemble, load_complete and solve
    assemble_and_solve(tTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    update_and_clip();
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);

    // projected nodal gradient
    compute_projected_nodal_gradient();
  }
}

//--------------------------------------------------------------------------
//-------- update_and_clip -------------------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::update_and_clip()
{
  size_t numClip[2] = {0,0};
  double minT = +1.0e16;
  double maxT = -1.0e16;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*temperature_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    double *temperature = stk::mesh::field_data(*temperature_, b);
    double *tTmp    = stk::mesh::field_data(*tTmp_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      double temperatureNp1 = temperature[k] + tTmp[k];
      // clip now
      if ( temperatureNp1 < minimumT_ ) {
        minT = std::min(temperatureNp1, minT);
        temperatureNp1 = minimumT_;
        numClip[0]++;
      }
      else if ( temperatureNp1 > maximumT_ ) {
        maxT = std::max(temperatureNp1, maxT);
        temperatureNp1 = maximumT_;
        numClip[1]++;
      }
      temperature[k] = temperatureNp1;
    }
  }

  // parallel assemble clipped value
  if ( outputClippingDiag_ ) {
    size_t g_numClip[2] = {};
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
    stk::all_reduce_sum(comm, numClip, g_numClip, 2);

    if ( g_numClip[0] > 0 ) {
      double g_minT = 0;
      stk::all_reduce_min(comm, &minT, &g_minT, 1);
      NaluEnv::self().naluOutputP0() << "temperature clipped (-) " << g_numClip[0] << " times; min: " << g_minT << std::endl;
    }

    if ( g_numClip[1] > 0 ) {
      double g_maxT = 0;
      stk::all_reduce_max(comm, &maxT, &g_maxT, 1);
      NaluEnv::self().naluOutputP0() << "temperature clipped (+) " << g_numClip[1] << " times; max: " << g_maxT << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &zN = temperature_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &zNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), zN, zNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_ 
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_Z, "dtdx", "qTmp", "temperature", "PNGradZEQS");
  }
  // fill the map for expected boundary condition names; can be more complex...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "temperature_bc");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "temperature"); // odd in that it can be strong or weak
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "temperature");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "temperature");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
TemperatureEquationSystem::compute_projected_nodal_gradient()
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
