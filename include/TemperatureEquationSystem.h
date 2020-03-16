/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TemperatureEquationSystem_h
#define TemperatureEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class AlgorithmDriver;
class Realm;
class AssembleNodalGradAlgorithmDriver;
class LinearSystem;
class EquationSystems;
class ProjectedNodalGradientEquationSystem;

class TemperatureEquationSystem : public EquationSystem {

public:

  TemperatureEquationSystem(
    EquationSystems& equationSystems,
    const double minT,
    const double maxT,
    const bool outputClippingDiag);
  virtual ~TemperatureEquationSystem();

  void populate_derived_quantities();
  
  virtual void register_nodal_fields_alt(
    stk::mesh::Part *part);

  void register_interior_algorithm_alt(
    stk::mesh::Part *part, 
    const std::vector<std::string> &terms);
  
  void register_inflow_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const InflowBoundaryConditionData &inflowBCData);
  
  void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const OpenBoundaryConditionData &openBCData);

  void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const WallBoundaryConditionData &wallBCData);

  virtual void register_symmetry_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const SymmetryBoundaryConditionData &symmetryBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_overset_bc();

  virtual void register_initial_condition_fcn(
      stk::mesh::Part *part,
      const std::map<std::string, std::string> &theNames,
      const std::map<std::string, std::vector<double> > &theParams);

  void initialize();
  void reinitialize_linear_system();
  
  void predict_state();
  
  void solve_and_update();
  void update_and_clip();

  void manage_projected_nodal_gradient(
    EquationSystems& eqSystems);
  void compute_projected_nodal_gradient();

  const double minimumT_;
  const double maximumT_;
  const bool managePNG_;
  const bool outputClippingDiag_;

  ScalarFieldType *temperature_;
  VectorFieldType *dtdx_;
  ScalarFieldType *tTmp_;
  ScalarFieldType *density_;
  ScalarFieldType *specificHeat_;
  ScalarFieldType *thermalCond_;
  
  AssembleNodalGradAlgorithmDriver *assembleNodalGradAlgDriver_;
  
  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;
  
  bool isInit_;
};


} // namespace nalu
} // namespace Sierra

#endif
