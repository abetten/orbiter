/*
 * wormhole.h
 *
 *  Created on: Sep 23, 2026
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER5_TOP_LEVEL_WORMHOLE_WORMHOLE_H_
#define SRC_LIB_LAYER5_TOP_LEVEL_WORMHOLE_WORMHOLE_H_



namespace orbiter {
namespace layer5_applications {
namespace wormhole {



// #############################################################################
// induced_action_on_specific_orbit.cpp
// #############################################################################

//! induced action on a previously computed orbit


class induced_action_on_specific_orbit {

public:

	std::string label;

	orbits::orbits_create *Orbits;


	//orbits::orbits_on_polynomials *On_polynomials;
	//groups::schreier *Schreier;

	//orbits_on_polynomials
	//layer4_classification::orbits_schreier::orbit_of_equations *Orb;
	layer5_applications::orbits::orbits_on_polynomials *Orb;
	layer4_classification::orbits_schreier::orbit_of_equations *Orbit_of_equations;

	layer3_group_actions::actions::action *A;

	layer3_group_actions::induced_actions::action_on_homogeneous_polynomials *AonHPD;

	layer3_group_actions::induced_actions::wormhole_action *Wormhole_action;

	actions::action *A_induced;

	induced_action_on_specific_orbit();
	~induced_action_on_specific_orbit();
	void init_action_on_polynomial_orbit(
			std::string &orbit_object_label,
			int verbose_level);
	actions::action *compute_induced_action(
			actions::action *A_old,
			int verbose_level);


};


}}}



#endif /* SRC_LIB_LAYER5_TOP_LEVEL_WORMHOLE_WORMHOLE_H_ */
