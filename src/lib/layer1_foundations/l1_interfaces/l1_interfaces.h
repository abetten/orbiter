/*
 * l1_interfaces.h
 *
 *  Created on: Mar 18, 2023
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER1_FOUNDATIONS_L1_INTERFACES_L1_INTERFACES_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_L1_INTERFACES_L1_INTERFACES_H_




namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


// #############################################################################
// interface_gap_low.cpp:
// #############################################################################

//! interface to GAP at the foundation level

class interface_gap_low {
public:

	interface_gap_low();
	~interface_gap_low();
	void fining_set_stabilizer_in_collineation_group(
			field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			std::string &fname,
			int verbose_level);
	void collineation_set_stabilizer(
			std::ostream &ost,
			field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			int verbose_level);
	void write_matrix(
			std::ostream &ost,
			field_theory::finite_field *F,
			int *Mtx, int d,
			int verbose_level);
	void write_element_of_finite_field(
			std::ostream &ost,
			field_theory::finite_field *F, int a);

};

// #############################################################################
// interface_magma_low.cpp:
// #############################################################################

//! interface to magma at the foundation level

class interface_magma_low {
public:

	interface_magma_low();
	~interface_magma_low();
	void magma_set_stabilizer_in_collineation_group(
			field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			std::string &fname,
			int verbose_level);
	void export_colored_graph_to_magma(
			graph_theory::colored_graph *Gamma,
			std::string &fname, int verbose_level);

};





}}}




#endif /* SRC_LIB_LAYER1_FOUNDATIONS_L1_INTERFACES_L1_INTERFACES_H_ */
