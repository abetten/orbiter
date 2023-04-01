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
// expression_parser_sajeeb.cpp:
// #############################################################################

//! interface to Sajeeb's expression parser

class expression_parser_sajeeb {
public:

	expression_parser::formula *Formula;

	void *private_data;

	expression_parser_sajeeb();
	~expression_parser_sajeeb();
	void init_formula(
			expression_parser::formula *Formula,
			int verbose_level);
	void get_subtrees(
			ring_theory::homogeneous_polynomial_domain *Poly,
			int verbose_level);
	void evaluate(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::map<std::string, std::string> &symbol_table, int *Values,
			int verbose_level);


};



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
