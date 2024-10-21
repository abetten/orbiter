/*
 * knowledge_base.h
 *
 *  Created on: May 20, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_KNOWLEDGE_BASE_KNOWLEDGE_BASE_H_
#define SRC_LIB_FOUNDATIONS_KNOWLEDGE_BASE_KNOWLEDGE_BASE_H_


namespace orbiter {
namespace layer1_foundations {
namespace knowledge_base {




// #############################################################################
// knowledge_base.cpp:
// #############################################################################

//! provides access to pre-computed combinatorial data in encoded form


class knowledge_base {
public:
	knowledge_base();
	~knowledge_base();


	// the index i is zero-based:


	int quartic_curves_nb_reps(
			int q);
	int *quartic_curves_representative(
			int q, int i);
	long int *quartic_curves_bitangents(
			int q, int i);
	void quartic_curves_stab_gens(
			int q, int i,
			int *&data, int &nb_gens,
			int &data_size, std::string &stab_order_str);


	int cubic_surface_nb_reps(
			int q);
	int *cubic_surface_representative(
			int q, int i);
	void cubic_surface_stab_gens(
			int q, int i, int *&data, int &nb_gens,
		int &data_size, std::string &stab_order_str);
	int cubic_surface_nb_Eckardt_points(
			int q, int i);
	long int *cubic_surface_Lines(
			int q, int i);

	int hyperoval_nb_reps(
			int q);
	int *hyperoval_representative(
			int q, int i);
	void hyperoval_gens(
			int q, int i, int *&data, int &nb_gens,
		int &data_size, std::string &stab_order_str);


	int DH_nb_reps(
			int k, int n);
	long int *DH_representative(
			int k, int n, int i);
	void DH_stab_gens(
			int k, int n, int i, int *&data, int &nb_gens,
		int &data_size, std::string &stab_order_str);

	int Spread_nb_reps(
			int q, int k);
	long int *Spread_representative(
			int q, int k, int i, int &sz);
	void Spread_stab_gens(
			int q, int k, int i, int *&data, int &nb_gens,
		int &data_size, std::string &stab_order_str);

	int BLT_nb_reps(
			int q);
	long int *BLT_representative(
			int q, int no);
	void BLT_stab_gens(
			int q, int no, int *&data, int &nb_gens,
		int &data_size, std::string &stab_order_str);

	void override_polynomial_subfield(
			std::string &poly, int q);
	void override_polynomial_extension_field(
			std::string &poly, int q);

	void get_projective_plane_list_of_lines(
			int *&list_of_lines,
			int &order, int &nb_lines, int &line_size,
			const char *label, int verbose_level);

	int tensor_orbits_nb_reps(
			int n);
	long int *tensor_orbits_rep(
			int n, int idx);

	void retrieve_BLT_set_from_database_embedded(
			orthogonal_geometry::quadratic_form *Quadratic_form,
			int BLT_k,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	void retrieve_BLT_set_from_database(
			orthogonal_geometry::quadratic_form *Quadratic_form,
			int f_embedded,
			int BLT_k,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);

	// finitefield_tables.cpp:
	void get_primitive_polynomial(
			std::string &poly, int p, int e,
			int verbose_level);

};


}}}


#endif /* SRC_LIB_FOUNDATIONS_KNOWLEDGE_BASE_KNOWLEDGE_BASE_H_ */
