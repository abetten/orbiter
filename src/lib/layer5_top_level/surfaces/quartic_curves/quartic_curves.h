/*
 * quartic_curves.cpp
 *
 *  Created on: May 25, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_QUARTIC_CURVES_QUARTIC_CURVES_CPP_
#define SRC_LIB_TOP_LEVEL_QUARTIC_CURVES_QUARTIC_CURVES_CPP_




namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {


// #############################################################################
// quartic_curve_activity_description.cpp
// #############################################################################

//! description of an activity associated with a quartic curve


class quartic_curve_activity_description {
public:

	int f_report;

	int f_export_something;
	std::string export_something_what;


	//int f_export_points;

	int f_create_surface;

	int f_extract_orbit_on_bitangents_by_length;
	int extract_orbit_on_bitangents_by_length_length;

	int f_extract_specific_orbit_on_bitangents_by_length;
	int extract_specific_orbit_on_bitangents_by_length_length;
	int extract_specific_orbit_on_bitangents_by_length_index;

	int f_extract_specific_orbit_on_kovalevski_points_by_length;
	int extract_specific_orbit_on_kovalevski_points_by_length_length;
	int extract_specific_orbit_on_kovalevski_points_by_length_index;


	quartic_curve_activity_description();
	~quartic_curve_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// quartic_curve_activity.cpp
// #############################################################################

//! an activity associated with a quartic curve


class quartic_curve_activity {
public:

	quartic_curve_activity_description *Descr;
	quartic_curve_create *QC;


	quartic_curve_activity();
	~quartic_curve_activity();
	void init(quartic_curve_activity_description *Quartic_curve_activity_description,
			quartic_curve_create *QC, int verbose_level);
	void perform_activity(int verbose_level);
	void do_report(
			quartic_curve_create *QC,
			int verbose_level);

};




// #############################################################################
// quartic_curve_create_description.cpp
// #############################################################################



//! to describe a quartic curve from the command line


class quartic_curve_create_description {

public:

	int f_space;
	std::string space_label;

	int f_space_pointer;
	projective_geometry::projective_space_with_action *space_pointer;

	int f_label_txt;
	std::string label_txt;

	int f_label_tex;
	std::string label_tex;

	int f_label_for_summary;
	std::string label_for_summary;

	int f_catalogue;
	int iso;

	int f_by_coefficients;
	std::string coefficients_text;


	int f_by_equation;
	std::string equation_name_of_formula;
	std::string equation_name_of_formula_tex;
	std::string equation_managed_variables;
	std::string equation_text;
	std::string equation_parameters;
	std::string equation_parameters_tex;

	int f_from_cubic_surface;
	std::string from_cubic_surface_label;
	int from_cubic_surface_point_orbit_idx;


	int f_override_group;
	std::string override_group_order;
	int override_group_nb_gens;
	std::string override_group_gens;

	std::vector<std::string> transform_coeffs;
	std::vector<int> f_inverse_transform;



	quartic_curve_create_description();
	~quartic_curve_create_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();
	//int get_q();
};



// #############################################################################
// quartic_curve_create.cpp
// #############################################################################


//! to create a quartic curve from a description using class quartic_curve_create_description


class quartic_curve_create {

public:
	quartic_curve_create_description *Descr;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int f_ownership;

	int q;
	field_theory::finite_field *F;

	int f_semilinear;


	projective_geometry::projective_space_with_action *PA;

	quartic_curve_domain_with_action *QCDA;

	algebraic_geometry::quartic_curve_object *QO;

	quartic_curve_object_with_action *QOA;

	int f_has_group;
	groups::strong_generators *Sg;
	int f_has_nice_gens;
	data_structures_groups::vector_ge *nice_gens;

	int f_has_quartic_curve_from_surface;
	quartic_curves::quartic_curve_from_surface *QC_from_surface;



	quartic_curve_create();
	~quartic_curve_create();
	void create_quartic_curve(
			quartic_curve_create_description *Quartic_curve_descr,
			int verbose_level);
	void init_with_data(
			quartic_curve_create_description *Descr,
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
	void init(
			quartic_curve_create_description *Descr,
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
	void create_quartic_curve_from_description(
			quartic_curve_domain_with_action *DomA, int verbose_level);
	void override_group(std::string &group_order_text,
			int nb_gens, std::string &gens_text, int verbose_level);
	void create_quartic_curve_by_coefficients(std::string &coefficients_text,
			int verbose_level);
	void create_quartic_curve_by_coefficient_vector(int *eqn15,
			int verbose_level);
	void create_quartic_curve_from_catalogue(
			quartic_curve_domain_with_action *DomA,
			int iso,
			int verbose_level);
	void create_quartic_curve_by_equation(
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			std::string &managed_variables,
			std::string &equation_text,
			std::string &equation_parameters,
			std::string &equation_parameters_tex,
			int verbose_level);
	void create_quartic_curve_from_cubic_surface(
			std::string &cubic_surface_label,
			int pt_orbit_idx,
			//int f_TDO,
			int verbose_level);
	void apply_transformations(
		std::vector<std::string> &transform_coeffs,
		std::vector<int> &f_inverse_transform,
		int verbose_level);
	void apply_single_transformation(int f_inverse,
			int *transformation_coeffs,
			int sz, int verbose_level);
	void compute_group(
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
	// ToDo
	void report(std::ostream &ost, int verbose_level);
	void print_general(std::ostream &ost, int verbose_level);
	void export_something(std::string &what, int verbose_level);

};


// #############################################################################
// quartic_curve_domain_with_action.cpp
// #############################################################################

//! a domain for quartic curves in projective space with group action



class quartic_curve_domain_with_action {

public:


	projective_geometry::projective_space_with_action *PA;

	int f_semilinear;

	algebraic_geometry::quartic_curve_domain *Dom; // do not free

	actions::action *A; // linear group PGGL(3,q)


	actions::action *A_on_lines; // linear group PGGL(3,q) acting on lines

	int *Elt1;

	induced_actions::action_on_homogeneous_polynomials *AonHPD_4_3;


	quartic_curve_domain_with_action();
	~quartic_curve_domain_with_action();
	void init(algebraic_geometry::quartic_curve_domain *Dom,
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);

};


// #############################################################################
// quartic_curve_from_surface.cpp
// #############################################################################


//! construction of a quartic curve from a cubic surface by means of projecting the intersection with the tangent cone at a point


class quartic_curve_from_surface {

public:

	std::string label;
	std::string label_tex;


	int f_has_SC;
	cubic_surfaces_in_general::surface_create *SC;

	cubic_surfaces_in_general::surface_object_with_action *SOA;

	int pt_orbit;
	int equation_nice[20]; // equation after transformation
	int *transporter; // the transformation that maps the point off the lines to (1,0,0,0)

	int v[4]; // = (1,0,0,0)
	int pt_A; // = SOA->SO->SOP->Pts_not_on_lines[i];
	int pt_B; // = SOA->Surf->rank_point(v);

	long int *Lines_nice; // surface lines after transformation
	int nb_lines;

	long int *Bitangents;
	int nb_bitangents; // = nb_lines + 1


	// computed by split_nice_equation starting from the equation of the cubic surface:
	int *f1; // terms involving X0^2, with X0^2 removed (linear)
	int *f2; // terms involving X0, with X0 removed (quadratic)
	int *f3; // terms free of X0 (cubic)

	long int *Pts_on_surface; // points on the transformed cubic surface
	int nb_pts_on_surface;

	// the equation of the quartic curve:
	int *curve; // poly1 + poly2 = f2^2 - 4 * f1 * f3
	int *poly1; // f2 * f2
	int *poly2; // -4 * f1 * f3
	int two, four, mfour; // 2, 4, -4 in F

	int *tangent_quadric; // = 2 * x_0 * f_1 + f_2
	long int *Pts_on_tangent_quadric; // = SOA->Surf->Poly2_4->enumerate_points(tangent_quadric)
	int nb_pts_on_tangent_quadric;

	//int *line_type;
	//int *type_collected;

	//int *Class_pts;
	//int nb_class_pts;
	//long int *Pts_intersection;
	//int nb_pts_intersection;

	long int *Pts_on_curve; // = SOA->Surf->Poly4_x123->enumerate_points(curve)
	int sz_curve;

#if 0
	strong_generators *gens_copy;
	set_and_stabilizer *moved_surface;
	//strong_generators *stab_gens_moved_surface;
	strong_generators *stab_gens_P0;
#endif

	groups::strong_generators *Stab_gens_quartic;



	quartic_curve_from_surface();
	~quartic_curve_from_surface();
	void init(cubic_surfaces_in_general::surface_object_with_action *SOA, int verbose_level);
	void init_surface_create(cubic_surfaces_in_general::surface_create *SC,
			int verbose_level);
	void init_labels(std::string &label, std::string &label_tex,
			int verbose_level);
	void quartic(int pt_orbit, int verbose_level);
	void compute_quartic(int pt_orbit,
		int *equation, long int *Lines, int nb_lines,
		int verbose_level);
	void compute_stabilizer(int verbose_level);
	void cheat_sheet_quartic_curve(
			std::ostream &ost,
			int f_TDO,
			int verbose_level);

};




// #############################################################################
// quartic_curve_object_with_action.cpp
// #############################################################################


//! an instance of a quartic curve together with its stabilizer


class quartic_curve_object_with_action {

public:

	field_theory::finite_field *F; // do not free

	quartic_curve_domain_with_action *DomA;

	algebraic_geometry::quartic_curve_object *QO; // do not free
	groups::strong_generators *Aut_gens;
		// generators for the automorphism group

	int f_has_nice_gens;
	data_structures_groups::vector_ge *nice_gens;

	groups::strong_generators *projectivity_group_gens;
	groups::sylow_structure *Syl;

	actions::action *A_on_points;

	groups::schreier *Orbits_on_points;

	quartic_curve_object_with_action();
	~quartic_curve_object_with_action();
	void init(quartic_curve_domain_with_action *DomA,
			algebraic_geometry::quartic_curve_object *QO,
			groups::strong_generators *Aut_gens,
			int verbose_level);
	void export_something(std::string &what,
			std::string &fname_base, int verbose_level);

};





}}}}



#endif /* SRC_LIB_TOP_LEVEL_QUARTIC_CURVES_QUARTIC_CURVES_CPP_ */
