/*
 * quartic_curves.cpp
 *
 *  Created on: May 25, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_QUARTIC_CURVES_QUARTIC_CURVES_CPP_
#define SRC_LIB_TOP_LEVEL_QUARTIC_CURVES_QUARTIC_CURVES_CPP_




namespace orbiter {
namespace top_level {



// #############################################################################
// quartic_curve_activity_description.cpp
// #############################################################################

//! description of an activity associated with a quartic curve


class quartic_curve_activity_description {
public:

	int f_report;

	int f_report_with_group;

	int f_export_points;


	quartic_curve_activity_description();
	~quartic_curve_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

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

	int f_q;
	int q;

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

	int f_override_group;
	std::string override_group_order;
	int override_group_nb_gens;
	std::string override_group_gens;

	std::vector<std::string> transform_coeffs;
	std::vector<int> f_inverse_transform;



	quartic_curve_create_description();
	~quartic_curve_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	int get_q();
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
	finite_field *F;

	int f_semilinear;


	projective_space_with_action *PA;

	quartic_curve_domain_with_action *QCDA;

	quartic_curve_object *QO;

	quartic_curve_object_with_action *QOA;

	int f_has_group;
	strong_generators *Sg;
	int f_has_nice_gens;
	vector_ge *nice_gens;




	quartic_curve_create();
	~quartic_curve_create();
	void null();
	void freeself();
	void init_with_data(
			quartic_curve_create_description *Descr,
			projective_space_with_action *PA,
			quartic_curve_domain_with_action *QCDA,
			int verbose_level);
	void init(
			quartic_curve_create_description *Descr,
			projective_space_with_action *PA,
			quartic_curve_domain_with_action *QCDA,
			int verbose_level);
	void create_quartic_curve_from_description(quartic_curve_domain_with_action *DomA, int verbose_level);
	void override_group(std::string &group_order_text,
			int nb_gens, std::string &gens_text, int verbose_level);
	void create_quartic_curve_by_coefficients(std::string &coefficients_text,
			int verbose_level);
	void create_quartic_curve_by_coefficient_vector(int *eqn15,
			int verbose_level);
	void create_quartic_curve_from_catalogue(quartic_curve_domain_with_action *DomA,
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
	void apply_transformations(
		std::vector<std::string> &transform_coeffs,
		std::vector<int> &f_inverse_transform,
		int verbose_level);
	void compute_group(projective_space_with_action *PA,
			int verbose_level);

};


// #############################################################################
// quartic_curve_domain_with_action.cpp
// #############################################################################

//! a domain for quartic curves in projective space with group action



class quartic_curve_domain_with_action {

public:


	projective_space_with_action *PA;

	int f_semilinear;

	quartic_curve_domain *Dom; // do not free

	action *A; // linear group PGGL(3,q)


	action *A_on_lines; // linear group PGGL(3,q) acting on lines

	int *Elt1;

	action_on_homogeneous_polynomials *AonHPD_4_3;


	quartic_curve_domain_with_action();
	~quartic_curve_domain_with_action();
	void init(quartic_curve_domain *Dom,
			projective_space_with_action *PA,
			int verbose_level);

};


// #############################################################################
// quartic_curve_from_surface.cpp
// #############################################################################


//! construction of a quartic curve from a cubic surface by means of projecting the intersection with the tangent cone at a point


class quartic_curve_from_surface {

public:

	surface_object_with_action *SOA;

	int pt_orbit;
	int equation_nice[20];
	int *transporter;
	int v[4];
	int pt_A, pt_B;

	long int *Lines_nice;
	int nb_lines;

	long int *Bitangents;
	int nb_bitangents; // = nb_lines + 1


	int *f1;
	int *f2;
	int *f3;

	long int *Pts_on_surface;
	int nb_pts_on_surface;

	int *curve;
	int *poly1;
	int *poly2;
	int two, four, mfour;

	int *tangent_quadric;
	long int *Pts_on_tangent_quadric;
	int nb_pts_on_tangent_quadric;

	//int *line_type;
	//int *type_collected;

	int *Class_pts;
	int nb_class_pts;
	long int *Pts_intersection;
	int nb_pts_intersection;

	long int *Pts_on_curve;
	int sz_curve;

	strong_generators *gens_copy;
	set_and_stabilizer *moved_surface;
	//strong_generators *stab_gens_moved_surface;
	strong_generators *stab_gens_P0;

	strong_generators *Stab_gens_quartic;



	quartic_curve_from_surface();
	~quartic_curve_from_surface();
	void init(surface_object_with_action *SOA, int verbose_level);
	void quartic(std::string &surface_prefix, int pt_orbit, int f_TDO, int verbose_level);
	void compute_quartic(int pt_orbit,
		int *equation, long int *Lines, int nb_lines,
		int verbose_level);
	void compute_stabilizer(int verbose_level);
	void cheat_sheet_quartic_curve(
			std::string &surface_prefix,
			std::ostream &ost,
			std::ostream &ost_curves,
			int f_TDO,
			int verbose_level);

};




// #############################################################################
// quartic_curve_object_with_action.cpp
// #############################################################################


//! an instance of a quartic curve together with its stabilizer


class quartic_curve_object_with_action {

public:

	finite_field *F; // do not free

	quartic_curve_domain_with_action *DomA;

	quartic_curve_object *QO; // do not free
	strong_generators *Aut_gens;
		// generators for the automorphism group

	int f_has_nice_gens;
	vector_ge *nice_gens;

	strong_generators *projectivity_group_gens;
	sylow_structure *Syl;

	action *A_on_points;

	schreier *Orbits_on_points;

	quartic_curve_object_with_action();
	~quartic_curve_object_with_action();
	void init(quartic_curve_domain_with_action *DomA,
			quartic_curve_object *QO,
			strong_generators *Aut_gens,
			int verbose_level);

};





}}




#endif /* SRC_LIB_TOP_LEVEL_QUARTIC_CURVES_QUARTIC_CURVES_CPP_ */
