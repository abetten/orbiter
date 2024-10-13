/*
 * tdo_scheme_compute.cpp
 *
 *  Created on: Dec 18, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


tdo_scheme_compute::tdo_scheme_compute()
{
	Enc = NULL;
	Decomp = NULL;

	f_TDA = false;
	nb_orbits = 0;
	orbit_first = NULL;
	orbit_len = NULL;
	orbit = NULL;

}

tdo_scheme_compute::~tdo_scheme_compute()
{
}

void tdo_scheme_compute::init(
		canonical_form_classification::encoded_combinatorial_object *Enc,
		int max_depth,
		int verbose_level)
// used by combinatorial_object_with_properties::compute_TDO
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_scheme_compute::init" << endl;
	}

	tdo_scheme_compute::Enc = Enc;

	Decomp = NEW_OBJECT(decomposition);

	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"before Decomp->init_incidence_matrix" << endl;
	}

	Decomp->init_incidence_matrix(
			Enc->nb_rows, Enc->nb_cols, Enc->get_Incma(),
			verbose_level);

	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"after Decomp->init_incidence_matrix" << endl;
	}


	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"before Decomp->setup_default_partition" << endl;
	}
	Decomp->setup_default_partition(verbose_level);
	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"after Decomp->setup_default_partition" << endl;
	}


	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"before Decomp->compute_TDO_old" << endl;
	}

	Decomp->compute_TDO_old(max_depth, verbose_level);

	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"after Decomp->compute_TDO_old" << endl;
	}

	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"before Decomp->get_row_scheme" << endl;
	}
	Decomp->get_row_scheme(verbose_level);
	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"after Decomp->get_row_scheme" << endl;
	}

	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"before Decomp->get_col_scheme" << endl;
	}
	Decomp->get_col_scheme(verbose_level);
	if (f_v) {
		cout << "tdo_scheme_compute::init "
				"after Decomp->get_col_scheme" << endl;
	}

	if (f_v) {
		cout << "tdo_scheme_compute::init done" << endl;
	}
}

void tdo_scheme_compute::init_TDA(
		canonical_form_classification::encoded_combinatorial_object *Enc,
		int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA" << endl;
	}


	tdo_scheme_compute::Enc = Enc;

	f_TDA = true;
	tdo_scheme_compute::nb_orbits = nb_orbits;
	tdo_scheme_compute::orbit_first = orbit_first;
	tdo_scheme_compute::orbit_len = orbit_len;
	tdo_scheme_compute::orbit = orbit;


	Decomp = NEW_OBJECT(decomposition);

	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"before Decomp->init_incidence_matrix" << endl;
	}

	Decomp->init_incidence_matrix(
			Enc->nb_rows, Enc->nb_cols, Enc->get_Incma(),
			verbose_level);

	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"after Decomp->init_incidence_matrix" << endl;
	}


	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"before Decomp->setup_default_partition" << endl;
	}
	Decomp->setup_default_partition(verbose_level);
	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"after Decomp->setup_default_partition" << endl;
	}


	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"before Decomp->Stack->split_by_orbit_partition" << endl;
	}

	Decomp->Stack->split_by_orbit_partition(
			nb_orbits,
		orbit_first, orbit_len, orbit,
		0 /*offset*/,
		verbose_level);
	//Decomp->compute_TDO_old(max_depth, verbose_level);

	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"after Decomp->Stack->split_by_orbit_partition" << endl;
	}


	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"before Decomp->get_row_scheme" << endl;
	}
	Decomp->get_row_scheme(verbose_level);
	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"after Decomp->get_row_scheme" << endl;
	}

	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"before Decomp->get_col_scheme" << endl;
	}
	Decomp->get_col_scheme(verbose_level);
	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA "
				"after Decomp->get_col_scheme" << endl;
	}

	if (f_v) {
		cout << "tdo_scheme_compute::init_TDA done" << endl;
	}
}

void tdo_scheme_compute::print_schemes(
		std::ostream &ost,
		canonical_form_classification::objects_report_options
			*Report_options,
		int verbose_level)
// called from combinatorial_object_with_properties::print_TDO
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_scheme_compute::print_schemes" << endl;
	}

	ost << "\\subsection*{tdo\\_scheme\\_compute::print\\_schemes}" << endl;

	if (f_v) {
		cout << "tdo_scheme_compute::print_schemes "
				"before Decomp->print_schemes" << endl;
	}
	Decomp->print_schemes(ost, Report_options, verbose_level);
	if (f_v) {
		cout << "tdo_scheme_compute::print_schemes "
				"after Decomp->print_schemes" << endl;
	}

#if 0
	int f_enter_math = false;
	int f_print_subscripts = true;

	ost << "$$" << endl;
	Decomp->Scheme->print_row_decomposition_tex(
		ost,
		f_enter_math, f_print_subscripts,
		verbose_level);
	ost << "$$" << endl;
	ost << "$$" << endl;
	Decomp->Scheme->print_column_decomposition_tex(
		ost,
		f_enter_math, f_print_subscripts,
		verbose_level);
	ost << "$$" << endl;


	Decomp->Scheme->RC->print_classes_of_decomposition_tex(ost);
#endif

	if (f_v) {
		cout << "tdo_scheme_compute::print_schemes done" << endl;
	}

}


}}}

