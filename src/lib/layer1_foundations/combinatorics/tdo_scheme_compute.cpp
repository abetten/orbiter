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
}

tdo_scheme_compute::~tdo_scheme_compute()
{
}

void tdo_scheme_compute::init(
		encoded_combinatorial_object *Enc,
		int max_depth,
		int verbose_level)
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

void tdo_scheme_compute::print_schemes(
		std::ostream &ost)
{
	int verbose_level = 0;
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

}


}}}

