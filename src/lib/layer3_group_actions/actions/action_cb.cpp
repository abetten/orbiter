// action_cb.cpp
//
// Anton Betten
// 1/1/2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace actions {



void action::all_elements(
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::all_elements" << endl;
	}

	if (!f_has_sims) {
		cout << "action::all_elements !f_has_sims" << endl;
		exit(1);
	}

	ring_theory::longinteger_object go;
	long int i, goi;

	group_order(go);
	goi = go.as_int();

	vec = NEW_OBJECT(data_structures_groups::vector_ge);
	vec->init(this, 0 /*verbose_level*/);
	vec->allocate(goi, verbose_level);


	for (i = 0; i < goi; i++) {
		Sims->element_unrank_lint(i, vec->ith(i));
	}

	if (f_v) {
		cout << "action::all_elements done" << endl;
	}
}


void action::all_elements_save_csv(
		std::string &fname, int verbose_level)
{

	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::all_elements_save_csv" << endl;
	}

	if (!f_has_sims) {
		cout << "action::all_elements_save_csv !f_has_sims" << endl;
		exit(1);
	}
	data_structures_groups::vector_ge *vec;
	int i;
	int *data;
	int *Elt;

	all_elements(vec, verbose_level);
	data = NEW_int(make_element_size);


	{
		ofstream ost(fname);

		ost << "Row,Element" << endl;
		for (i = 0; i < vec->len; i++) {
			Elt = vec->ith(i);

			Group_element->element_code_for_make_element(Elt, data);

			stringstream ss;
			Int_vec_print_str_naked(ss, data, make_element_size);
			ost << i << ",\"" << ss.str() << "\"" << endl;
		}
		ost << "END" << endl;
	}
	if (f_v) {
		cout << "action::all_elements_save_csv Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_OBJECT(vec);
	FREE_int(data);

	if (f_v) {
		cout << "action::all_elements_save_csv done" << endl;
	}
}



}}}


