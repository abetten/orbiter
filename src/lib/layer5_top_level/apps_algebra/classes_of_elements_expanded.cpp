/*
 * classes_of_elements_expanded.cpp
 *
 *  Created on: Feb 25, 2025
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


classes_of_elements_expanded::classes_of_elements_expanded()
{
	Record_birth();

	Classes = NULL;
	sims_G = NULL;
	Any_group = NULL;
	expand_by_go = 0;
	//std::string label;
	//std::string label_latex;

	Idx = NULL;
	nb_idx = 0;

	A_conj = NULL;

	Orbit_of_elements = NULL;

}



classes_of_elements_expanded::~classes_of_elements_expanded()
{
	Record_death();

	if (Idx) {
		FREE_int(Idx);
	}
	if (A_conj) {
		FREE_OBJECT(A_conj);
	}
	if (Orbit_of_elements) {
		FREE_OBJECT(Orbit_of_elements);
	}
}



void classes_of_elements_expanded::init(
		interfaces::conjugacy_classes_and_normalizers *Classes,
		groups::sims *sims_G,
		groups::any_group *Any_group,
		int expand_by_go,
		std::string &label,
		std::string &label_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classes_of_elements_expanded::init" << endl;
	}

	classes_of_elements_expanded::Classes = Classes;
	classes_of_elements_expanded::sims_G = sims_G;
	classes_of_elements_expanded::Any_group = Any_group;
	classes_of_elements_expanded::expand_by_go = expand_by_go;
	classes_of_elements_expanded::label = label;
	classes_of_elements_expanded::label_latex = label_latex;


	int i;

	Idx = NEW_int(Classes->nb_classes);
	other::orbiter_kernel_system::file_io Fio;

	nb_idx = 0;
	for (i = 0; i < Classes->nb_classes; i++) {
		if (Classes->class_size[i] <= expand_by_go) {
			Idx[nb_idx++] = i;
		}
	}
	if (f_v) {
		cout << "classes_of_elements_expanded::init "
				"We found " << nb_idx
				<< "conjugacy classes of groups of order "
						"at most " << expand_by_go << endl;
		cout << "They are: ";
		Int_vec_print(cout, Idx, nb_idx);
		cout << endl;
	}

	//Sims_G = Any_group->Subgroup_gens->create_sims(verbose_level);



	int h, idx;



	if (f_v) {
		cout << "classes_of_elements_expanded::init "
				"before Any_group->A->create_induced_action_by_conjugation" << endl;
	}
	A_conj = Any_group->A->Induced_action->create_induced_action_by_conjugation(
			sims_G /*Base_group*/, false /* f_ownership */,
			false /* f_basis */, NULL /* old_G */,
			verbose_level - 2);
	if (f_v) {
		cout << "classes_of_elements_expanded::init "
				"after Any_group->A->create_induced_action_by_conjugation" << endl;
	}


	Orbit_of_elements = (orbit_of_elements **) NEW_pvoid(nb_idx);

	for (h = 0; h < nb_idx; h++) {

		idx = Idx[h];



		Orbit_of_elements[h] = NEW_OBJECT(orbit_of_elements);

		if (f_v) {
			cout << "classes_of_elements_expanded::init "
					"before Orbit_of_elements->init" << endl;
		}

		Orbit_of_elements[h]->init(
				Any_group,
				sims_G,
				A_conj,
				Classes,
				idx,
				verbose_level);

		if (f_v) {
			cout << "classes_of_elements_expanded::init "
					"after Orbit_of_elements->init" << endl;
		}

#if 0
		long int *Table;
		int orbit_length, set_size;

		Orbit_of_elements[h]->Orbits_P->get_table_of_orbits(
				Table,
				orbit_length, set_size, verbose_level);

		if (f_v) {
			cout << "classes_of_elements_expanded::init "
					"table of subgroups:" << endl;
			Lint_matrix_print(Table, orbit_length, set_size);
			cout << "classes_of_elements_expanded::init "
					"h=" << h << " idx=" << idx << " number of subgroups = " << orbit_length << endl;
		}
#endif



	}



	if (f_v) {
		cout << "classes_of_elements_expanded::init done" << endl;
	}
}


void classes_of_elements_expanded::report(
		std::string &label,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classes_of_elements_expanded::report" << endl;
	}

	{
		string fname;
		string title;
		string author, extra_praeamble;

		fname = label + "_report_classes_of_elements.tex";
		title = "The group $" + label_tex + "$";
		author = "";


		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "classes_of_elements_expanded::report "
						"before report2" << endl;
			}

			report2(ost, verbose_level - 1);

			if (f_v) {
				cout << "classes_of_elements_expanded::report "
						"after report2" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "classes_of_elements_expanded::report done" << endl;
	}
}


void classes_of_elements_expanded::report2(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classes_of_elements_expanded::report2" << endl;
	}


	Classes->report_summary_table(ost, verbose_level);

	int h, idx;

	h = 0;
	for (idx = 0; idx < Classes->nb_classes; idx++) {

		Classes->Conjugacy_class[idx]->report_single_class(ost, verbose_level - 1);

		if (h < nb_idx && Idx[h] == idx) {

			std::string class_label;

			class_label = Classes->Conjugacy_class[h]->class_label();

			ost << "Class " << h << " / " << nb_idx << " is class " << idx << " = " << class_label << " of size " << Orbit_of_elements[h]->orbit_length << "\\\\" << endl;
			cout << "classes_of_subgroups_expanded::init "
					"Class size = " << Orbit_of_elements[h]->orbit_length << endl;

			int i;

			for (i = 0; i < Orbit_of_elements[h]->orbit_length; i++) {
				ost << Orbit_of_elements[h]->Table_of_elements[i];
				if (i < Orbit_of_elements[h]->orbit_length - 1) {
					ost << ", ";
				}
			}
			ost << "\\\\" << endl;
			ost << endl;
			ost << "\\bigskip" << endl;
			ost << endl;



			h++;
		}

	}

#if 0

	int h, idx;

	for (h = 0; h < nb_idx; h++) {



		idx = Idx[h];

		Classes->Conjugacy_class[idx]->report_single_class(ost, verbose_level - 1);

		std::string class_label;

		class_label = Classes->Conjugacy_class[idx]->class_label();

		ost << "Class " << h << " / " << nb_idx << " is class " << idx << " = " << class_label << " of size " << Orbit_of_elements[h]->orbit_length << "\\\\" << endl;
		cout << "classes_of_subgroups_expanded::init "
				"Class size = " << Orbit_of_elements[h]->orbit_length << endl;

		int i, j;

		for (i = 0; i < Orbit_of_elements[h]->orbit_length; i++) {
			ost << Orbit_of_elements[h]->Table_of_elements[i];
			if (i < Orbit_of_elements[h]->orbit_length - 1) {
				ost << ", ";
			}
		}
		ost << "\\\\" << endl;
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

	}
#endif

	if (f_v) {
		cout << "classes_of_elements_expanded::report2 done" << endl;
	}
}




}}}


