/*
 * group_theory_with_magma.cpp
 *
 *  Created on: Sep 5, 2026
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {



group_theory_with_magma::group_theory_with_magma()
{
	Record_birth();

}


group_theory_with_magma::~group_theory_with_magma()
{
	Record_death();
}



void group_theory_with_magma::all_elements_by_class(
		groups::sims *Sims,
		groups::any_group *Any_group,
		layer5_applications::apps_algebra::classes_of_elements_expanded *Classes_of_elements_expanded,
		int class_order,
		int class_id,
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_with_magma::all_elements_by_class, "
				"class_order = " << class_order << " class_id = " << class_id << endl;
	}


	//algebra::ring_theory::longinteger_object go;
	//long int goi;

	//Sims->group_order(go);
	//goi = go.as_int();


	//classes_of_elements_expanded *Classes_of_elements_expanded;
	//data_structures_groups::vector_ge *Reps;

#if 0
	if (f_v) {
		cout << "group_theory_with_magma::all_elements_by_class "
				"before get_classses_expanded" << endl;
	}
	get_classses_expanded(
			Sims,
			Any_group,
			goi,
			Classes_of_elements_expanded,
			Reps,
			verbose_level);
	if (f_v) {
		cout << "group_theory_with_magma::all_elements_by_class "
				"after get_classses_expanded" << endl;
	}
#endif

#if 0
	interfaces::conjugacy_classes_and_normalizers *Classes;
	groups::sims *sims_G;
	groups::any_group *Any_group;
	int expand_by_go;
	std::string label;
	std::string label_latex;

	int *Idx;
	int nb_idx;

	actions::action *A_conj;

	orbit_of_elements **Orbit_of_elements; // [nb_idx]
#endif

	int nb_classes;
	int h, cnt;


	nb_classes = Classes_of_elements_expanded->Classes->nb_classes;

	cout << "nb_classes = " << nb_classes << endl;
	cnt = 0;
	for (h = 0; h < nb_classes; h++) {
		if (Classes_of_elements_expanded->Classes->class_order_of_element[h] == class_order) {
			cout << "class " << h << " consists of elements of order " << class_order << endl;
			if (cnt == class_id) {
				cout << "found class, h=" << h << endl;
				break;
			}
			cnt++;
		}
	}
	if (h == nb_classes) {
		cout << "did not find class class_order =" << class_order << " class_id=" << class_id << endl;
		exit(1);
	}
	cout << "found class, h=" << h << endl;

	int idx;

	for (idx = 0; idx < Classes_of_elements_expanded->nb_idx; idx++) {
		if (Classes_of_elements_expanded->Idx[idx] == h) {
			cout << "found class, idx=" << idx << endl;
			break;
		}
	}
	if (idx == Classes_of_elements_expanded->nb_idx) {
		cout << "did not find class" << endl;
		exit(1);
	}



	//int go_P;

	//go_P = Classes_of_elements_expanded->Orbit_of_elements[h]->go_P;

#if 0
	int idx;


	long int go_P;
	int *Element;
	long int Element_rk;
	long int *Elements_P;
	orbits_schreier::orbit_of_sets *Orbits_P;

	int orbit_length;
	long int *Table_of_elements; // sorted

#endif

	int class_size;
	long int *Table;

	class_size = Classes_of_elements_expanded->Orbit_of_elements[idx]->orbit_length;
	Table = Classes_of_elements_expanded->Orbit_of_elements[idx]->Table_of_elements;


	cout << "class_size = " << class_size << endl;
	cout << "Table_of_elements = ";
	Lint_vec_print_fully(cout, Table, class_size);
	cout << endl;



	vec = NEW_OBJECT(data_structures_groups::vector_ge);
	vec->init(Sims->A, 0 /*verbose_level*/);
	vec->allocate(class_size, verbose_level);

	int i;

	for (i = 0; i < class_size; i++) {
		Sims->element_unrank_lint(Table[i], vec->ith(i));
	}


	if (f_v) {
		cout << "group_theory_with_magma::all_elements_by_class done" << endl;
	}
}


void group_theory_with_magma::get_classses_expanded(
		groups::sims *Sims,
		groups::any_group *Any_group,
		int expand_by_go,
		classes_of_elements_expanded *&Classes_of_elements_expanded,
		data_structures_groups::vector_ge *&Reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded" << endl;
	}
	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded A = " << Any_group->A->label << endl;
		cout << "group_theory_with_magma::get_classses_expanded A_base = " << Any_group->A_base->label << endl;
	}

	interfaces::conjugacy_classes_and_normalizers *class_data;


	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded "
				"before AG->get_conjugacy_classes_of_elements" << endl;
	}
	Any_group->get_conjugacy_classes_of_elements(
			Sims, class_data, verbose_level - 2);
	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded "
				"after AG->get_conjugacy_classes_of_elements" << endl;
	}


	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded "
				"before class_data->get_representatives" << endl;
	}
	class_data->get_representatives(
			Sims,
			Reps,
			verbose_level - 2);
	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded "
				"after class_data->get_representatives" << endl;
	}



	//classes_of_elements_expanded *Classes_of_elements_expanded;

	Classes_of_elements_expanded = NEW_OBJECT(classes_of_elements_expanded);


	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded "
				"before Classes_of_elements_expanded->init" << endl;
	}
	Classes_of_elements_expanded->init(
			class_data,
			Sims,
			Any_group,
			expand_by_go,
			Any_group->label,
			Any_group->label_tex,
			verbose_level);
	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded "
				"after Classes_of_elements_expanded->init" << endl;
	}


	//FREE_OBJECT(class_data);
	//FREE_OBJECT(Classes_of_elements_expanded);

	if (f_v) {
		cout << "group_theory_with_magma::get_classses_expanded done" << endl;
	}

}

void group_theory_with_magma::split_by_classes(
		groups::sims *Sims,
		groups::any_group *Any_group,
		classes_of_elements_expanded *Classes_of_elements_expanded,
		std::string &fname,
		std::string &col_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes" << endl;
	}

	//classes_of_elements_expanded *Classes_of_elements_expanded;
	//data_structures_groups::vector_ge *Reps;


#if 0
	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"before get_classses_expanded" << endl;
	}
	get_classses_expanded(
			Sims,
			Any_group,
			expand_by_go,
			Classes_of_elements_expanded,
			Reps,
			verbose_level);
	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"after get_classses_expanded" << endl;
	}
#endif



	int nb_classes;

	nb_classes = Classes_of_elements_expanded->Classes->nb_classes;

	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::set_of_sets *SoS;

	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"before read_column_as_set_of_sets" << endl;
	}
	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS,
			verbose_level);

	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"after read_column_as_set_of_sets" << endl;
	}

	other::data_structures::sorting Sorting;


	std::string *Table;
	int nb_rows, nb_cols;

	nb_rows = SoS->nb_sets;
	nb_cols = 3 + nb_classes;

	Table = new std::string [nb_rows * nb_cols];

	int i;

	int *First;

	First = NEW_int(nb_classes);


	// we assume that all classes have been expanded:
	int j;
	for (j = 0; j < nb_classes; j++) {
		if (j == 0) {
			First[j] = 0;
		}
		else {
			First[j] = First[j - 1] + Classes_of_elements_expanded->Orbit_of_elements[j - 1]->orbit_length;
		}
	}


	std::string *Table2;
	int nb_rows2 = nb_rows + 1;
	int nb_cols2 = nb_cols;


	Table2 = new std::string [nb_rows2 * nb_cols2];

	for (j = 0; j < nb_classes; j++) {

		long int class_size;


		class_size = Classes_of_elements_expanded->Orbit_of_elements[j]->orbit_length;

		Table2[0 * nb_cols2 + 3 + j] = std::to_string(class_size);

	}


	for (i = 0; i < SoS->nb_sets; i++) {


		if (f_v) {
			cout << "group_theory_with_magma::split_by_classes "
					"The set " << i << " / " << SoS->nb_sets << " is : ";
			Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
			cout << endl;
		}

		long int *vec_combined;

		vec_combined = NEW_lint(SoS->Set_size[i]);

		std::vector<std::vector<long int>> V;

		// prepare V to be a vector of empty vectors:
		for (j = 0; j < nb_classes; j++) {
			std::vector<long int> v;

			V.push_back(v);
		}

		int h, idx;

		for (h = 0; h < SoS->Set_size[i]; h++) {

			long int a;
			int f_found;

			a = SoS->Sets[i][h];
			f_found = false;

			for (j = 0; j < nb_classes; j++) {
				if (Sorting.lint_vec_search(
						Classes_of_elements_expanded->Orbit_of_elements[j]->Table_of_elements,
						Classes_of_elements_expanded->Orbit_of_elements[j]->orbit_length,
						a, idx, 0 /*verbose_level*/)) {
					//V[j].push_back(idx);
					V[j].push_back(a);
					f_found = true;
					break;
				}
			}
			if (!f_found) {
				cout << "group_theory_with_magma::split_by_classes did not find element " << endl;
			}

		}

		Table[i * nb_cols + 0] = std::to_string(i);
		Table[i * nb_cols + 1] = "\"" + Lint_vec_stringify(SoS->Sets[i], SoS->Set_size[i]) + "\"";


		Table2[(i + 1) * nb_cols2 + 0] = std::to_string(i);
		Table2[(i + 1) * nb_cols2 + 1] = std::to_string(SoS->Set_size[i]);


#if 0
		int cur;

		cur = 0;
		for (j = 0; j < nb_classes; j++) {
			long int *vec;
			int len;

			len = V[j].size();
			vec = NEW_lint(len);
			for (h = 0; h < len; h++) {
				vec[h] = First[j] + V[j][h];
			}
			Lint_vec_copy(vec, vec_combined + cur, len);
			cur += len;
			FREE_lint(vec);
		}
		Table[i * nb_cols + 2] = "\"" + Lint_vec_stringify(vec_combined, SoS->Set_size[i]) + "\"";
#endif
		for (j = 0; j < nb_classes; j++) {

			long int *vec;
			int len;

			len = V[j].size();
			vec = NEW_lint(len);
			for (h = 0; h < len; h++) {
				vec[h] = V[j][h];
			}



			Table[i * nb_cols + 3 + j] = "\"" + Lint_vec_stringify(vec, len) + "\"";

			Table2[(i + 1) * nb_cols2 + 3 + j] = std::to_string(len);

			FREE_lint(vec);

		}

		FREE_lint(vec_combined);


	}




	FREE_int(First);


	std::string fname_identify;
	std::string fname_identify_size;

	fname_identify = Any_group->label + "_split_by_classes.csv";
	fname_identify_size = Any_group->label + "_split_by_classes_size.csv";

	std::string *Col_headings;

	Col_headings = new string [nb_cols];

	Col_headings[0] = "line";
	Col_headings[1] = "set";
	Col_headings[2] = "set_out";

	for (j = 0; j < nb_classes; j++) {
		Col_headings[3 + j] = "C" + std::to_string(j);
	}

	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"nb_rows = " << nb_rows << endl;
		cout << "group_theory_with_magma::split_by_classes "
				"nb_cols = " << nb_cols << endl;
	}

	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"writing file " << fname_identify << endl;
	}

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_identify,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"written file " << fname_identify << " of size "
				<< Fio.file_size(fname_identify) << endl;
	}




	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"writing file " << fname_identify_size << endl;
	}

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_identify_size,
			nb_rows2, nb_cols2, Table2,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "group_theory_with_magma::split_by_classes "
				"written file " << fname_identify_size << " of size "
				<< Fio.file_size(fname_identify_size) << endl;
	}




	delete [] Col_headings;
	delete [] Table;
	delete [] Table2;


	FREE_OBJECT(SoS);
	//FREE_OBJECT(Classes_of_elements_expanded);
	//FREE_OBJECT(Reps);


}


void group_theory_with_magma::identify_elements_by_classes(
		groups::sims *Sims,
		groups::any_group *Any_group_H,
		groups::any_group *Any_group_G,
		classes_of_elements_expanded *Classes_of_elements_expanded,
		std::string &fname, std::string &col_label,
		int *&Class_index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes" << endl;
	}


	//classes_of_elements_expanded *Classes_of_elements_expanded;
	//data_structures_groups::vector_ge *Reps;

#if 0
	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes "
				"before get_classses_expanded" << endl;
	}
	get_classses_expanded(
			Sims,
			Any_group_H,
			expand_by_go,
			Classes_of_elements_expanded,
			Reps,
			verbose_level);
	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes "
				"after get_classses_expanded" << endl;
	}
#endif

	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::set_of_sets *SoS;

	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes "
				"before read_column_as_set_of_sets" << endl;
	}
	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS,
			verbose_level);

	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes "
				"after read_column_as_set_of_sets" << endl;
	}

	int nb_elements;

	nb_elements = SoS->nb_sets;

	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes "
				"nb_elements = " << nb_elements << endl;
	}

	other::data_structures::sorting Sorting;

	Class_index = NEW_int(nb_elements);

	int nb_classes;

	nb_classes = Classes_of_elements_expanded->Classes->nb_classes;
	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes "
				"nb_classes = " << nb_classes << endl;
		cout << "group_theory_with_magma::identify_elements_by_classes "
				"nb_expanded_classes = " << Classes_of_elements_expanded->nb_idx << endl;
	}

	int *Elt;
	int *data;
	int i, j, idx;
	long int a;
	int f_found;
	int f_is_member;

	Elt = NEW_int(Any_group_G->A->elt_size_in_int);
	data = NEW_int(Any_group_G->A->make_element_size);


	for (i = 0; i < nb_elements; i++) {

		Lint_vec_copy_to_int(SoS->Sets[i], data, Any_group_G->A->make_element_size);

		Any_group_G->A->Group_element->make_element(Elt, data, 0 /*verbose_level */);


		algebra::ring_theory::longinteger_object rk;

		if (f_v) {
			cout << "group_theory_with_magma::identify_elements_by_classes "
					"before Sims->test_membership_and_rank_element" << endl;
		}
		f_is_member = Sims->test_membership_and_rank_element(
				rk, Elt, 0 /*verbose_level */);
		if (f_v) {
			cout << "group_theory_with_magma::identify_elements_by_classes "
					"after Sims->test_membership_and_rank_element" << endl;
		}


		if (f_is_member) {

			a = rk.as_lint();

			f_found = false;

			for (j = 0; j < Classes_of_elements_expanded->nb_idx; j++) {
				if (Sorting.lint_vec_search(
						Classes_of_elements_expanded->Orbit_of_elements[j]->Table_of_elements,
						Classes_of_elements_expanded->Orbit_of_elements[j]->orbit_length,
						a, idx, 0 /*verbose_level*/)) {
					Class_index[i] = j;
					f_found = true;
					break;
				}
			}
			if (!f_found) {
				if (f_v) {
					cout << "group_theory_with_magma::identify_elements_by_classes "
							"did not find element " << endl;
				}
				Class_index[i] = -2;
				//exit(1);
			}
		}
		else {
			Class_index[i] = -1;
		}

	}

	//other::orbiter_kernel_system::file_io Fio;
	std::string new_col_label;

	new_col_label= "class_idx";

	string fname_out;

	Fio.Csv_file_support->append_column_of_int(
			fname, fname_out,
			Class_index, nb_elements,
			new_col_label,
			verbose_level);

	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes written file "
				<< fname_out << " of size " << Fio.file_size(fname_out) << endl;
	}


	//FREE_OBJECT(Classes_of_elements_expanded);
	//FREE_OBJECT(Reps);
	FREE_OBJECT(SoS);
	FREE_int(Elt);
	FREE_int(data);

	if (f_v) {
		cout << "group_theory_with_magma::identify_elements_by_classes done" << endl;
	}
}

void group_theory_with_magma::diagram_of_elements(
		groups::any_group *AG,
		classes_of_elements_expanded *Classes_of_elements_expanded,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_with_magma::diagram_of_elements" << endl;
	}


	algebra::basic_algebra::group_diagram *Group_diagram;

	Group_diagram = NEW_OBJECT(algebra::basic_algebra::group_diagram);

	int base_len;
	int i;
	long int *transversal_length;


	//groups::strong_generators *Subgroup_gens;
	//groups::sims *Subgroup_sims;

	groups::sims *Sims;
	if (AG->Subgroup_sims == NULL) {
		cout << "group_theory_with_magma::diagram_of_elements "
				"Subgroup_sims == NULL" << endl;
		exit(1);
	}

	Sims = AG->Subgroup_sims;


	base_len = Sims->my_base_len; //base_len();

	transversal_length = NEW_lint(base_len);

	for (i = 0; i < base_len; i++) {
		transversal_length[i] = Sims->get_orbit_length(i); //transversal_length_i(i);
	}

	Group_diagram->init(
			AG->label,
			transversal_length, base_len,
			verbose_level);





	// classes_of_elements_expanded *Classes_of_elements_expanded;
	//data_structures_groups::vector_ge *Reps;

#if 0
	int expand_by_go = Sims->group_order_lint();

	if (f_v) {
		cout << "group_theory_with_magma::diagram_of_elements "
				"before get_classses_expanded" << endl;
	}
	get_classses_expanded(
			Sims,
			AG,
			expand_by_go,
			Classes_of_elements_expanded,
			Reps,
			verbose_level - 2);
	if (f_v) {
		cout << "group_theory_with_magma::diagram_of_elements "
				"after get_classses_expanded" << endl;
	}
#endif

	other::data_structures::set_of_sets_lint *Classes;

	if (f_v) {
		cout << "group_theory_with_magma::diagram_of_elements "
				"before Classes_of_elements_expanded->get_classes_as_set_of_sets" << endl;
	}
	Classes = Classes_of_elements_expanded->get_classes_as_set_of_sets(
				verbose_level - 2);
	if (f_v) {
		cout << "group_theory_with_magma::diagram_of_elements "
				"after Classes_of_elements_expanded->get_classes_as_set_of_sets" << endl;
	}


	int j, len;
	long int a;
	long int *Coloring;
	long int pos_i, pos_j;

	Coloring = NEW_lint(Group_diagram->nb_rows * Group_diagram->nb_cols);



	for (i = 0; i < Classes->nb_sets; i++) {
		len = Classes->Set_size[i];
		for (j = 0; j < len; j++) {
			a = Classes->Sets[i][j];

			Group_diagram->place_element_by_rank(
					a, pos_i, pos_j);
			Coloring[pos_i * Group_diagram->nb_cols + pos_j] = i;

		}
	}


	string fname;

	fname = AG->label + "_class_diagram.csv";

	other::orbiter_kernel_system::file_io Fio;

	std::string col_heading;

	col_heading = "Class";

	Fio.Csv_file_support->lint_matrix_write_csv_tabulated(
			fname, col_heading,
			Coloring, Group_diagram->nb_rows, Group_diagram->nb_cols, verbose_level);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	FREE_OBJECT(Group_diagram);
	FREE_lint(transversal_length);
	FREE_lint(Coloring);

	if (f_v) {
		cout << "group_theory_with_magma::diagram_of_elements done" << endl;
	}
}



}}}





