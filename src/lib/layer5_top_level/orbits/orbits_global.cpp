/*
 * orbits_global.cpp
 *
 *  Created on: Feb 16, 2024
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {



orbits_global::orbits_global()
{
	Record_birth();

}

orbits_global::~orbits_global()
{
	Record_death();

}


void orbits_global::compute_orbit_of_set(
		long int *the_set, int set_size,
		actions::action *A1, actions::action *A2,
		data_structures_groups::vector_ge *gens,
		std::string &label_set,
		std::string &label_group,
		long int *&Table,
		int &orbit_length,
		int verbose_level)
// called by any_group::orbits_on_set_from_file
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set" << endl;
	}
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set A1=";
		A1->print_info();
		cout << "orbits_global::compute_orbit_of_set A2=";
		A2->print_info();
	}
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set computing orbit of the set: ";
		Lint_vec_print(cout, the_set, set_size);
		cout << endl;
	}


	orbits_schreier::orbit_of_sets *OS;

	OS = NEW_OBJECT(orbits_schreier::orbit_of_sets);

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"before OS->init" << endl;
	}
	OS->init(
			A1, A2, the_set, set_size, gens, verbose_level - 2);
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"after OS->init" << endl;
	}

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"Found an orbit of length " << OS->used_length << endl;
	}

	int set_size1;

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"before OS->get_table_of_orbits" << endl;
	}
	OS->get_table_of_orbits_and_hash_values(
			Table,
			orbit_length, set_size1, verbose_level - 2);
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"after OS->get_table_of_orbits" << endl;
	}

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"before OS->get_table_of_orbits" << endl;
	}
	OS->get_table_of_orbits(
			Table,
			orbit_length, set_size, verbose_level);
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"after OS->get_table_of_orbits" << endl;
	}


	string fname_csv;

#if 0
	// write transporter as csv file:


	data_structures_groups::vector_ge *Coset_reps;

	if (f_v) {
		cout << "before OS->make_table_of_coset_reps" << endl;
	}
	OS->make_table_of_coset_reps(Coset_reps, verbose_level);
	if (f_v) {
		cout << "after OS->make_table_of_coset_reps" << endl;
	}

	fname = label_set + "_orbit_under_" + label_group + "_transporter.csv";

	Coset_reps->write_to_csv_file_coded(fname, verbose_level);

	// testing Coset_reps

	if (f_v) {
		cout << "testing Coset_reps " << endl;
	}

	long int rk0 = the_set[0];
	long int rk1;

	for (int i = 0; i < orbit_length; i++) {
		rk1 = A2->element_image_of(rk0, Coset_reps->ith(i), 0);
		if (rk1 != Table[i * set_size + 0]) {
			cout << "rk1 != Table[i * set_size + 0], i=" << i << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "testing Coset_reps passes" << endl;
	}
#endif

	// write as csv file:


	fname_csv = label_set + "_orbit_under_" + label_group + ".csv";

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"Writing orbit to file " << fname_csv << endl;
	}
	other::orbiter_kernel_system::file_io Fio;


	std::string *sTable;
	int idx;
	int nb_rows = orbit_length;
	int nb_cols = 1;


	sTable = new string[nb_rows * nb_cols];

	for (idx = 0; idx < nb_rows; idx++) {

		string s;

		s = Lint_vec_stringify(Table + idx * set_size, set_size);
		sTable[idx * nb_cols + 0] = "\"" + s + "\"";
	}

	string headings;

	headings.assign("Set");


	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings(fname_csv,
			nb_rows, nb_cols, sTable,
			headings,
			verbose_level);
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}


	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"Written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;
	}

	delete [] sTable;


#if 0
	// write as txt file:


	fname = label_set + "_orbit_under_" + label_group + ".txt";

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"Writing table to file " << fname << endl;
	}
	{
		ofstream ost(fname);
		int i;
		for (i = 0; i < orbit_length; i++) {
			ost << set_size;
			for (int j = 0; j < set_size; j++) {
				ost << " " << Table[i * set_size + j];
			}
			ost << endl;
		}
		ost << -1 << " " << orbit_length << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
#endif

	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"before FREE_OBJECT(OS)" << endl;
	}
	FREE_OBJECT(OS);
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set "
				"after FREE_OBJECT(OS)" << endl;
	}
	//FREE_OBJECT(Coset_reps);
	if (f_v) {
		cout << "orbits_global::compute_orbit_of_set done" << endl;
	}
}


void orbits_global::orbits_on_points(
		actions::action *A2,
		groups::strong_generators *Strong_gens,
		int f_load_save,
		std::string &prefix,
		groups::orbits_on_something *&Orb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_points" << endl;
	}
	//cout << "computing orbits on points:" << endl;


	//orbits_on_something *Orb;

	Orb = NEW_OBJECT(groups::orbits_on_something);

	if (f_v) {
		cout << "orbits_global::orbits_on_points "
				"before Orb->init" << endl;
	}
	Orb->init(
			A2,
			Strong_gens,
			f_load_save,
			prefix,
			verbose_level);
	if (f_v) {
		cout << "orbits_global::orbits_on_points "
				"after Orb->init" << endl;
	}




	if (f_v) {
		cout << "orbits_global::orbits_on_points done" << endl;
	}
}




void orbits_global::orbits_on_points_from_vector_ge(
		actions::action *A2,
		data_structures_groups::vector_ge *gens,
		int f_load_save,
		std::string &prefix,
		groups::orbits_on_something *&Orb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_vector_ge" << endl;
	}
	//cout << "computing orbits on points:" << endl;


	//orbits_on_something *Orb;

	Orb = NEW_OBJECT(groups::orbits_on_something);

	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_vector_ge "
				"before Orb->init_from_vector_ge" << endl;
	}
	Orb->init_from_vector_ge(
			A2,
			gens,
			f_load_save,
			prefix,
			verbose_level);
	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_vector_ge "
				"after Orb->init_from_vector_ge" << endl;
	}




	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_vector_ge done" << endl;
	}
}

void orbits_global::orbits_on_set_system_from_file(
		groups::any_group *AG,
		std::string &fname_csv,
		int number_of_columns, int first_column,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_set_system_from_file" << endl;
	}
	if (f_v) {
		cout << "computing orbits on set system from file "
			<< fname_csv << ":" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;
	int *M;
	int m, n;
	long int *Table;
	int i, j;

	Fio.Csv_file_support->int_matrix_read_csv(
			fname_csv, M,
			m, n, verbose_level);
	if (f_v) {
		cout << "read a matrix of size " << m << " x " << n << endl;
	}



	Table = NEW_lint(m * number_of_columns);
	for (i = 0; i < m; i++) {
		for (j = 0; j < number_of_columns; j++) {
			Table[i * number_of_columns + j] =
					M[i * n + first_column + j];
		}
	}
	actions::action *A_on_sets;
	int set_size;

	set_size = number_of_columns;

	if (f_v) {
		cout << "creating action on sets:" << endl;
	}
	A_on_sets = AG->A->Induced_action->create_induced_action_on_sets(
			m /* nb_sets */,
			set_size, Table,
			verbose_level);

	actions::action_global Action;
	groups::schreier *Sch;
	int first, a;

	if (f_v) {
		cout << "computing orbits on sets:" << endl;
	}
	Action.compute_orbits_on_points(
			A_on_sets, Sch,
			AG->get_strong_generators()->gens, verbose_level);

	if (f_v) {
		cout << "The orbit lengths are:" << endl;
		Sch->print_orbit_lengths(cout);
	}

	if (f_v) {
		cout << "The orbits are:" << endl;
		//Sch->print_and_list_orbits(cout);
		for (i = 0; i < Sch->nb_orbits; i++) {
			cout << " Orbit " << i << " / " << Sch->nb_orbits
					<< " : " << Sch->orbit_first[i] << " : " << Sch->orbit_len[i];
			cout << " : ";

			first = Sch->orbit_first[i];
			a = Sch->orbit[first + 0];
			cout << a << " : ";
			Lint_vec_print(cout, Table + a * set_size, set_size);
			cout << endl;
			//Sch->print_and_list_orbit_tex(i, ost);
		}
	}
	string fname;
	other::data_structures::string_tools ST;

	fname = fname_csv;
	ST.chop_off_extension(fname);
	fname += "_orbit_reps.txt";

	{
		ofstream ost(fname);

		for (i = 0; i < Sch->nb_orbits; i++) {

			first = Sch->orbit_first[i];
			a = Sch->orbit[first + 0];
			ost << set_size;
			for (j = 0; j < set_size; j++) {
				ost << " " << Table[a * set_size + j];
			}
			ost << endl;
		}
		ost << -1 << " " << Sch->nb_orbits << endl;
	}
	if (f_v) {
		cout << "orbits_global::orbits_on_set_system_from_file done" << endl;
	}
}

void orbits_global::orbits_on_set_from_file(
		groups::any_group *AG,
		std::string &fname_csv, int verbose_level)
// called from group_theoretic_activity: f_orbit_of_set_from_file
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_set_from_file" << endl;
	}

	if (f_v) {
		cout << "orbits_global::orbits_on_set_from_file "
				"computing orbit of set from file "
			<< fname_csv << ":" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;
	long int *the_set;
	int set_sz;

	Fio.read_set_from_file(
			fname_csv,
			the_set, set_sz,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "orbits_global::orbits_on_set_from_file "
				"read a set of size " << set_sz << endl;
	}


	string label_set;
	other::data_structures::string_tools ST;

	label_set.assign(fname_csv);
	ST.chop_off_extension(label_set);

	orbits::orbits_global Orbits;
	long int *Table;
	int size;

	if (f_v) {
		cout << "orbits_global::orbits_on_set_from_file "
				"before Orbits.compute_orbit_of_set" << endl;
	}

	Orbits.compute_orbit_of_set(
			the_set, set_sz,
			AG->A_base, AG->A,
			AG->Subgroup_gens->gens,
			label_set,
			AG->label,
			Table, size,
			verbose_level);

	if (f_v) {
		cout << "orbits_global::orbits_on_set_from_file "
				"after Orbits.compute_orbit_of_set" << endl;
	}

	FREE_lint(Table);

	if (f_v) {
		cout << "orbits_global::orbits_on_set_from_file done" << endl;
	}
}


void orbits_global::orbit_of(
		groups::any_group *AG,
		int point_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbit_of" << endl;
	}
	groups::schreier *Sch;
	Sch = NEW_OBJECT(groups::schreier);

	if (f_v) {
		cout << "orbits_global::orbit_of "
				"computing orbit of point " << point_idx << ":" << endl;
	}

	//A->all_point_orbits(*Sch, verbose_level);

	Sch->init(AG->A_base, verbose_level - 2);

#if 0
	if (!A->f_has_strong_generators) {
		cout << "orbits_global::orbit_of !f_has_strong_generators" << endl;
		exit(1);
		}
#endif

	Sch->init_generators(
			*AG->Subgroup_gens->gens /* *strong_generators */,
			verbose_level - 2);
	Sch->initialize_tables();
	Sch->compute_point_orbit(point_idx, verbose_level);

	int orbit_idx = 0;

	if (f_v) {
		cout << "orbits_global::orbit_of "
				"computing orbit of point " << point_idx << " done" << endl;
	}

	string fname_tree_mask;

	fname_tree_mask = AG->label + "_orbit_of_point_" + std::to_string(point_idx) + ".layered_graph";


	if (f_v) {
		cout << "orbits_global::orbit_of "
				"before Sch->export_tree_as_layered_graph_and_save" << endl;
	}
	Sch->export_tree_as_layered_graph_and_save(
			orbit_idx,
			fname_tree_mask,
			verbose_level - 1);
	if (f_v) {
		cout << "orbits_global::orbit_of "
				"after Sch->export_tree_as_layered_graph_and_save" << endl;
	}

	groups::strong_generators *SG_stab;
	algebra::ring_theory::longinteger_object full_group_order;

	AG->Subgroup_gens->group_order(full_group_order);


	if (f_v) {
		cout << "orbits_global::orbit_of computing the stabilizer "
				"of the rep of orbit " << orbit_idx << endl;
		cout << "orbits_global::orbit_of "
				"orbit length = " << Sch->orbit_len[orbit_idx] << endl;
	}

	if (f_v) {
		cout << "orbits_global::orbit_of "
				"before Sch->stabilizer_orbit_rep" << endl;
	}

	SG_stab = Sch->stabilizer_orbit_rep(
			AG->A_base,
			full_group_order,
			0 /* orbit_idx */, 0 /*verbose_level*/);

	if (f_v) {
		cout << "orbits_global::orbit_of "
				"after Sch->stabilizer_orbit_rep" << endl;
	}


	cout << "orbits_global::orbit_of "
			"The stabilizer of the orbit rep has been computed:" << endl;
	SG_stab->print_generators(cout, verbose_level - 1);
	SG_stab->print_generators_tex();

#if 0

	groups::schreier *shallow_tree;

	if (f_v) {
		cout << "orbits_global::orbit_of "
				"computing shallow Schreier tree:" << endl;
	}

	#if 0
	enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
			shallow_schreier_tree_standard;
			//shallow_schreier_tree_Seress_deterministic;
			//shallow_schreier_tree_Seress_randomized;
			//shallow_schreier_tree_Sajeeb;
	#endif

	int orbit_idx = 0;
	int f_randomized = true;

	Sch->shallow_tree_generators(orbit_idx,
			f_randomized,
			shallow_tree,
			verbose_level);

	if (f_v) {
		cout << "orbits_global::orbit_of "
				"computing shallow Schreier tree done." << endl;
	}

	fname_tree_mask = label + "_%d_shallow.layered_graph";

	shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
			fname_tree_mask,
			verbose_level - 1);
#endif

	if (f_v) {
		cout << "orbits_global::orbit_of done" << endl;
	}
}

void orbits_global::orbits_on_points(
		groups::any_group *AG,
		groups::orbits_on_something *&Orb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_points" << endl;
	}
	orbits::orbits_global Orbits;


	int f_load_save = false;
	string prefix;

	prefix.assign(AG->label);

	if (f_v) {
		cout << "orbits_global::orbits_on_points "
				"before Orbits.orbits_on_points" << endl;
	}
	Orbits.orbits_on_points(
			AG->A,
			AG->Subgroup_gens,
			f_load_save,
			prefix,
			Orb,
			verbose_level);
	if (f_v) {
		cout << "orbits_global::orbits_on_points "
				"after Orbits.orbits_on_points" << endl;
	}


	if (f_v) {
		cout << "orbits_global::orbits_on_points done" << endl;
	}
}

void orbits_global::orbits_on_points_from_generators(
		groups::any_group *AG,
		data_structures_groups::vector_ge *gens,
		groups::orbits_on_something *&Orb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_generators" << endl;
	}
	orbits::orbits_global Orbits;


	int f_load_save = false;
	string prefix;

	prefix.assign(AG->label);

	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_generators "
				"before Orbits.orbits_on_points_from_vector_ge" << endl;
	}
	Orbits.orbits_on_points_from_vector_ge(
			AG->A,
			gens,
			f_load_save,
			prefix,
			Orb,
			verbose_level);

	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_generators "
				"after Orbits.orbits_on_points_from_vector_ge" << endl;
	}


	if (f_v) {
		cout << "orbits_global::orbits_on_points_from_generators done" << endl;
	}
}


void orbits_global::orbits_on_subsets(
		groups::any_group *AG,
		poset_classification::poset_classification_control *Control,
		poset_classification::poset_classification *&PC,
		int subset_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"subset_size=" << subset_size << endl;
	}
	poset_classification::poset_with_group_action *Poset;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);

	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"control=" << endl;
		Control->print();
	}
	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"label=" << AG->label << endl;
	}
	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"A_base=" << endl;
		AG->A_base->print_info();
	}
	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"A=" << endl;
		AG->A->print_info();
	}
	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"group order" << endl;

		algebra::ring_theory::longinteger_object go;

		AG->Subgroup_gens->group_order(go);

		cout << go << endl;
	}


	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"before Poset->init_subset_lattice" << endl;
	}
	Poset->init_subset_lattice(
			AG->A_base, AG->A,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			subset_size,
			verbose_level);
	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	if (f_v) {
		cout << "orbits_global::orbits_on_subsets "
				"before orbits_on_poset_post_processing" << endl;
	}
	orbits_on_poset_post_processing(
			AG,
			PC, subset_size,
			verbose_level);
	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"after orbits_on_poset_post_processing" << endl;
	}


	if (f_v) {
		cout << "orbits_global::orbits_on_subsets done" << endl;
	}
}


void orbits_global::orbits_of_one_subset(
		groups::any_group *AG,
		long int *set, int sz,
		std::string &label_set,
		actions::action *A_base, actions::action *A,
		long int *&Table,
		int &size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_of_one_subset" << endl;
	}


	orbits::orbits_global Orbits;

	if (f_v) {
		cout << "orbits_global::orbits_of_one_subset "
				"before Orbits.compute_orbit_of_set" << endl;
	}

	Orbits.compute_orbit_of_set(
			set, sz,
			A_base, A,
			AG->Subgroup_gens->gens,
			label_set,
			AG->label,
			Table, size,
			verbose_level);

	if (f_v) {
		cout << "orbits_global::orbits_of_one_subset "
				"after Orbits.compute_orbit_of_set" << endl;
	}



	//FREE_lint(Table);
	if (f_v) {
		cout << "orbits_global::orbits_of_one_subset" << endl;
	}

}


void orbits_global::orbits_on_poset_post_processing(
		groups::any_group *AG,
		poset_classification::poset_classification *PC,
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::orbits_on_poset_post_processing" << endl;
	}



#if 0
	if (Descr->f_test_if_geometric) {
		int d = Descr->test_if_geometric_depth;

		//for (depth = 0; depth <= orbits_on_subsets_size; depth++) {

		cout << "Orbits on subsets of size " << d << ":" << endl;
		PC->list_all_orbits_at_level(d,
				false /* f_has_print_function */,
				NULL /* void (*print_function)(std::ostream &ost, int len, int *S, void *data)*/,
				NULL /* void *print_function_data*/,
				true /* f_show_orbit_decomposition */,
				true /* f_show_stab */,
				false /* f_save_stab */,
				true /* f_show_whole_orbit*/);
		int nb_orbits, orbit_idx;

		nb_orbits = PC->nb_orbits_at_level(d);
		for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {

			int orbit_length;
			long int *Orbit;

			cout << "before PC->get_whole_orbit depth " << d
					<< " orbit " << orbit_idx
					<< " / " << nb_orbits << ":" << endl;
			PC->get_whole_orbit(
					d, orbit_idx,
					Orbit, orbit_length, verbose_level);
			cout << "depth " << d << " orbit " << orbit_idx
					<< " / " << nb_orbits << " has length "
					<< orbit_length << ":" << endl;
			Orbiter->Lint_vec.matrix_print(Orbit, orbit_length, d);

			action *Aut;
			longinteger_object ago;
			nauty_interface_with_group Nauty;

			Aut = Nauty.create_automorphism_group_of_block_system(
				A2->degree /* nb_points */,
				orbit_length /* nb_blocks */,
				depth /* block_size */, Orbit,
				verbose_level);
			Aut->group_order(ago);
			cout << "The automorphism group of the set system "
					"has order " << ago << endl;

			FREE_OBJECT(Aut);
			FREE_lint(Orbit);
		}
		if (nb_orbits == 2) {
			cout << "the number of orbits at depth " << depth
					<< " is two, we will try create_automorphism_"
					"group_of_collection_of_two_block_systems" << endl;
			long int *Orbit1;
			int orbit_length1;
			long int *Orbit2;
			int orbit_length2;

			cout << "before PC->get_whole_orbit depth " << d
					<< " orbit " << orbit_idx
					<< " / " << nb_orbits << ":" << endl;
			PC->get_whole_orbit(
					depth, 0 /* orbit_idx*/,
					Orbit1, orbit_length1, verbose_level);
			cout << "depth " << d << " orbit " << 0
					<< " / " << nb_orbits << " has length "
					<< orbit_length1 << ":" << endl;
			Orbiter->Lint_vec.matrix_print(Orbit1, orbit_length1, d);

			PC->get_whole_orbit(
					depth, 1 /* orbit_idx*/,
					Orbit2, orbit_length2, verbose_level);
			cout << "depth " << d << " orbit " << 1
					<< " / " << nb_orbits << " has length "
					<< orbit_length2 << ":" << endl;
			Orbiter->Lint_vec.matrix_print(Orbit2, orbit_length2, d);

			action *Aut;
			longinteger_object ago;
			nauty_interface_with_group Nauty;

			Aut = Nauty.create_automorphism_group_of_collection_of_two_block_systems(
				A2->degree /* nb_points */,
				orbit_length1 /* nb_blocks */,
				depth /* block_size */, Orbit1,
				orbit_length2 /* nb_blocks */,
				depth /* block_size */, Orbit2,
				verbose_level);
			Aut->group_order(ago);
			cout << "The automorphism group of the collection of two set systems "
					"has order " << ago << endl;

			FREE_OBJECT(Aut);
			FREE_lint(Orbit1);
			FREE_lint(Orbit2);

		} // if nb_orbits == 2
	} // if (f_test_if_geometric)
#endif



	if (f_v) {
		cout << "orbits_global::orbits_on_poset_post_processing done" << endl;
	}
}








#if 0
void orbits_global::do_conjugacy_class_of_element(
		apps_algebra::any_group *AG,
		std::string &elt_label, std::string &elt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::do_conjugacy_class_of_element" << endl;
	}


	int *data, sz;

	Int_vec_scan(elt_text, data, sz);

	if (f_v) {
		cout << "computing conjugacy class of ";
		Int_vec_print(cout, data, sz);
		cout << endl;
	}


	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	A->make_element(Elt, data, 0 /* verbose_level */);

	if (!A->f_has_sims) {
		if (f_v) {
			cout << "orbits_global::do_conjugacy_class_of_element "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			groups::sims *S;

			S = LG->Strong_gens->create_sims(verbose_level);

			if (f_v) {
				cout << "orbits_global::do_conjugacy_class_of_element "
						"before init_sims" << endl;
			}
			A->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "orbits_global::do_conjugacy_class_of_element "
						"after init_sims" << endl;
			}
		}

	}
	groups::sims *S;

	S = A->Sims;

	long int the_set[1];
	int set_size = 1;

	the_set[0] = S->element_rank_lint(Elt);

	if (f_v) {
		cout << "computing conjugacy class of " << endl;
		A->element_print_latex(Elt, cout);
		cout << "which is the set ";
		Lint_vec_print(cout, the_set, set_size);
		cout << endl;
	}


	actions::action A_conj;
	if (f_v) {
		cout << "orbits_global::do_conjugacy_class_of_element "
				"before A_conj.induced_action_by_conjugation" << endl;
	}
	A_conj.induced_action_by_conjugation(S, S,
			false /* f_ownership */, false /* f_basis */,
			verbose_level);
	if (f_v) {
		cout << "orbits_global::do_conjugacy_class_of_element "
				"created action by conjugation" << endl;
	}



	//schreier Classes;
	//Classes.init(&A_conj, verbose_level - 2);
	//Classes.init_generators(*A1->Strong_gens->gens, verbose_level - 2);
	//cout << "Computing orbits:" << endl;
	//Classes.compute_all_point_orbits(1 /*verbose_level - 1*/);
	//cout << "found " << Classes.nb_orbits << " conjugacy classes" << endl;




	algebra_global_with_action Algebra;

	long int *Table;
	int orbit_length;

	Algebra.compute_orbit_of_set(
			the_set, set_size,
			A, &A_conj,
			LG->Strong_gens->gens,
			elt_label,
			LG->label,
			Table,
			orbit_length,
			verbose_level);


	// write as txt file:

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname = elt_label + "_orbit_under_" + LG->label + "_elements_coded.csv";

	if (f_v) {
		cout << "Writing table to file " << fname << endl;
	}
	{
		ofstream ost(fname);
		int i;

		// header line:
		ost << "ROW";
		for (int j = 0; j < A->make_element_size; j++) {
			ost << ",C" << j;
		}
		ost << endl;

		for (i = 0; i < orbit_length; i++) {

			ost << i;
			S->element_unrank_lint(Table[i], Elt);

			for (int j = 0; j < A->make_element_size; j++) {
				ost << "," << Elt[j];
			}
			ost << endl;
		}
		ost << "END" << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	FREE_int(Elt);
	FREE_int(data);
	FREE_lint(Table);

	if (f_v) {
		cout << "orbits_global::do_conjugacy_class_of_element done" << endl;
	}
}
#endif

#if 0
void orbits_global::do_orbits_on_group_elements_under_conjugation(
		apps_algebra::any_group *AG,
		std::string &fname_group_elements_coded,
		std::string &fname_transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_global::do_orbits_on_group_elements_under_conjugation" << endl;
	}




	if (!AG->A->f_has_sims) {
		if (f_v) {
			cout << "orbits_global::do_orbits_on_group_elements_under_conjugation "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			//sims *S;

			AG->A->Known_groups->create_sims(verbose_level);

#if 0
			if (f_v) {
				cout << "orbits_global::do_orbits_on_group_elements_under_conjugation before init_sims" << endl;
			}
			A2->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "orbits_global::do_orbits_on_group_elements_under_conjugation after init_sims" << endl;
			}
#endif
		}

	}





	groups::sims *S;

	S = AG->A->Sims;

	if (f_v) {
		cout << "the group has order " << S->group_order_lint() << endl;
	}
	int *Elt;

	Elt = NEW_int(AG->A->elt_size_in_int);

	if (f_v) {
		cout << "computing the element ranks:" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	long int *the_ranks;
	data_structures_groups::vector_ge *Transporter;
	int m, n;
	int i;

	{
		int *M;
		Fio.Csv_file_support->int_matrix_read_csv(
				fname_group_elements_coded,
				M, m, n, 0 /*verbose_level*/);
		if (f_v) {
			cout << "read a set of size " << m << endl;
		}
		the_ranks = NEW_lint(m);
		for (i = 0; i < m; i++) {

			if (false) {
				cout << i << " : ";
				Int_vec_print(cout, M + i * n, n);
				cout << endl;
			}

			AG->A->Group_element->make_element(
					Elt, M + i * n, 0 /* verbose_level */);
			if (false) {
				cout << "computing rank of " << endl;
				AG->A->Group_element->element_print_latex(Elt, cout);
			}

			the_ranks[i] = S->element_rank_lint(Elt);
			if (false) {
				cout << i << " : " << the_ranks[i] << endl;
			}
		}

		FREE_int(M);
	}

	Transporter = NEW_OBJECT(data_structures_groups::vector_ge);
	Transporter->init(S->A, 0);
	{
		int *M;
		Fio.Csv_file_support->int_matrix_read_csv(
				fname_transporter,
				M, m, n, 0 /*verbose_level*/);
		if (f_v) {
			cout << "read a set of size " << m << endl;
		}
		Transporter->allocate(m, 0);
		for (i = 0; i < m; i++) {

			if (false) {
				cout << i << " : ";
				Int_vec_print(cout, M + i * n, n);
				cout << endl;
			}

			AG->A->Group_element->make_element(
					Transporter->ith(i), M + i * n, 0 /* verbose_level */);
			if (false) {
				cout << "computing rank of " << endl;
				AG->A->Group_element->element_print_latex(Elt, cout);
			}

		}

		FREE_int(M);
	}




	if (f_v) {
		cout << "computing conjugacy classes on the set " << endl;
		Lint_vec_print(cout, the_ranks, m);
		cout << endl;
	}


	apps_algebra::algebra_global_with_action Algebra;

	if (f_v) {
		cout << "orbits_global::do_orbits_on_group_elements_under_conjugation "
				"before Algebra.orbits_under_conjugation" << endl;
	}
	Algebra.orbits_under_conjugation(
			the_ranks, m, S,
			AG->get_strong_generators(),
			Transporter,
			verbose_level);
	if (f_v) {
		cout << "orbits_global::do_orbits_on_group_elements_under_conjugation "
				"after Algebra.orbits_under_conjugation" << endl;
	}




	FREE_int(Elt);

	if (f_v) {
		cout << "orbits_global::do_orbits_on_group_elements_under_conjugation done" << endl;
	}
}
#endif




}}}



