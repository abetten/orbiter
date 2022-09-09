/*
 * isomorph_worker.cpp
 *
 *  Created on: Aug 3, 2022
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {


isomorph_worker::isomorph_worker()
{
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);


	Isomorph_arguments = NULL;
	Isomorph_global = NULL;
	Iso = NULL;

	if (f_v) {
		cout << "isomorph_worker::isomorph_worker before layer2_discreta::discreta_init" << endl;
	}

	layer2_discreta::discreta_init();

	if (f_v) {
		cout << "isomorph_worker::isomorph_worker after layer2_discreta::discreta_init" << endl;
	}


}

isomorph_worker::~isomorph_worker()
{
}


void isomorph_worker::init(
		isomorph_arguments *Isomorph_arguments,
		actions::action *A_base, actions::action *A,
		poset_classification::poset_classification *gen,
		int size, int level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker::init" << endl;
	}

	isomorph_worker::Isomorph_arguments = Isomorph_arguments;

	Isomorph_global = NEW_OBJECT(isomorph_global);

	Iso = NEW_OBJECT(isomorph);

	//int f_use_database_for_starter = TRUE;

	if (f_v) {
		cout << "isomorph_worker::init before Iso->init" << endl;
	}
	Iso->init(Isomorph_arguments->prefix_iso,
			A_base, A, gen, size, level,
			Isomorph_arguments->f_use_database_for_starter,
			Isomorph_arguments->f_implicit_fusion,
			verbose_level);
		// sets size, level and initializes file names
	if (f_v) {
		cout << "isomorph_worker::init after Iso->init" << endl;
	}


	if (f_v) {
		cout << "isomorph_worker::init done" << endl;
	}
}


void isomorph_worker::execute(isomorph_arguments *Isomorph_arguments,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "isomorph_worker::execute" << endl;
	}

	if (!Isomorph_arguments->f_init_has_been_called) {
		cout << "isomorph_worker::execute please "
				"call Isomorph_arguments->init before execute" << endl;
		exit(1);
	}


#if 0
	Iso->read_data_files_for_starter(Iso->level, Isomorph_arguments->prefix_classify,
			verbose_level);
#else
	if (f_v) {
		cout << "isomorph_worker::execute before Iso->Sub->compute_nb_starter" << endl;
	}
	Iso->Sub->compute_nb_starter(Iso->level, verbose_level);
	if (f_v) {
		cout << "isomorph_worker::execute after Iso->Sub->compute_nb_starter" << endl;
	}
#endif


	if (Isomorph_arguments->f_build_db) {

		if (f_v) {
			cout << "isomorph_worker::execute before build_db" << endl;
		}


		build_db(verbose_level);


		if (f_v) {
			cout << "isomorph_worker::execute after build_db" << endl;
		}

	}

	else if (Isomorph_arguments->f_read_solutions) {


		if (f_v) {
			cout << "isomorph_worker::execute before read_solutions" << endl;
		}

		read_solutions(verbose_level);

		if (f_v) {
			cout << "isomorph_worker::execute after read_solutions" << endl;
		}

	}
	else if (Isomorph_arguments->f_compute_orbits) {


		if (f_v) {
			cout << "isomorph_worker::execute before "
					"compute_orbits" << endl;
		}

		compute_orbits(verbose_level);


		if (f_v) {
			cout << "isomorph_worker::execute after "
					"isomorph_compute_orbits" << endl;
		}
	}
	else if (Isomorph_arguments->f_isomorph_testing) {


		if (f_v) {
			cout << "isomorph_worker::execute before isomorph_testing" << endl;
		}

		isomorph_testing(verbose_level);

		if (f_v) {
			cout << "isomorph_worker::execute after isomorph_testing" << endl;
		}
	}
	else if (Isomorph_arguments->f_isomorph_report) {


		if (f_v) {
			cout << "isomorph_worker::execute before isomorph_report" << endl;
		}

		isomorph_report(verbose_level);

		if (f_v) {
			cout << "isomorph_worker::execute after isomorph_report" << endl;
		}
	}
#if 0
	else if (Isomorph_arguments->f_classification_graph) {

		if (f_v) {
			cout << "isomorph_worker::execute before "
					"isomorph_classification_graph" << endl;
		}
		isomorph_classification_graph(A, A2, gen,
			target_size,
			prefix_with_directory, prefix_iso,
			ECA->starter_size,
			verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after "
					"isomorph_classification_graph" << endl;
		}
	}
	else if (Isomorph_arguments->f_isomorph_report) {

		if (callback_report == NULL) {
			cout << "isomorph_worker::execute "
					"callback_report == NULL" << endl;
			exit(1);
			}
		if (f_v) {
			cout << "isomorph_worker::execute before isomorph_worker" << endl;
		}
		isomorph_worker(A, A2, gen,
			target_size, prefix_with_directory, prefix_iso,
			callback_report, callback_data,
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after isomorph_worker" << endl;
		}
	}
	else if (Isomorph_arguments->f_subset_orbits) {

		isomorph_worker_data WD;

		WD.the_set = NULL;
		WD.set_size = 0;
		WD.callback_data = callback_data;

		if (f_subset_orbits_file) {
			Fio.read_set_from_file(subset_orbits_fname,
					WD.the_set, WD.set_size, verbose_level);
			}
		if (f_v) {
			cout << "isomorph_worker::execute before isomorph_worker" << endl;
		}
		isomorph_worker(A, A2, gen,
			target_size, prefix_with_directory, prefix_iso,
			callback_subset_orbits, &WD,
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after isomorph_worker" << endl;
		}

		if (WD.the_set) {
			FREE_lint(WD.the_set);
		}
	}
	else if (Isomorph_arguments->f_down_orbits) {

		if (f_v) {
			cout << "isomorph_worker::execute before isomorph_compute_down_orbits" << endl;
		}
		isomorph_compute_down_orbits(A, A2, gen,
			target_size,
			prefix_with_directory, prefix_iso,
			callback_data,
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after isomorph_compute_down_orbits" << endl;
		}
	}
#endif

	if (f_v) {
		cout << "isomorph_worker::execute done" << endl;
	}
}

void isomorph_worker::build_db(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker::build_db" << endl;
	}
	if (f_v) {
		cout << "isomorph_worker::build_db before Iso->iso_test_init" << endl;
	}
	Iso->Folding->iso_test_init(verbose_level);
	if (f_v) {
		cout << "isomorph_worker::build_db after Iso->iso_test_init" << endl;
	}

	int i;

	for (i = 0; i <= Iso->level; i++) {
		if (f_v) {
			cout << "isomorph_worker::build_db creating level database for "
					"level " << i << " / " << Iso->level << endl;
		}
		Iso->Sub->create_level_database(i, 0 /*verbose_level*/);
		if (f_v) {
			cout << "isomorph_worker::build_db creating level database for "
					"level " << i << " / " << Iso->level << " done" << endl;
		}
	}
	if (f_v) {
		cout << "isomorph_worker::build_db after isomorph_build_db" << endl;
	}
	if (f_v) {
		cout << "isomorph_worker::build_db done" << endl;
	}

}

void isomorph_worker::read_solutions(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker::read_solutions" << endl;
	}
	string *fname_array;
	int nb_files = 1;
	char str[1000];
	int *List_of_cases;


	if (Isomorph_arguments->f_list_of_cases) {

		if (f_v) {
			cout << "-list_of_cases " << Isomorph_arguments->list_of_cases_fname << endl;
		}

		string fname;
		int m, n;
		int i, a;
		orbiter_kernel_system::file_io Fio;


		Fio.int_matrix_read_csv(Isomorph_arguments->list_of_cases_fname, List_of_cases,
				m, n, verbose_level);

		nb_files = m;
		fname_array = new string[nb_files];

		for (i = 0; i < nb_files; i++) {

			fname.assign(Isomorph_arguments->solution_prefix);
			fname.append(Isomorph_arguments->base_fname);
			a = List_of_cases[i];
			sprintf(str, "_%d_%d_sol.txt", Iso->level, a);
			fname.append(str);

			fname_array[i] = fname;

		}


	}
	else {


		string fname;
		fname.assign(Isomorph_arguments->solution_prefix);
		fname.append(Isomorph_arguments->base_fname);
		sprintf(str, "_%d_sol.txt", Iso->level);
		fname.append(str);


		int i;


		nb_files = Iso->Sub->gen->nb_orbits_at_level(Iso->level); // Iso->nb_starter;
		fname_array = new string[nb_files];
		List_of_cases = NEW_int(nb_files);

		for (i = 0; i < nb_files; i++) {

			List_of_cases[i] = i;

			fname.assign(Isomorph_arguments->solution_prefix);
			fname.append(Isomorph_arguments->base_fname);
			sprintf(str, "_%d_sol.txt", i);
			fname.append(str);

			fname_array[i] = fname;

		}

	}
	if (f_v) {
		int i;
		cout << "list of cases, file names:" << endl;
		for (i = 0; i < nb_files; i++) {
			cout << i << " : " << fname_array[i] << endl;
		}
		cout << "nb_files = " << nb_files << endl;
	}

	if (f_v) {
		cout << "isomorph_worker::read_solutions before isomorph_read_solution_files" << endl;
	}

#if 0
	if (f_v) {
		cout << "isomorph_global::read_solutions "
				"before Iso.read_data_files_for_starter" << endl;
	}
	Iso->read_data_files_for_starter(Iso->level,
			Isomorph_arguments->prefix_classify,
			verbose_level);

#endif

	if (f_v) {
		cout << "isomorph_global::read_solutions "
				"before Iso->Lifting->count_solutions" << endl;
	}
	int f_get_statistics = FALSE;
	int f_has_final_test_function = FALSE;
	int *Nb_sol_per_file;

	Iso->Lifting->count_solutions(nb_files, fname_array, List_of_cases, Nb_sol_per_file,
			f_get_statistics,
			f_has_final_test_function,
			NULL /* final_test_function */, NULL /* final_test_data */,
			verbose_level);
			//
			// now we know Iso->N, the number of solutions
			// from the clique finder

	if (f_v) {
		cout << "isomorph_global::read_solutions "
				"after Iso.count_solutions Iso->Lifting->N = " << Iso->Lifting->N << endl;
	}
	//registry_dump_sorted_by_size();

	if (f_v) {
		cout << "isomorph_global::read_solutions "
				"before Iso->Lifting->read_solutions_from_clique_finder" << endl;
	}

	Iso->Lifting->read_solutions_from_clique_finder(
			nb_files, fname_array, List_of_cases, Nb_sol_per_file,
			verbose_level);

	if (f_v) {
		cout << "isomorph_global::read_solutions "
				"after Iso->read_solutions_from_clique_finder" << endl;
	}

}


void isomorph_worker::compute_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker::compute_orbits" << endl;
	}

	if (f_v) {
		cout << "isomorph_worker::compute_orbits before Iso->Lifting->init_solution" << endl;
	}
	Iso->Lifting->init_solution(verbose_level);
	if (f_v) {
		cout << "isomorph_worker::compute_orbits after Iso->Lifting->init_solution" << endl;
	}

	if (f_v) {
		cout << "isomorph_worker::compute_orbits before Iso->Lifting->orbits_of_stabilizer" << endl;
	}
	Iso->Lifting->orbits_of_stabilizer(verbose_level);
	if (f_v) {
		cout << "isomorph_worker::compute_orbits after Iso->Lifting->orbits_of_stabilizer" << endl;
	}

	if (f_v) {
		cout << "isomorph_worker::compute_orbits before Iso->Lifting->write_orbit_data" << endl;
	}
	Iso->Lifting->write_orbit_data(verbose_level);
	if (f_v) {
		cout << "isomorph_worker::compute_orbits after Iso->Lifting->write_orbit_data" << endl;
	}


	if (f_v) {
		cout << "isomorph_worker::compute_orbits done" << endl;
	}

}


void isomorph_worker::isomorph_testing(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing" << endl;
	}


	if (f_v) {
		cout << "isomorph_worker::isomorph_testing before Iso->Lifting->init_solution" << endl;
	}
	Iso->Lifting->init_solution(verbose_level - 1);
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing after Iso->Lifting->init_solution" << endl;
	}

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing before Iso->Lifting->load_table_of_solutions" << endl;
	}
	Iso->Lifting->load_table_of_solutions(verbose_level - 1);
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing after Iso->Lifting->load_table_of_solutions" << endl;
	}

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing before Iso->Lifting->read_orbit_data" << endl;
	}
	Iso->Lifting->read_orbit_data(verbose_level - 1);
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing after Iso->Lifting->read_orbit_data" << endl;
	}

	Iso->Sub->depth_completed = Iso->level /*- 2*/;

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing "
				"before Iso->gen->recreate_schreier_vectors_up_to_level" << endl;
	}
	Iso->Sub->gen->recreate_schreier_vectors_up_to_level(Iso->level - 1,
			verbose_level - 1);
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing "
				"after Iso->gen->recreate_schreier_vectors_up_to_level" << endl;
	}

	int i;

	if (f_v) {
		for (i = 0; i <= Iso->level + 1; i++) {
			cout << "gen->first_node_at_level[" << i << "]="
					<< Iso->Sub->gen->first_node_at_level(i) << endl;
		}
		//cout << "Iso.depth_completed=" << Iso.depth_completed << endl;
	}

#if 0
	cout << "Node 28:" << endl;
	Iso.gen->root[28].print_node(Iso.gen);
#endif

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing before Iso->Folding->iso_test_init" << endl;
	}
	Iso->Folding->iso_test_init(verbose_level - 1);
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing after Iso->Folding->iso_test_init" << endl;
	}

	int f_implicit_fusion = FALSE;

	//Iso.gen->f_allowed_to_show_group_elements = FALSE;

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing before Iso->Lifting->read_starter_nb_orbits" << endl;
	}
	Iso->Lifting->read_starter_nb_orbits(verbose_level); // added Oct 30, 2014
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing after Iso->Lifting->read_starter_nb_orbits" << endl;
	}


	if (f_v) {
		cout << "isomorph_worker::isomorph_testing before Iso->Folding->isomorph_testing" << endl;
	}

	std::string play_back_file_name;

	Iso->Folding->isomorph_testing(0 /*t0*/,
			FALSE /*f_play_back*/, play_back_file_name,
			f_implicit_fusion, 1 /* print_mod*/,
			verbose_level);
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing after Iso->Folding->isomorph_testing" << endl;
	}

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing before Iso->Folding->Reps->save" << endl;
	}
	Iso->Folding->Reps->save(verbose_level - 1);
	if (f_v) {
		cout << "isomorph_worker::isomorph_testing after Iso->Folding->Reps->save" << endl;
	}


	orbiter_kernel_system::file_io Fio;

	long int data1[1000];
	int id, orbit;

	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);

	string fname;

	fname.assign(Isomorph_arguments->prefix_iso);
	fname.append("orbits.txt");

	{
		ofstream fp(fname);
		fp << "# " << Iso->size << endl;
		for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {


			id = Iso->Lifting->orbit_perm[Iso->Lifting->orbit_fst[Iso->Folding->Reps->rep[orbit]]];

			Iso->Lifting->load_solution(id, data1);
			if (FALSE) {
				cout << "read representative of orbit " << orbit
						<< " (id=" << id << ")" << endl;
				Lint_vec_print(cout, data1, Iso->size);
				cout << endl;
			}

	#if 0
			for (i = 0; i < Iso.size; i++) {
				cout << setw(8) << data1[i] << ", ";
				}
			cout << endl;
	#endif
			fp << Iso->size;
			for (i = 0; i < Iso->size; i++) {
				fp << " " << data1[i];
			}
			ring_theory::longinteger_object go;

			Iso->Folding->Reps->stab[orbit]->group_order(go);
			fp << " ";
			go.print_not_scientific(fp);
			fp << endl;

			//write_set_to_file(fname, data1, Iso.size, verbose_level - 1);
		}
		fp << "-1 " << Iso->Folding->Reps->count << endl;

	}

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	Iso->Lifting->close_solution_database(verbose_level - 1);

#if 0
	Iso.print_set_function = callback_print_isomorphism_type_extend_regulus;
	Iso.print_set_data = this;
	Iso.print_isomorphism_types(verbose_level);
#endif

#if 0
	isomorph_testing(A, A2, gen,
		target_size, prefix_with_directory, prefix_iso,
		ECA->starter_size,
		f_event_file, event_file_name, print_mod,
		verbose_level);
#endif

	if (f_v) {
		cout << "isomorph_worker::isomorph_testing done" << endl;
	}
}

void isomorph_worker::isomorph_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker::isomorph_report" << endl;
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		fname.assign(Iso->prefix);
		snprintf(str, 1000, "_isomorphism_types");
		fname.append(str);
		fname.append(".tex");
		snprintf(str, 1000, "Isomorphism Types");
		title.assign(str);



		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "isomorph_worker::create_latex_report before report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "isomorph_worker::create_latex_report after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "isomorph_worker::isomorph_report done" << endl;
	}
}

void isomorph_worker::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker::report" << endl;
	}

	ost << "\\section{The Objects in Numeric Form}" << endl << endl;



	long int data[1000];

	representatives *Reps;
	int rep;
	int first;
	int id;

	Reps = Iso->Folding->Reps;

	int h, i;

	if (f_v) {
		cout << "isomorph_worker::report count = " << Reps->count << endl;
	}

	//ost << "\\clearpage" << endl << endl;
	for (h = 0; h < Reps->count; h++) {
		rep = Iso->Folding->Reps->rep[h];
		first = Iso->Lifting->orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso->Lifting->orbit_perm[first];
		Iso->Lifting->load_solution(id, data);
		for (i = 0; i < Iso->size; i++) {
			ost << data[i];
			if (i < Iso->size - 1) {
				ost << ", ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\begin{verbatim}" << endl << endl;
	ost << "int " << Iso->prefix << "_size = " << Iso->size << ";" << endl;
	ost << "int " << Iso->prefix << "_nb_reps = " << Reps->count << ";" << endl;
	ost << "int " << Iso->prefix << "_reps[] = {" << endl;
	for (h = 0; h < Reps->count; h++) {
		rep = Iso->Folding->Reps->rep[h];
		first = Iso->Lifting->orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso->Lifting->orbit_perm[first];
		Iso->Lifting->load_solution(id, data);
		ost << "\t";
		for (i = 0; i < Iso->size; i++) {
			ost << data[i];
			ost << ", ";
		}
		ost << endl;
	}
	ost << "};" << endl;
	ost << "const char *" << Iso->prefix << "_stab_order[] = {" << endl;
	for (h = 0; h < Reps->count; h++) {

		ring_theory::longinteger_object go;

		rep = Iso->Folding->Reps->rep[h];
		first = Iso->Lifting->orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso->Lifting->orbit_perm[first];
		Iso->Lifting->load_solution(id, data);
		if (Iso->Folding->Reps->stab[h]) {
			Iso->Folding->Reps->stab[h]->group_order(go);
			ost << "\"";
			go.print_not_scientific(ost);
			ost << "\"," << endl;
		}
		else {
			ost << "\"";
			ost << "1";
			ost << "\"," << endl;
		}
	}
	ost << "};" << endl;

	{
		int *stab_gens_first;
		int *stab_gens_len;
		int fst;

		stab_gens_first = NEW_int(Reps->count);
		stab_gens_len = NEW_int(Reps->count);
		fst = 0;
		ost << "int " << Iso->prefix << "_stab_gens[] = {" << endl;
		for (h = 0; h < Reps->count; h++) {
			data_structures_groups::vector_ge *gens;
			int *tl;
			int j;

			gens = NEW_OBJECT(data_structures_groups::vector_ge);
			tl = NEW_int(Iso->A_base->base_len());

			if (f_vv) {
				cout << "isomorph_global::report_data_in_source_code_inside_tex_with_selection before extract_strong_"
						"generators_in_order" << endl;
			}
			Iso->Folding->Reps->stab[h]->extract_strong_generators_in_order(
					*gens, tl, 0);

			stab_gens_first[h] = fst;
			stab_gens_len[h] = gens->len;
			fst += gens->len;

			for (j = 0; j < gens->len; j++) {
				if (f_vv) {
					cout << "isomorph_worker::isomorph_report before "
							"extract_strong_generators_in_order "
							"generator " << j
							<< " / " << gens->len << endl;
				}
				ost << "";
				Iso->A_base->element_print_for_make_element(gens->ith(j), ost);
				ost << endl;
			}

			FREE_int(tl);
			FREE_OBJECT(gens);
		}
		ost << "};" << endl;
		ost << "int " << Iso->prefix << "_stab_gens_fst[] = { ";
		for (h = 0; h < Reps->count; h++) {
			ost << stab_gens_first[h];
			if (h < Reps->count - 1) {
				ost << ", ";
			}
		}
		ost << "};" << endl;
		ost << "int " << Iso->prefix << "_stab_gens_len[] = { ";
		for (h = 0; h < Reps->count; h++) {
			ost << stab_gens_len[h];
			if (h < Reps->count - 1) {
				ost << ", ";
			}
		}
		ost << "};" << endl;
		ost << "int " << Iso->prefix << "_make_element_size = "
				<< Iso->A_base->make_element_size << ";" << endl;
	}
	ost << "\\end{verbatim}" << endl << endl;

	if (f_v) {
		cout << "isomorph_worker::report done" << endl;
	}
}



}}

