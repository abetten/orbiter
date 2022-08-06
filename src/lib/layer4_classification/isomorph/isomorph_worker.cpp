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
	Isomorph_arguments = NULL;
	Isomorph_global = NULL;
	Iso = NULL;
}

isomorph_worker::~isomorph_worker()
{
}


void isomorph_worker::init(isomorph_arguments *Isomorph_arguments,
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
		cout << "isomorph_global::build_db before Iso->init" << endl;
	}
	Iso->init(Isomorph_arguments->prefix_iso,
			A_base, A, gen, size, level,
			Isomorph_arguments->f_use_database_for_starter,
			Isomorph_arguments->f_implicit_fusion,
			verbose_level);
		// sets size, level and initializes file names
	if (f_v) {
		cout << "isomorph_global::build_db after Iso->init" << endl;
	}


	if (f_v) {
		cout << "isomorph_worker::init done" << endl;
	}
}


void isomorph_worker::execute(int verbose_level)
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

	if (f_v) {
		cout << "isomorph_worker::execute before layer2_discreta::discreta_init" << endl;
	}

	layer2_discreta::discreta_init();

	if (f_v) {
		cout << "isomorph_worker::execute after layer2_discreta::discreta_init" << endl;
	}



#if 0
	Iso->read_data_files_for_starter(Iso->level, Isomorph_arguments->prefix_classify,
			verbose_level);
#else
	if (f_v) {
		cout << "isomorph_worker::execute before Iso->compute_nb_starter" << endl;
	}
	Iso->compute_nb_starter(Iso->level, verbose_level);
	if (f_v) {
		cout << "isomorph_worker::execute after Iso->compute_nb_starter" << endl;
	}
#endif


	if (Isomorph_arguments->f_build_db) {

		if (f_v) {
			cout << "isomorph_worker::execute build_db" << endl;
		}


		if (f_v) {
			cout << "isomorph_worker::execute before Iso->iso_test_init" << endl;
		}
		Iso->iso_test_init(verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->iso_test_init" << endl;
		}

		int i;

		for (i = 0; i <= Iso->level; i++) {
			if (f_v) {
				cout << "isomorph_worker::execute creating level database for "
						"level " << i << " / " << Iso->level << endl;
			}
			Iso->create_level_database(i, verbose_level);
			if (f_v) {
				cout << "isomorph_worker::execute creating level database for "
						"level " << i << " / " << Iso->level << " done" << endl;
			}
		}



		if (f_v) {
			cout << "isomorph_worker::execute after isomorph_build_db" << endl;
		}
	}

	else if (Isomorph_arguments->f_read_solutions) {

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

			List_of_cases = NEW_int(1);
			List_of_cases[0] = 0;

			string fname;
			fname.assign(Isomorph_arguments->solution_prefix);
			fname.append(Isomorph_arguments->base_fname);
			sprintf(str, "_%d_sol.txt", Iso->level);
			fname.append(str);

			fname_array = new string[1];
			fname_array[0] = fname;
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
			cout << "isomorph_worker::execute before isomorph_read_solution_files" << endl;
		}

#if 0
		if (f_v) {
			cout << "isomorph_global::read_solution_files "
					"before Iso.read_data_files_for_starter" << endl;
		}
		Iso->read_data_files_for_starter(Iso->level,
				Isomorph_arguments->prefix_classify,
				verbose_level);

#endif

		if (f_v) {
			cout << "isomorph_global::read_solution_files "
					"before Iso.count_solutions" << endl;
		}
		int f_get_statistics = FALSE;
		int f_has_final_test_function = FALSE;
		int *Nb_sol_per_file;

		Iso->count_solutions(nb_files, fname_array, List_of_cases, Nb_sol_per_file,
				f_get_statistics,
				f_has_final_test_function,
				NULL /* final_test_function */, NULL /* final_test_data */,
				verbose_level);
				//
				// now we know Iso->N, the number of solutions
				// from the clique finder

		//registry_dump_sorted_by_size();

		Iso->read_solutions_from_clique_finder(
				nb_files, fname_array, List_of_cases, Nb_sol_per_file,
				verbose_level);


		if (f_v) {
			cout << "isomorph_worker::execute after isomorph_read_solution_files" << endl;
		}

	}
	else if (Isomorph_arguments->f_compute_orbits) {


		if (f_v) {
			cout << "isomorph_worker::execute before "
					"isomorph_compute_orbits" << endl;
		}

#if 0
		if (f_v) {
			cout << "isomorph_worker::execute before Iso->read_data_files_for_starter" << endl;
		}
		Iso->read_data_files_for_starter(Iso->level,
				Isomorph_arguments->prefix_classify,
				verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->read_data_files_for_starter" << endl;
		}
#endif

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->init_solution" << endl;
		}
		Iso->init_solution(verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->init_solution" << endl;
		}

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->orbits_of_stabilizer" << endl;
		}
		Iso->orbits_of_stabilizer(verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->orbits_of_stabilizer" << endl;
		}

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->write_orbit_data" << endl;
		}
		Iso->write_orbit_data(verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->write_orbit_data" << endl;
		}

		if (f_v) {
			cout << "isomorph_worker::execute after "
					"isomorph_compute_orbits" << endl;
		}
	}
	else if (Isomorph_arguments->f_isomorph_testing) {


		if (f_v) {
			cout << "isomorph_worker::execute before isomorph_testing" << endl;
		}

#if 0
		if (f_v) {
			cout << "isomorph_worker::execute before Iso->read_data_files_for_starter" << endl;
		}
		Iso->read_data_files_for_starter(Iso->level,
				Isomorph_arguments->prefix_classify,
				verbose_level);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->read_data_files_for_starter" << endl;
		}
#endif

		//Iso.compute_nb_starter(search_depth, verbose_level);

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->init_solution" << endl;
		}
		Iso->init_solution(verbose_level - 1);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->init_solution" << endl;
		}

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->load_table_of_solutions" << endl;
		}
		Iso->load_table_of_solutions(verbose_level - 1);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->load_table_of_solutions" << endl;
		}

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->read_orbit_data" << endl;
		}
		Iso->read_orbit_data(verbose_level - 1);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->read_orbit_data" << endl;
		}

		Iso->depth_completed = Iso->level /*- 2*/;

		if (f_v) {
			cout << "isomorph_global::isomorph_testing "
					"before Iso->gen->recreate_schreier_vectors_up_to_level" << endl;
		}
		Iso->gen->recreate_schreier_vectors_up_to_level(Iso->level - 1,
				verbose_level - 1);
		if (f_v) {
			cout << "isomorph_global::isomorph_testing "
					"after Iso->gen->recreate_schreier_vectors_up_to_level" << endl;
		}

		int i;

		if (f_v) {
			for (i = 0; i <= Iso->level + 1; i++) {
				cout << "gen->first_node_at_level[" << i << "]="
						<< Iso->gen->first_node_at_level(i) << endl;
			}
			//cout << "Iso.depth_completed=" << Iso.depth_completed << endl;
		}

	#if 0
		cout << "Node 28:" << endl;
		Iso.gen->root[28].print_node(Iso.gen);
	#endif

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->iso_test_init" << endl;
		}
		Iso->iso_test_init(verbose_level - 1);
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->iso_test_init" << endl;
		}

		int f_implicit_fusion = FALSE;

		//Iso.gen->f_allowed_to_show_group_elements = FALSE;

		if (f_v) {
			cout << "isomorph_worker::execute before Iso->read_starter_nb_orbits" << endl;
		}
		Iso->read_starter_nb_orbits(verbose_level); // added Oct 30, 2014
		if (f_v) {
			cout << "isomorph_worker::execute after Iso->read_starter_nb_orbits" << endl;
		}


		if (f_v) {
			cout << "isomorph_global::isomorph_testing before Iso->isomorph_testing" << endl;
		}

		std::string play_back_file_name;

		Iso->isomorph_testing(0 /*t0*/,
				FALSE /*f_play_back*/, play_back_file_name,
				f_implicit_fusion, 1 /* print_mod*/,
				verbose_level);
		if (f_v) {
			cout << "isomorph_global::isomorph_testing after Iso->isomorph_testing" << endl;
		}

		if (f_v) {
			cout << "isomorph_global::isomorph_testing before Iso->Reps->save" << endl;
		}
		Iso->Reps->save(verbose_level - 1);
		if (f_v) {
			cout << "isomorph_global::isomorph_testing after Iso->Reps->save" << endl;
		}


		long int data1[1000];
		int id, orbit;

		Iso->setup_and_open_solution_database(verbose_level - 1);

		string fname;

		fname.assign(Isomorph_arguments->prefix_iso);
		fname.append("orbits.txt");

		{
			ofstream fp(fname);
			fp << "# " << Iso->size << endl;
			for (orbit = 0; orbit < Iso->Reps->count; orbit++) {


				id = Iso->orbit_perm[Iso->orbit_fst[Iso->Reps->rep[orbit]]];

				Iso->load_solution(id, data1);
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

				Iso->Reps->stab[orbit]->group_order(go);
				fp << " ";
				go.print_not_scientific(fp);
				fp << endl;

				//write_set_to_file(fname, data1, Iso.size, verbose_level - 1);
			}
			fp << "-1 " << Iso->Reps->count << endl;

		}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		Iso->close_solution_database(verbose_level - 1);

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
			cout << "isomorph_worker::execute after isomorph_testing" << endl;
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



}}

