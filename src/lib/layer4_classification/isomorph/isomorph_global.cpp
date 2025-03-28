// isomorph_global.cpp
// 
// Anton Betten
// started Aug 1, 2012
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace isomorph {


#if 0
static void callback_compute_down_orbits_worker(
		isomorph *Iso, void *data, int verbose_level);
#endif

isomorph_global::isomorph_global()
{
	Record_birth();
	A_base = NULL;
	A = NULL;
	gen = NULL;
}

isomorph_global::~isomorph_global()
{
	Record_death();

}

void isomorph_global::init(
		actions::action *A_base,
		actions::action *A,
		poset_classification::poset_classification *gen,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_global::init" << endl;
	}
	isomorph_global::A_base = A_base;
	isomorph_global::A = A;
	isomorph_global::gen = gen;


	if (f_v) {
		cout << "isomorph_global::init done" << endl;
	}
}

#if 0
void isomorph_global::read_statistic_files(
	int size, std::string &prefix_classify,
	std::string &prefix, int level,
	std::string *fname, int nb_files,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = false;
	

	if (f_v) {
		cout << "isomorph_global::read_statistic_files" << endl;
		cout << "nb_files = " << nb_files << endl;
		cout << "prefix_classify = " << prefix_classify << endl;
		cout << "prefix = " << prefix << endl;
	}

	
	layer2_discreta::typed_objects::discreta_init();

	{
		isomorph Iso;
		int f_use_database_for_starter = true;
		orbiter_kernel_system::file_io Fio;


		if (f_v) {
			cout << "size = " << size << endl;
		}

		if (f_v) {
			cout << "isomorph_global::read_statistic_files "
					"before Iso.init" << endl;
		}
		Iso.init(
				prefix, A_base, A, gen, size, level,
			f_use_database_for_starter,
			f_implicit_fusion, verbose_level);
			// sets level and initializes file names



		if (f_v) {
			cout << "isomorph_global::read_statistic_files "
					"before Iso.read_data_files_for_starter" << endl;
		}
		Iso.Sub->read_data_files_for_starter(
				level,
			prefix_classify, verbose_level);




		// Row,Case_nb,Nb_sol,Nb_backtrack,Nb_col,Dt,Dt_in_sec
	
		data_structures::spreadsheet *S;
		int i, h, Case_nb, Nb_sol, Nb_backtrack, Nb_col, Dt, Dt_in_sec;
		int case_nb, nb_sol, nb_backtrack, nb_col, dt, dt_in_sec;
		int *Stats;
	
		S = NEW_OBJECTS(data_structures::spreadsheet, nb_files);
		for (i = 0; i < nb_files; i++) {
			cout << "reading file " << fname[i] << ":" << endl;
			S[i].read_spreadsheet(fname[i], 0 /* verbose_level */);
		}
	
		cout << "Allocating array Stats for " << Iso.Sub->nb_starter
				<< " starter cases" << endl;

		Stats = NEW_int(6 * Iso.Sub->nb_starter);
		Int_vec_zero(Stats, 6 * Iso.Sub->nb_starter);
		for (i = 0; i < Iso.Sub->nb_starter; i++) {
			Stats[i * 6 + 0] = -1;
		}

		cout << "Reading all the statistic files" << endl;
	
		for (h = 0; h < nb_files; h++) {
			Case_nb = S[h].find_by_column("Case_nb");
			Nb_sol = S[h].find_by_column("Nb_sol");
			Nb_backtrack = S[h].find_by_column("Nb_backtrack");
			Nb_col = S[h].find_by_column("Nb_col");
			Dt = S[h].find_by_column("Dt");
			Dt_in_sec = S[h].find_by_column("Dt_in_sec");
			for (i = 1; i < S[h].nb_rows; i++) {
				case_nb = S[h].get_lint(i, Case_nb);
				nb_sol = S[h].get_lint(i, Nb_sol);
				nb_backtrack = S[h].get_lint(i, Nb_backtrack);
				nb_col = S[h].get_lint(i, Nb_col);
				dt = S[h].get_lint(i, Dt);
				dt_in_sec = S[h].get_lint(i, Dt_in_sec);
				Stats[case_nb * 6 + 0] = 1;
				Stats[case_nb * 6 + 1] = nb_sol;
				Stats[case_nb * 6 + 2] = nb_backtrack;
				Stats[case_nb * 6 + 3] = nb_col;
				Stats[case_nb * 6 + 4] = dt;
				Stats[case_nb * 6 + 5] = dt_in_sec;
			}
		}

		cout << "Read all the statistic files" << endl;
		for (i = 0; i < Iso.Sub->nb_starter; i++) {
			if (Stats[i * 6 + 0] == -1) {
				cout << "The run is incomplete, "
						"I don't have data for case "
						<< i << " for instance" << endl;
				exit(1);
			}
		}

		cout << "The run is complete" << endl;



		cout << "The cases where solutions exist are:" << endl;
		for (i = 0; i < Iso.Sub->nb_starter; i++) {
			if (Stats[i * 6 + 1]) {
				cout << setw(5) << i << " : " << setw(5)
						<< Stats[i * 6 + 1] << " : ";

			}
		}

		int Nb_cases = 0;

		Nb_cases = 0;
		for (i = 0; i < Iso.Sub->nb_starter; i++) {
			if (Stats[i * 6 + 1]) {
				Nb_cases++;
			}
		}
	
		int *Stats_short;


		Stats_short = NEW_int(6 * Nb_cases);
		h = 0;
		for (i = 0; i < Iso.Sub->nb_starter; i++) {
			if (Stats[i * 6 + 1]) {
				Int_vec_copy(Stats + 6 * i, Stats_short + 6 * h, 6);
				Stats_short[h * 6 + 0] = i;
				h++;
			}
		}
	
		// Row,Case_nb,Nb_sol,Nb_backtrack,Nb_col,Dt,Dt_in_sec
	
		const char *Column_label[] = {
			"Case_nb",
			"Nb_sol",
			"Nb_backtrack",
			"Nb_col",
			"Dt",
			"Dt_in_sec"
			};
		string fname_collected;

		fname_collected.assign("stats_collected.csv");
		Fio.Csv_file_support->int_matrix_write_csv_with_labels(
				fname_collected,
				Stats_short, Nb_cases, 6, Column_label);
	
		cout << "Written file " << fname_collected << " of size "
				<< Fio.file_size(fname_collected) << endl;

		
		Nb_sol = 0;
		Nb_col = 0;
		Dt_in_sec = 0;
		for (i = 0; i < Iso.Sub->nb_starter; i++) {
			Nb_sol += Stats[i * 6 + 1];
			Nb_col += Stats[i * 6 + 3];
			Dt_in_sec += Stats[i * 6 + 5];
		}

		cout << "In total we have:" << endl;
		cout << "Nb_sol = " << Nb_sol << endl;
		cout << "Nb_col = " << Nb_col << endl;
		cout << "Nb_col (average) = "
				<< (double) Nb_col / Iso.Sub->nb_starter << endl;
		cout << "Dt_in_sec = " << Dt_in_sec << endl;

		

		FREE_OBJECTS(S);
#if 0
		if (f_v) {
			cout << "isomorph_read_statistic_files "
					"before Iso.count_solutions" << endl;
		}
		int f_get_statistics = false;


		Iso.count_solutions(nb_files, fname,
				f_get_statistics, verbose_level);
				//
				// now we know Iso.N, the number of solutions
				// from the clique finder

		registry_dump_sorted_by_size();

		Iso.build_up_database(nb_files, fname, verbose_level);
#endif

	}
	cout << "isomorph_global::read_statistic_files done" << endl;

	//discreta_exit();
}


void isomorph_global::init_solutions_from_memory(
	int size, std::string &prefix_classify,
	std::string &prefix_iso, int level,
	long int **Solutions, int *Nb_sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = false;

	if (f_v) {
		cout << "isomorph_global::init_solutions_from_memory" << endl;
		cout << "prefix_classify = " << prefix_classify << endl;
		cout << "prefix_iso = " << prefix_iso << endl;
	}

	
	layer2_discreta::typed_objects::discreta_init();

	{
		isomorph Iso;
		int f_use_database_for_starter = true;


		if (f_v) {
			cout << "size = " << size << endl;
		}

		if (f_v) {
			cout << "isomorph_global::init_solutions_from_memory "
					"before Iso.init" << endl;
		}
		Iso.init(
				prefix_iso, A_base, A, gen, size,
			level, f_use_database_for_starter,
			f_implicit_fusion,
			0/*verbose_level - 2*/);
		if (f_v) {
			cout << "isomorph_global::init_solutions_from_memory "
					"after Iso.init" << endl;
		}
	

	
		if (f_v) {
			cout << "isomorph_global::init_solutions_from_memory "
					"before Iso.read_data_files_for_starter" << endl;
		}
		Iso.Sub->read_data_files_for_starter(
				level,
				prefix_classify,
				0/*verbose_level - 4*/);
	
		if (f_v) {
			cout << "isomorph_global::init_solutions_from_memory "
					"after Iso.read_data_files_for_starter" << endl;
		}
		
	

		if (f_v) {
			cout << "isomorph_global::init_solutions_from_memory "
					"before Iso.init_solutions" << endl;
		}
		//int f_get_statistics = false;
		Iso.Lifting->init_solutions(
				Solutions, Nb_sol, verbose_level - 1);
				//
				// now we know Iso.N, the number of solutions
				// from the clique finder

		if (f_v) {
			cout << "isomorph_global::init_solutions_from_memory "
					"after Iso.init_solutions" << endl;
		}

	}
	if (f_v) {
		cout << "isomorph_global::init_solutions_from_memory done" << endl;
	}

	//discreta_exit();
}


void isomorph_global::classification_graph(
	int size, std::string &prefix_classify,
	std::string &prefix_iso, int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = false;
	//int t0;


	//t0 = os_ticks();


	if (f_v) {
		cout << "isomorph_global::classification_graph" << endl;
	}

	
	layer2_discreta::typed_objects::discreta_init();

	{
		isomorph Iso;
		int f_use_database_for_starter = false;
	

		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"before Iso.init" << endl;
		}
		Iso.init(
				prefix_iso,
			A_base, A, gen,
			size, level,
			f_use_database_for_starter,
			f_implicit_fusion,
			verbose_level - 1);
		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"after Iso.init" << endl;
		}



		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"before Iso.read_everything_including_classification" << endl;
		}
		Iso.read_everything_including_classification(
				prefix_classify, verbose_level);
		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"after Iso.read_everything_including_classification" << endl;
		}


	
	
	
		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"before Iso.Folding->write_classification_matrix" << endl;
		}
		Iso.Folding->write_classification_matrix(verbose_level);
		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"after Iso.Folding->write_classification_matrix" << endl;
		}

		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"before Iso.Folding->write_classification_graph" << endl;
		}
		Iso.Folding->write_classification_graph(verbose_level);
		if (f_v) {
			cout << "isomorph_global::classification_graph "
					"after Iso.Folding->write_classification_graph" << endl;
		}

		Iso.Folding->decomposition_matrix(verbose_level);



	}
	cout << "isomorph_global::classification_graph done" << endl;

	//discreta_exit();
}


void isomorph_global::identify(
	int size, std::string &prefix_classify,
	std::string &prefix_iso, int level,
	int identify_nb_files,
	std::string *fname, int *Iso_type,
	int f_save, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = false;
	long int *the_set;
	int set_size;
	string fname_transporter;




	if (f_v) {
		cout << "isomorph_global::identify" << endl;
	}
	
	
	layer2_discreta::typed_objects::discreta_init();

	{
		isomorph Iso;
		int f_use_database_for_starter = false;
		orbiter_kernel_system::file_io Fio;
	

		if (f_v) {
			cout << "isomorph_global::identify "
					"before Iso.init" << endl;
		}
		Iso.init(
				prefix_iso, A_base, A, gen,
			size, level,
			f_use_database_for_starter,
			f_implicit_fusion,
			verbose_level);



		Iso.read_everything_including_classification(
				prefix_classify, verbose_level);


		int i;

		for (i = 0; i < identify_nb_files; i++) {

			Fio.read_set_from_file(
					fname[i], the_set, set_size,
					verbose_level);
			if (f_v) {
				cout << "isomorph_global::identify "
						"read file " << fname[i] << endl;
				cout << "the_set = ";
				Lint_vec_print(cout, the_set, set_size);
				cout << endl;
			}


			fname_transporter = "transporter_" + fname[i];
	
	
			if (f_v) {
				cout << "isomorph_identify "
						"before Iso.identify" << endl;
			}
			Iso_type[i] = Iso.Folding->identify(
					the_set,
					f_implicit_fusion, verbose_level - 2);
			if (f_v) {
				cout << "isomorph_identify "
						"after Iso.identify" << endl;
			}

			if (f_save) {

	#if 0
				FILE *f2;
				f2 = fopen(fname_transporter, "wb");
				Iso.A_base->element_write_file_fp(Iso.transporter,
						f2, 0/* verbose_level*/);

				fclose(f2);
	#else
				Iso.A_base->Group_element->element_write_file(
						Iso.Folding->transporter,
						fname_transporter, 0 /* verbose_level*/);
	#endif
				cout << "isomorph_identify written file " << fname_transporter
						<< " of size "
						<< Fio.file_size(fname_transporter) << endl;
			}
	
	
			if (f_v) {
				cout << "isomorph_global::identify The set in " << fname[i]
					<< " belongs to isomorphism type " << Iso_type[i] << endl;
			}

			FREE_lint(the_set);
		}


		if (f_v) {
			cout << "isomorph_global::identify Summary:" << endl;
			for (i = 0; i < identify_nb_files; i++) {
				cout << i << " : " << fname[i] << " : " << Iso_type[i] << endl;
			}
		}


	}
	cout << "isomorph_global::identify done" << endl;
	//discreta_exit();
}

void isomorph_global::identify_table(
	int size, std::string &prefix_classify,
	std::string &prefix_iso, int level,
	int nb_rows, long int *Table, int *Iso_type,
	int verbose_level)
// Table[nb_rows * size]
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = false;
	long int *the_set;
	int set_size;




	if (f_v) {
		cout << "isomorph_global::identify_table" << endl;
	}
	
	
	layer2_discreta::typed_objects::discreta_init();

	{
		isomorph Iso;
		int f_use_database_for_starter = false;
	

		if (f_v) {
			cout << "isomorph_global::identify_table "
					"before Iso.init" << endl;
		}
		Iso.init(
				prefix_iso, A_base, A, gen,
			size, level,
			f_use_database_for_starter,
			f_implicit_fusion,
			verbose_level);
			// sets level and initializes file names




		Iso.read_everything_including_classification(
				prefix_classify, verbose_level);

		int i;

	
	
		set_size = size;
		the_set = NEW_lint(set_size);
	
		for (i = 0; i < nb_rows; i++) {
		
			Lint_vec_copy(Table + i * set_size, the_set, set_size);

			if (f_v) {
				cout << "isomorph_global::identify_table "
						"Identifying set no " << i << endl;
				cout << "the_set = ";
				Lint_vec_print(cout, the_set, set_size);
				cout << endl;
			}



			if (f_v) {
				cout << "isomorph_global::identify_table "
						"before Iso.Folding->identify" << endl;
			}
			Iso_type[i] = Iso.Folding->identify(
					the_set,
					f_implicit_fusion, verbose_level - 2);
			if (f_v) {
				cout << "isomorph_global::identify_table "
						"after Iso.Folding->identify" << endl;
			}


	
			if (f_v) {
				cout << "isomorph_global::identify_table The set no " << i
						<< " belongs to isomorphism type "
						<< Iso_type[i] << endl;
			}

		}
		FREE_lint(the_set);


		if (f_v) {
			cout << "isomorph_global::identify_table Summary:" << endl;
			for (i = 0; i < nb_rows; i++) {
				cout << i << " : " << Iso_type[i] << endl;
			}
		}


	}
	cout << "isomorph_global::identify_table done" << endl;
	//discreta_exit();
}

void isomorph_global::worker(
	int size,
	std::string &prefix_classify,
	std::string &prefix_iso,
	void (*work_callback)(
			isomorph *Iso, void *data, int verbose_level),
	void *work_data, 
	int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_global::worker" << endl;
		cout << "isomorph_global::worker size=" << size << endl;
		cout << "isomorph_global::worker level=" << level << endl;
	}

	
	layer2_discreta::typed_objects::discreta_init();

	{
		isomorph Iso;
		int f_use_database_for_starter = false;
		int f_implicit_fusion = false;

		if (f_v) {
			cout << "isomorph_global::worker "
					"before Iso.init" << endl;
		}
		Iso.init(
				prefix_iso,
			A_base, A, gen,
			size, level,
			f_use_database_for_starter,
			f_implicit_fusion,
			verbose_level);
			// sets level and initializes file names
		if (f_v) {
			cout << "isomorph_global::worker "
					"after Iso.init" << endl;
		}

		if (f_v) {
			cout << "isomorph_global::worker "
					"before Iso.read_everything_including_classification" << endl;
		}
		Iso.read_everything_including_classification(
				prefix_classify, verbose_level);
		if (f_v) {
			cout << "isomorph_global::worker "
					"after Iso.read_everything_including_classification" << endl;
		}
	



		if (f_v) {
			cout << "isomorph_global::worker "
					"before Iso.setup_and_open_solution_database" << endl;
		}
		Iso.Lifting->setup_and_open_solution_database(verbose_level - 1);
		if (f_v) {
			cout << "isomorph_global::worker "
					"after Iso.setup_and_open_solution_database" << endl;
		}
		if (f_v) {
			cout << "isomorph_global::worker "
					"before Iso.Sub->setup_and_open_level_database" << endl;
		}
		Iso.Sub->setup_and_open_level_database(verbose_level - 1);
		if (f_v) {
			cout << "isomorph_global::worker "
					"after Iso.Sub->setup_and_open_level_database" << endl;
		}

	#if 0
		Iso.print_set_function = callback_print_isomorphism_type_extend_regulus;
		Iso.print_set_data = this;
		Iso.print_isomorphism_types(f_select, select_first,
				select_len, verbose_level);
	#endif


		if (f_v) {
			cout << "isomorph_global::worker "
					"before work_callback" << endl;
		}
		(*work_callback)(&Iso, work_data, verbose_level);
		if (f_v) {
			cout << "isomorph_global::worker "
					"after work_callback" << endl;
		}
	
	
		Iso.Lifting->close_solution_database(verbose_level - 1);
		Iso.Sub->close_level_database(verbose_level - 1);

	}
	cout << "isomorph_global::worker done" << endl;

}

void isomorph_global::compute_down_orbits(
	int size,
	std::string &prefix_classify,
	std::string &prefix,
	int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_global::compute_down_orbits "
				"level = " << level << endl;
		cout << "isomorph_compute_down_orbits "
				"verbose_level = " << verbose_level << endl;
	}
	worker(
		size, prefix_classify, prefix, 
		callback_compute_down_orbits_worker,
		this,
		level, verbose_level);
	if (f_v) {
		cout << "isomorph_global::compute_down_orbits done" << endl;
	}
}
#endif

void isomorph_global::compute_down_orbits_for_isomorphism_type(
	isomorph *Iso, int orbit,
	int &cnt_orbits, int &cnt_special_orbits,
	int *&special_orbit_identify, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int id, rep, first; //, c;
	long int data[1000];
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "isomorph_global::compute_down_orbits_for_isomorphism_type "
				"orbit=" << orbit << endl;
	}

	cnt_orbits = 0;
	cnt_special_orbits = 0;

	rep = Iso->Folding->Reps->rep[orbit];
	first = Iso->Lifting->flag_orbit_solution_first[rep];
	//c = Iso->starter_number[first];
	id = Iso->Lifting->orbit_perm[first];
	Iso->Lifting->load_solution(id, data, verbose_level - 1);

	

	groups::sims *Stab;
	groups::strong_generators *Strong_gens;

	Stab = Iso->Folding->Reps->stab[orbit];

	if (f_vv) {
		cout << "isomorph_global::compute_down_orbits_for_isomorphism_type "
				"computing induced action on the set (in data)" << endl;
	}


	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(Stab, verbose_level - 2);
	

	Iso->Folding->induced_action_on_set(
			Stab, data, 0 /*verbose_level*/);
		
	if (f_vv) {
		cout << "data after induced_action_on_set:" << endl;
		Lint_vec_print(cout, data, Iso->size);
		cout << endl;
	}
		
	algebra::ring_theory::longinteger_object go1;
			
	Iso->Folding->AA->group_order(go1);

	if (f_vv) {
		cout << "action " << Iso->Folding->AA->label << " computed, "
				"group order is " << go1 << endl;

		cout << "Order of the group that is induced on the object is ";
		cout << "$";
		go1.print_not_scientific(cout);
		cout << "$\\\\" << endl;
	}

	if (false /*go1.is_one()*/) {
		cnt_orbits = Combi.int_n_choose_k(Iso->size, Iso->level);
		cnt_special_orbits = 1;
	}
	else {
		long int *orbit_reps;
		int nb_orbits;

		if (f_vv) {
			cout << "isomorph_global::compute_down_orbits_for_isomorphism_type "
					"orbit=" << orbit << " / " << Iso->Folding->Reps->count
					<< " computing orbits on subsets" << endl;
		}
		poset_classification::poset_classification_control *Control;
		poset_classification::poset_with_group_action *Poset;
		Control = NEW_OBJECT(poset_classification::poset_classification_control);
		Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
		Poset->init_subset_lattice(
				Iso->A_base, Iso->Folding->AA, Strong_gens,
				verbose_level);
		Poset->orbits_on_k_sets(Control,
			Iso->level, orbit_reps, nb_orbits, verbose_level - 5);
		FREE_OBJECT(Poset);
		FREE_OBJECT(Control);
		if (f_vv) {
			cout << "isomorph_global::compute_down_orbits_for_isomorphism_type "
					"orbit=" << orbit << " / " << Iso->Folding->Reps->count
					<< " computing orbits on subsets done" << endl;
		}

		if (f_vvv) {
			cout << "Orbit reps: nb_orbits=" << nb_orbits << endl;
			Lint_matrix_print(orbit_reps, nb_orbits, Iso->level);
		}

		if (f_vv) {
			cout << "Number of orbits on $" << Iso->level << "$-sets is "
					<< nb_orbits << ".\\\\" << endl;
		}

		long int *rearranged_set;
		int *transporter;
		int u;
		int case_nb;
		int f_implicit_fusion = false;
		int idx;
		
		rearranged_set = NEW_lint(Iso->size);
		transporter = NEW_int(Iso->A_base->elt_size_in_int);

		cnt_orbits = nb_orbits;
		cnt_special_orbits = 0;
		special_orbit_identify = NEW_int(nb_orbits);
		for (u = 0; u < nb_orbits; u++) {

			if (f_vv) {
				cout << "iso type " << orbit << " / " << Iso->Folding->Reps->count
						<< " down_orbit " << u << " / "
						<< nb_orbits << ":" << endl;
				Lint_vec_print(cout, orbit_reps + u * Iso->level, Iso->level);
				cout << endl;
			}



			Sorting.rearrange_subset_lint_all(
					Iso->size, Iso->level,
					data, orbit_reps + u * Iso->level, rearranged_set,
				0/*verbose_level - 3*/);


			//int_vec_print(cout, rearranged_set, Iso.size);
			//cout << endl;
			int f_failure_to_find_point, f_found;

			Iso->A_base->Group_element->element_one(
					transporter, 0);
			case_nb = Iso->Folding->trace_set(
					rearranged_set, transporter,
				f_implicit_fusion, f_failure_to_find_point,
				0 /*verbose_level - 2*/);

			//cout << "f_failure_to_find_point="
			// << f_failure_to_find_point << endl;
			//cout << "case_nb=" << case_nb << endl;
			if (f_failure_to_find_point) {
				cout << "isomorph_global::compute_down_orbits_for_isomorphism_type "
						"f_failure_to_find_point" << endl;
				exit(1);
			}


			Iso->Sub->find_extension_easy_new(
					rearranged_set,
					case_nb, idx, f_found, 0 /* verbose_level */);


#if 0
			f_found = Iso.identify_solution_relaxed(prefix, transporter, 
				f_implicit_fusion, orbit_no0,
				f_failure_to_find_point, 3 /*verbose_level*/);
#endif

			//cout << "f_found=" << f_found << endl;
			if (!f_found) {
				if (f_vv) {
					cout << "isomorph_global::compute_down_orbits_for_isomorphism_type "
							"not found" << endl;
				}
				continue;
			}
			else {
				if (f_vv) {
					cout << "iso type " << orbit << " / " << Iso->Folding->Reps->count
							<< " down orbit " << u << " / " << nb_orbits
							<< " leads to orbit " << idx << endl;
				}
			}
			special_orbit_identify[cnt_special_orbits] = idx;
			cnt_special_orbits++;
		} // next u
		if (f_v) {
			cout << "Number of special orbits on $" << Iso->level
					<< "$-sets is " << cnt_special_orbits
					<< ".\\\\" << endl;
		}

		int *soi;
		int i;

		soi = NEW_int(cnt_special_orbits);
		for (i = 0; i < cnt_special_orbits; i++) {
			soi[i] = special_orbit_identify[i];
		}
		FREE_int(special_orbit_identify);
		special_orbit_identify = soi;


		FREE_lint(rearranged_set);
		FREE_int(transporter);
		FREE_lint(orbit_reps);
	}
	FREE_OBJECT(Strong_gens);

	if (f_v) {
		cout << "isomorph_global::compute_down_orbits_for_isomorphism_type "
				"done" << endl;
	}
}

void isomorph_global::report_data_in_source_code_inside_tex(
		isomorph &Iso,
		std::string &prefix,
		std::string &label_of_structure_plural, std::ostream &f,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int *selection;
	int selection_size;
	int i;

	if (f_v) {
		cout << "isomorph_global::report_data_in_source_code_inside_tex" << endl;
	}
	selection_size = Iso.Folding->Reps->count;
	selection = NEW_int(selection_size);
	for (i = 0; i < selection_size; i++) {
		selection[i] = i;
	}
	report_data_in_source_code_inside_tex_with_selection(
		Iso, prefix,
		label_of_structure_plural, f, 
		selection_size, selection, verbose_level);
	FREE_int(selection);
}


void isomorph_global::report_data_in_source_code_inside_tex_with_selection(
		isomorph &Iso, std::string &prefix,
		std::string &label_of_structure_plural, std::ostream &fp,
		int selection_size, int *selection,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int h, rep, first, /*c,*/ id, i, s;
	long int data[1000];

	if (f_v) {
		cout << "isomorph_global::report_data_in_source_code_inside_tex_with_selection" << endl;
	}

	fp << "\\section{The " << label_of_structure_plural
			<< " in Numeric Form}" << endl << endl;

	//fp << "\\clearpage" << endl << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];
		rep = Iso.Folding->Reps->rep[h];
		first = Iso.Lifting->flag_orbit_solution_first[rep];
		//c = Iso.starter_number[first];
		id = Iso.Lifting->orbit_perm[first];
		Iso.Lifting->load_solution(id, data, verbose_level - 1);
		for (i = 0; i < Iso.size; i++) {
			fp << data[i];
			if (i < Iso.size - 1) {
				fp << ", ";
			}
		}
		fp << "\\\\" << endl;
	}
	fp << "\\begin{verbatim}" << endl << endl;
	export_source_code_with_selection(
			Iso, prefix,
			fp,
			selection_size, selection,
			verbose_level);
	fp << "\\end{verbatim}" << endl << endl;
}


void isomorph_global::export_source_code_with_selection(
		isomorph &Iso, std::string &prefix,
		std::ostream &fp,
		int selection_size, int *selection,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int h, rep, first, /*c,*/ id, i, s;
	long int data[1000];

	if (f_v) {
		cout << "isomorph_global::export_source_code_with_selection" << endl;
	}

#if 0
	fp << "\\section{The " << label_of_structure_plural
			<< " in Numeric Form}" << endl << endl;

	//fp << "\\clearpage" << endl << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];
		rep = Iso.Folding->Reps->rep[h];
		first = Iso.Lifting->orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.Lifting->orbit_perm[first];
		Iso.Lifting->load_solution(id, data, verbose_level - 1);
		for (i = 0; i < Iso.size; i++) {
			fp << data[i];
			if (i < Iso.size - 1) {
				fp << ", ";
			}
		}
		fp << "\\\\" << endl;
	}
#endif

	fp << "int " << prefix << "_size = " << Iso.size << ";" << endl;
	fp << "int " << prefix << "_nb_reps = " << selection_size << ";" << endl;
	fp << "long int " << prefix << "_reps[] = {" << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];
		rep = Iso.Folding->Reps->rep[h];
		first = Iso.Lifting->flag_orbit_solution_first[rep];
		//c = Iso.starter_number[first];
		id = Iso.Lifting->orbit_perm[first];
		Iso.Lifting->load_solution(id, data, verbose_level - 1);
		fp << "\t";
		for (i = 0; i < Iso.size; i++) {
			fp << data[i];
			fp << ", ";
		}
		fp << endl;
	}
	fp << "};" << endl;
	fp << "const char *" << prefix << "_stab_order[] = {" << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];

		algebra::ring_theory::longinteger_object go;

		rep = Iso.Folding->Reps->rep[h];
		first = Iso.Lifting->flag_orbit_solution_first[rep];
		//c = Iso.starter_number[first];
		id = Iso.Lifting->orbit_perm[first];
		Iso.Lifting->load_solution(id, data, verbose_level - 1);
		if (Iso.Folding->Reps->stab[h]) {
			Iso.Folding->Reps->stab[h]->group_order(go);
			fp << "\"";
			go.print_not_scientific(fp);
			fp << "\"," << endl;
		}
		else {
			fp << "\"";
			fp << "1";
			fp << "\"," << endl;
		}
	}
	fp << "};" << endl;

	{
		int *stab_gens_first;
		int *stab_gens_len;
		int fst;

		stab_gens_first = NEW_int(selection_size);
		stab_gens_len = NEW_int(selection_size);
		fst = 0;
		fp << "int " << prefix << "_stab_gens[] = {" << endl;
		for (s = 0; s < selection_size; s++) {
			h = selection[s];
			data_structures_groups::vector_ge *gens;
			int *tl;
			int j;

			gens = NEW_OBJECT(data_structures_groups::vector_ge);
			tl = NEW_int(Iso.A_base->base_len());

			if (f_vv) {
				cout << "isomorph_global::export_source_code_with_selection "
						"before extract_strong_generators_in_order" << endl;
			}
			Iso.Folding->Reps->stab[h]->extract_strong_generators_in_order(
					*gens, tl, 0);

			stab_gens_first[s] = fst;
			stab_gens_len[s] = gens->len;
			fst += gens->len;

			for (j = 0; j < gens->len; j++) {
				if (f_vv) {
					cout << "isomorph_global::export_source_code_with_selection "
							"before extract_strong_generators_in_order "
							"generator " << j
							<< " / " << gens->len << endl;
				}
				fp << "";
				Iso.A_base->Group_element->element_print_for_make_element(
						gens->ith(j), fp);
				fp << endl;
			}

			FREE_int(tl);
			FREE_OBJECT(gens);
		}
		fp << "};" << endl;
		fp << "int " << prefix << "_stab_gens_fst[] = { ";
		for (s = 0; s < selection_size; s++) {
			fp << stab_gens_first[s];
			if (s < selection_size - 1) {
				fp << ", ";
			}
		}
		fp << "};" << endl;
		fp << "int " << prefix << "_stab_gens_len[] = { ";
		for (s = 0; s < selection_size; s++) {
			fp << stab_gens_len[s];
			if (s < selection_size - 1) {
				fp << ", ";
			}
		}
		fp << "};" << endl;
		fp << "int " << prefix << "_make_element_size = "
				<< Iso.A_base->make_element_size << ";" << endl;
	}
	if (f_v) {
		cout << "isomorph_global::export_source_code_with_selection done" << endl;
	}
}

#if 0
static void callback_compute_down_orbits_worker(
		isomorph *Iso, void *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int orbit;
	int *Nb_orbits;
	int nb_orbits = 0;
	int nb_special_orbits = 0;
	int **Down_orbit_identify;
	int *Down_identify;
	int h, i, idx;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "callback_compute_down_orbits_worker" << endl;
	}
	isomorph_global *IG = (isomorph_global *) data;

	//f_memory_debug = true;
	Nb_orbits = NEW_int(Iso->Folding->Reps->count * 2);
	Down_orbit_identify = NEW_pint(Iso->Folding->Reps->count);
	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {

		int cnt_orbits, cnt_special_orbits;
		int *special_orbit_identify;

		IG->compute_down_orbits_for_isomorphism_type(
				Iso, orbit, cnt_orbits, cnt_special_orbits,
				special_orbit_identify, verbose_level - 1);

		if (f_vv) {
			cout << "callback_compute_down_orbits_worker orbit "
					<< orbit << " / " << Iso->Folding->Reps->count
					<< " cnt_orbits=" << cnt_orbits
					<< " cnt_special_orbits=" << cnt_special_orbits << endl;
		}
		Nb_orbits[orbit * 2 + 0] = cnt_orbits;
		Nb_orbits[orbit * 2 + 1] = cnt_special_orbits;
		Down_orbit_identify[orbit] = special_orbit_identify;

		nb_orbits += cnt_orbits;
		nb_special_orbits += cnt_special_orbits;

#if 0
		if (orbit && ((orbit % 100) == 0)) {
			registry_dump_sorted();
		}
#endif

	}

	{
		string fname;

		fname.assign("Nb_down_orbits.csv");
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Nb_orbits, Iso->Folding->Reps->count, 2);
	}

	if (f_v) {
		cout << "callback_compute_down_orbits_worker" << endl;
		cout << "nb_orbits=" << nb_orbits << endl;
		cout << "nb_special_orbits=" << nb_special_orbits << endl;
	}

	Down_identify = NEW_int(nb_special_orbits * 3);
	h = 0;
	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {
		for (i = 0; i < Nb_orbits[orbit * 2 + 1]; i++) {
			idx = Down_orbit_identify[orbit][i];
			Down_identify[h * 3 + 0] = orbit;
			Down_identify[h * 3 + 1] = i;
			Down_identify[h * 3 + 2] = idx;
			h++;
		}
	}

	{
		string fname;

		fname.assign("Down_identify.csv");
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Down_identify, nb_special_orbits, 3);
	}

	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {
		FREE_int(Down_orbit_identify[orbit]);
	}
	FREE_pint(Down_orbit_identify);
	FREE_int(Down_identify);
	FREE_int(Nb_orbits);
	if (f_v) {
		cout << "callback_compute_down_orbits_worker done" << endl;
	}
}
#endif


}}}




