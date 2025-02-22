// blt_set_classify.cpp
// 
// Anton Betten
//
// started 8/13/2006
//
// moved to apps/blt  from blt.cpp 5/24/09
// moved to src/top_level/geometry  from apps/blt.cpp Jan 8, 2019
//
//
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


// global functions:
static void blt_set_classify_print(
		std::ostream &ost, int len, long int *S, void *data);
static void blt_set_classify_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);

blt_set_classify::blt_set_classify()
{
	Record_birth();
	OA = NULL;
	Blt_set_domain_with_action = NULL;
	Blt_set_domain = NULL;
	//LG = NULL;
	A = NULL;
	starter_size = 0;
	Strong_gens = NULL;
	f_semilinear = false;
	q = 0;
	Control = NULL;
	Poset = NULL;
	gen = NULL;
	nb_points_on_quadric = 0;
	target_size = 0;
	Worker = NULL;
	//null();
}

blt_set_classify::~blt_set_classify()
{
	Record_death();
	int f_v = false;

	if (f_v) {
		cout << "blt_set_classify::~blt_set_classify" << endl;
	}
	if (Control) {
		FREE_OBJECT(Control);
		Control = NULL;
	}
	if (Poset) {
		FREE_OBJECT(Poset);
		Poset = NULL;
	}
	if (gen) {
		FREE_OBJECT(gen);
		gen = NULL;
	}
	if (f_v) {
		cout << "blt_set_classify::~blt_set_classify done" << endl;
	}
}

void blt_set_classify::init(
		blt_set_classify_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify::init" << endl;
		cout << "blt_set_classify::init "
				"verbose_level = " << verbose_level << endl;
	}


	if (!Descr->f_orthogonal_space) {
		cout << "blt_set_classify::init please specify an orthogonal space" << endl;
		exit(1);
	}

	if (!Descr->f_starter_size) {
		cout << "please use option -starter_size <s>" << endl;
		exit(1);
	}


	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = Get_orthogonal_space(Descr->orthogonal_space_label);

	if (f_v) {
		cout << "blt_set_classify::init before init_basic" << endl;
	}
	init_basic(
			OA,
			OA->A,
			OA->A->Strong_gens,
			Descr->starter_size,
			verbose_level);
	if (f_v) {
		cout << "blt_set_classify::init before init_basic" << endl;
	}


	if (f_v) {
		cout << "blt_set_classify::init done" << endl;
	}
}


void blt_set_classify::init_basic(
		orthogonal_space_with_action *OA,
		actions::action *A,
		groups::strong_generators *Strong_gens,
		int starter_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify::init_basic" << endl;
		cout << "blt_set_classify::init_basic "
				"verbose_level = " << verbose_level << endl;
	}

	induced_actions::action_on_orthogonal *AO;
	geometry::orthogonal_geometry::orthogonal *O;
	int f_semilinear;


	blt_set_classify::OA = OA;
	blt_set_classify::starter_size = starter_size;
	blt_set_classify::Strong_gens = Strong_gens;
	blt_set_classify::A = A;


	Blt_set_domain_with_action = OA->Blt_set_domain_with_action;


	if (A->type_G != action_on_orthogonal_t) {
		cout << "the group must be of orthogonal type" << endl;
		exit(1);
	}
	AO = A->G.AO;
	O = AO->O;
	q = O->F->q;
	f_semilinear = A->subaction->is_semilinear_matrix_group();

	if (f_v) {
		cout << "blt_set_classify::init_basic" << endl;
		cout << "blt_set_classify::init_basic "
				"f_semilinear = " << f_semilinear << endl;
	}


#if 0
	if (f_v) {
		cout << "blt_set_classify::init_basic "
				"before lex_least_base_in_place" << endl;
	}
	A->lex_least_base_in_place(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "blt_set_classify::init_basic "
				"after lex_least_base_in_place" << endl;
	}
	if (f_v) {
		cout << "blt_set_classify::init_group "
				"computing lex least base done" << endl;
		cout << "blt_set::init_group base: ";
		lint_vec_print(cout, A->get_base(), A->base_len());
		cout << endl;
	}
#endif
	


	Blt_set_domain = OA->Blt_set_domain_with_action->Blt_set_domain;

	nb_points_on_quadric = Blt_set_domain->nb_points_on_quadric;
	target_size = Blt_set_domain->target_size;


	if (f_v) {
		cout << "blt_set_classify::init_basic q=" << q
				<< " target_size = " << target_size
				<< " nb_points_on_quadric = " << nb_points_on_quadric
				<< endl;
	}
	
	if (f_v) {
		cout << "blt_set_classify::init_basic finished" << endl;
	}
}


void blt_set_classify::compute_starter(
		poset_classification::poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify::compute_starter" << endl;
		cout << "blt_set_classify::compute_starter "
				"verbose_level = " << verbose_level << endl;
	}


	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(
			A, A,
			Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "blt_set_classify::compute_starter before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset->add_testing_without_group(
				blt_set_classify_early_test_func_callback,
				this /* void *data */,
				verbose_level);
	if (f_v) {
		cout << "blt_set_classify::compute_starter after "
				"Poset->add_testing_without_group" << endl;
	}

	Poset->f_print_function = false;
	Poset->print_function = blt_set_classify_print;
	Poset->print_function_data = (void *) this;

	gen = NEW_OBJECT(poset_classification::poset_classification);


	if (f_v) {
		cout << "blt_set_classify::compute_starter "
				"before gen->compute_orbits_on_subsets" << endl;
	}
	gen->compute_orbits_on_subsets(
			starter_size/* target_depth */,
			Control,
			Poset,
			verbose_level);

	if (f_v) {
		cout << "blt_set_classify::compute_starter "
				"after gen->compute_orbits_on_subsets" << endl;
	}


	if (f_v) {
		cout << "blt_set_classify::compute_starter finished" << endl;
	}
}


void blt_set_classify::do_poset_classification_activity(
		std::string &activity_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify::do_poset_classification_activity" << endl;
	}

	poset_classification::poset_classification_activity_description *Activity_description;

	Activity_description =
			Get_object_of_type_poset_classification_activity(activity_label);


	poset_classification::poset_classification_activity Activity;

	if (f_v) {
		cout << "blt_set_classify::do_poset_classification_activity "
				"before Activity.init" << endl;
	}
	Activity.init(
			Activity_description,
			gen,
			starter_size /* actual_size */,
			verbose_level);
	if (f_v) {
		cout << "blt_set_classify::do_poset_classification_activity "
				"after Activity.init" << endl;
	}

	if (f_v) {
		cout << "blt_set_classify::do_poset_classification_activity "
				"before Activity.perform_work" << endl;
	}
	Activity.perform_work(
			verbose_level);
	if (f_v) {
		cout << "blt_set_classify::do_poset_classification_activity "
				"after Activity.perform_work" << endl;
	}



	if (f_v) {
		cout << "blt_set_classify::do_poset_classification_activity done" << endl;
	}
}


#if 0
void blt_set_classify::init_group(int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_basis = true;

	if (f_v) {
		cout << "blt_set_classify::init_group" << endl;
	}

	if (f_vv) {
		cout << "blt_set_classify::init_group "
				"before A->init_orthogonal_group" << endl;
	}
	A = NEW_OBJECT(action);

	A->init_orthogonal_group_with_O(Blt_set_domain->O,
		true /* f_on_points */, 
		false /* f_on_lines */, 
		false /* f_on_points_and_lines */, 
		f_semilinear, f_basis, 0 /* verbose_level - 1*/);
	degree = A->degree;
	if (f_vv) {
		cout << "blt_set_classify::init_group "
				"after A->init_orthogonal_group" << endl;
		cout << "blt_set::init_group "
				"degree = " << degree << endl;
	}
	
	if (f_vv) {
		cout << "blt_set_classify::init_group "
				"computing lex least base" << endl;
	}
	A->lex_least_base_in_place(0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "blt_set_classify::init_group "
				"computing lex least base done" << endl;
		cout << "blt_set::init_group base: ";
		lint_vec_print(cout, A->get_base(), A->base_len());
		cout << endl;
	}
	
	//action_on_orthogonal *AO;

	//AO = A->G.AO;
	//O = AO->O;

	if (f_v) {
		cout << "blt_set_classify::init_group "
				"degree = " << A->degree << endl;
	}
		
	//init_orthogonal_hash(verbose_level);

#if 0
	if (A->degree < 200) {
		if (f_v) {
			cout << "blt_set_classify::init_group "
					"before test_Orthogonal" << endl;
			}
		test_Orthogonal(epsilon, n - 1, q);
	}
#endif
	//A->Sims->print_all_group_elements();

	if (false) {
		cout << "blt_set_classify::init_group before "
				"A->Sims->print_all_transversal_elements" << endl;
		A->Sims->print_all_transversal_elements();
		cout << "blt_set_classify::init_group after "
				"A->Sims->print_all_transversal_elements" << endl;
	}


	if (false /*f_vv*/) {
		Blt_set_domain->O->F->print();
	}


	
	if (f_v) {
		cout << "blt_set_classify::init_group "
				"allocating Pts and Candidates" << endl;
	}
	
	if (f_v) {
		cout << "blt_set_classify::init_group finished" << endl;
	}
}

void blt_set_classify::init_orthogonal_hash(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify::init_orthogonal_hash" << endl;
	}

	// ToDo:
	//Blt_set_domain->O->F->init_hash_table_parabolic(4, 0/*verbose_level*/);

	if (f_v) {
		cout << "blt_set_classify::init_orthogonal_hash finished" << endl;
	}
}
#endif



void blt_set_classify::create_graphs(
	int orbit_at_level_r, int orbit_at_level_m, 
	int level_of_candidates_file, 
	int f_lexorder_test, int f_eliminate_graphs_if_possible, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	other::orbiter_kernel_system::os_interface Os;


	if (f_v) {
		cout << "blt_set_classify::create_graphs" << endl;
		cout << "blt_set_classify::create_graphs "
				"starter_size = " << starter_size << endl;
		cout << "blt_set_classify::create_graphs "
				"f_lexorder_test=" << f_lexorder_test << endl;
	}


	//f_memory_debug = true;


	string fname;
	string fname_list_of_cases;
	string fname_time;
	int orbit;
	int nb_orbits;
	long int *list_of_cases;
	int nb_of_cases;

	long int *Time;
	int time_idx;
	other::orbiter_kernel_system::file_io Fio;

	string str;


	fname = Blt_set_domain->prefix + "_lvl_" + std::to_string(starter_size);

	str = "_" + std::to_string(starter_size) + "_" + std::to_string(orbit_at_level_r) + "_" + std::to_string(orbit_at_level_m);

	fname_list_of_cases = Blt_set_domain->prefix + "_list_of_cases" + str + ".csv";


	fname_time = Blt_set_domain->prefix + "_time" + str + ".csv";


	if (f_v) {
		cout << "blt_set_classify::create_graphs "
				"counting number of starter in file " << fname << endl;
	}
	nb_orbits = Fio.count_number_of_orbits_in_file(fname, 0);
	if (f_v) {
		cout << "blt_set_classify::create_graphs There are "
				<< nb_orbits << " starters" << endl;
	}
	if (nb_orbits < 0) {
		cout << "Something is wrong, nb_orbits is negative" << endl;
		exit(1);
	}


	Time = NEW_lint(nb_orbits * 2);
	Lint_vec_zero(Time, nb_orbits * 2);
	time_idx = 0;

	nb_of_cases = 0;
	list_of_cases = NEW_lint(nb_orbits);
	for (orbit = 0; orbit < nb_orbits; orbit++) {
		if ((orbit % orbit_at_level_m) != orbit_at_level_r) {
			continue;
		}
		if (f_v) {
			cout << "blt_set_classify::create_graphs "
					"creating graph associated "
					"with orbit " << orbit << " / " << nb_orbits
					<< ":" << endl;
		}

		
		combinatorics::graph_theory::colored_graph *CG = NULL;
		int nb_vertices = -1;

		int t0 = Os.os_ticks();
		
		if (f_v3) {
			cout << "blt_set_classify::create_graphs "
					"creating graph associated "
					"with orbit " << orbit << " / " << nb_orbits
					<< ": before create_graph" << endl;
		}
		if (create_graph(
				orbit, level_of_candidates_file,
			f_lexorder_test, f_eliminate_graphs_if_possible, 
			nb_vertices,
			CG,  
			verbose_level - 2)) {
			list_of_cases[nb_of_cases++] = orbit;

			string fname;

			fname = CG->fname_base + ".bin";
			CG->save(fname, verbose_level - 2);
			
			nb_vertices = CG->nb_points;
		}
		if (f_v3) {
			cout << "blt_set_classify::create_graphs "
					"creating graph associated "
					"with orbit " << orbit << " / " << nb_orbits
					<< ": after create_graph" << endl;
		}

		if (CG) {
			FREE_OBJECT(CG);
		}

		int t1 = Os.os_ticks();

		Time[time_idx * 2 + 0] = orbit;
		Time[time_idx * 2 + 1] = t1 - t0;
		time_idx++;
		
		if (f_vv) {
			if (nb_vertices >= 0) {
				cout << "blt_set_classify::create_graphs "
						"creating graph "
						"associated with orbit " << orbit << " / "
						<< nb_orbits << " with " << nb_vertices
						<< " vertices created" << endl;
			}
			else {
				cout << "blt_set_classify::create_graphs "
						"creating graph "
						"associated with orbit " << orbit << " / "
						<< nb_orbits << " is ruled out" << endl;
			}
		}
	}

	if (f_v) {
		cout << "blt_set_classify::create_graphs "
				"writing file "
				<< fname_time << endl;
	}
	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_time, Time, time_idx, 2);
	if (f_v) {
		cout << "blt_set_classify::create_graphs "
				"Written file "
				<< fname_time << " of size "
				<< Fio.file_size(fname_time) << endl;
	}

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_list_of_cases,
			list_of_cases, nb_of_cases, 1);
	if (f_v) {
		cout << "blt_set_classify::create_graphs "
				"Written file "
				<< fname_list_of_cases << " of size "
				<< Fio.file_size(fname_list_of_cases) << endl;
	}

	FREE_lint(Time);
	FREE_lint(list_of_cases);

	//registry_dump_sorted();
}

void blt_set_classify::create_graphs_list_of_cases(
	std::string &case_label,
	std::string &list_of_cases_text,
	int level_of_candidates_file, 
	int f_lexorder_test, int f_eliminate_graphs_if_possible, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "blt_set_classify::create_graphs_list_of_cases" << endl;
		cout << "blt_set_classify::create_graphs_list_of_cases "
				"case_label = " << case_label << endl;
	}

	
	//f_memory_debug = true;

	int *list_of_cases = NULL;
	int nb_of_cases;


	Int_vec_scan(list_of_cases_text, list_of_cases, nb_of_cases);
	if (f_v) {
		cout << "blt_set_classify::create_graphs_list_of_cases "
				"nb_of_cases = " << nb_of_cases << endl;
		cout << "blt_set_classify::create_graphs_list_of_cases "
				"starter_size = " << starter_size << endl;
		cout << "blt_set_classify::create_graphs_list_of_cases "
				"f_lexorder_test=" << f_lexorder_test << endl;
	}

	string fname;
	string fname_list_of_cases;
	int orbit;
	int nb_orbits;
	long int *list_of_cases_created;
	int nb_of_cases_created;
	int c;
	other::orbiter_kernel_system::file_io Fio;


	fname = Blt_set_domain->prefix + "_lvl_" + std::to_string(starter_size);


	fname_list_of_cases = Blt_set_domain->prefix + "_list_of_cases_" + case_label + "_lvl_" + std::to_string(starter_size);


	nb_orbits = Fio.count_number_of_orbits_in_file(fname, 0);
	if (f_v) {
		cout << "blt_set_classify::create_graphs_list_of_cases "
				"There are " << nb_orbits << " starters" << endl;
	}
	if (nb_orbits < 0) {
		cout << "Something is wrong, nb_orbits is negative" << endl;
		cout << "fname = " << fname << endl;
		exit(1);
	}


	nb_of_cases_created = 0;
	list_of_cases_created = NEW_lint(nb_orbits);
	for (c = 0; c < nb_of_cases; c++) {
		orbit = list_of_cases[c];
		if (f_v3) {
			cout << "blt_set_classify::create_graphs_list_of_cases case "
					<< c << " / " << nb_of_cases << " creating graph "
							"associated with orbit " << orbit << " / "
							<< nb_orbits << ":" << endl;
		}

		
		combinatorics::graph_theory::colored_graph *CG = NULL;
		int nb_vertices = -1;


		if (create_graph(
				orbit, level_of_candidates_file,
			f_lexorder_test, f_eliminate_graphs_if_possible, 
			nb_vertices,
			CG,  
			verbose_level - 2)) {
			list_of_cases_created[nb_of_cases_created++] = orbit;

			string fname;

			fname = Blt_set_domain->prefix + CG->fname_base + ".bin";

			CG->save(fname, verbose_level - 2);
			
			nb_vertices = CG->nb_points;
			//delete CG;
		}

		if (CG) {
			cout << "before FREE_OBJECT(CG)" << endl;
			FREE_OBJECT(CG);
			cout << "after FREE_OBJECT(CG)" << endl;
		}
		if (f_vv) {
			if (nb_vertices >= 0) {
				cout << "blt_set_classify::create_graphs_list_of_cases "
						"case " << c << " / " << nb_of_cases
						<< " creating graph associated with orbit "
						<< orbit << " / " << nb_orbits << " with "
						<< nb_vertices << " vertices created" << endl;
			}
			else {
				cout << "blt_set_classify::create_graphs_list_of_cases "
						"case " << c << " / " << nb_of_cases
						<< " creating graph associated with orbit "
						<< orbit << " / " << nb_orbits
						<< " is ruled out" << endl;
			}
		}
	}

	Fio.write_set_to_file(
			fname_list_of_cases,
			list_of_cases_created, nb_of_cases_created,
			0 /*verbose_level */);
	if (f_v) {
		cout << "blt_set_classify::create_graphs_list_of_cases "
				"Written file " << fname_list_of_cases
				<< " of size " << Fio.file_size(fname_list_of_cases) << endl;
	}
	if (f_v) {
		cout << "blt_set_classify::create_graphs_list_of_cases "
				"we created " << nb_of_cases_created
				<< " / " << nb_of_cases << " cases" << endl;
	}

	FREE_lint(list_of_cases_created);

	//registry_dump_sorted();
}


int blt_set_classify::create_graph(
	int orbit_at_level, int level_of_candidates_file, 
	int f_lexorder_test, int f_eliminate_graphs_if_possible, 
	int &nb_vertices,
	combinatorics::graph_theory::colored_graph *&CG,
	int verbose_level)
// returns true if a graph was written, false otherwise
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "blt_set_classify::create_graph" << endl;
		cout << "blt_set_classify::create_graph "
				"f_lexorder_test=" << f_lexorder_test << endl;
		cout << "blt_set_classify::create_graph "
				"orbit_at_level=" << orbit_at_level << endl;
		cout << "blt_set_classify::create_graph "
				"level_of_candidates_file="
				<< level_of_candidates_file << endl;
	}

	CG = NULL;
	
	int ret;

	data_structures_groups::orbit_rep *R;
	actions::action_global AG;



	int max_starter;
	int nb;

	nb_vertices = 0;


	R = NEW_OBJECT(data_structures_groups::orbit_rep);
	if (f_v) {
		cout << "blt_set_classify::create_graph "
				"before R->init_from_file" << endl;
	}

	R->init_from_file(
			A, Blt_set_domain->prefix,
		starter_size, orbit_at_level, level_of_candidates_file, 
		blt_set_classify_early_test_func_callback,
		this /* early_test_func_callback_data */, 
		verbose_level - 2);
	if (f_v) {
		cout << "blt_set_classify::create_graph "
				"after R->init_from_file" << endl;
	}
	nb = q + 1 - starter_size;


	if (f_v) {
		cout << "blt_set_classify::create_graph Case "
				<< orbit_at_level << " / " << R->nb_cases
				<< " Read starter : ";
		Lint_vec_print(cout, R->rep, starter_size);
		cout << endl;
	}

	max_starter = R->rep[starter_size - 1];

	if (f_vv) {
		cout << "blt_set_classify::create_graph "
				"Case " << orbit_at_level
				<< " / " << R->nb_cases << " max_starter="
				<< max_starter << endl;
		cout << "blt_set_classify::create_graph "
				"Case " << orbit_at_level
				<< " / " << R->nb_cases << " Group order="
				<< *R->stab_go << endl;
		cout << "blt_set_classify::create_graph "
				"Case " << orbit_at_level
				<< " / " << R->nb_cases << " nb_candidates="
				<< R->nb_candidates << " at level "
				<< starter_size << endl;
	}



	if (f_lexorder_test) {
		int nb_candidates2;
	
		if (f_v3) {
			cout << "blt_set_classify::create_graph "
					"Case " << orbit_at_level
					<< " / " << R->nb_cases
					<< " Before lexorder_test" << endl;
		}
		AG.lexorder_test(A, R->candidates,
			R->nb_candidates, nb_candidates2,
			R->Strong_gens->gens, max_starter, 0 /*verbose_level - 3*/);
		if (f_vv) {
			cout << "blt_set_classify::create_graph "
					"After lexorder_test nb_candidates="
					<< nb_candidates2 << " eliminated "
					<< R->nb_candidates - nb_candidates2
					<< " candidates" << endl;
		}
		R->nb_candidates = nb_candidates2;
	}


	// we must do this. 
	// For instance, what if we have no points left,
	// then the minimal color stuff break down.
	//if (f_eliminate_graphs_if_possible) {
	if (R->nb_candidates < nb) {
		if (f_v) {
			cout << "blt_set_classify::create_graph "
					"Case " << orbit_at_level << " / "
					<< R->nb_cases << " nb_candidates < nb, "
							"the case is eliminated" << endl;
		}
		FREE_OBJECT(R);
		return false;
	}
		//}


	nb_vertices = R->nb_candidates;


	if (f_v) {
		cout << "blt_set_classify::create_graph before "
				"Blt_set_domain->create_graph" << endl;
		}
	ret = Blt_set_domain->create_graph(
			orbit_at_level, R->nb_cases,
			R->rep, starter_size,
			R->candidates, R->nb_candidates,
			f_eliminate_graphs_if_possible,
			CG,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "blt_set_classify::create_graph after "
				"Blt_set_domain->create_graph" << endl;
	}

	FREE_OBJECT(R);
	return ret;
}



void blt_set_classify::lifting_prepare_function_new(
		solvers_package::exact_cover *E, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	combinatorics::solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	int i, j, a;

	if (f_v) {
		cout << "blt_set_classify::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}




	int nb_free_points, nb_needed;
	long int *free_point_list; // [nb_free_points]
	int *point_idx; // [nb_points_total]
		// point_idx[i] = index of a point in free_point_list
		// or -1 if the point is in points_covered_by_starter


	nb_needed = q + 1 - starter_size;


	if (f_vv) {
		cout << "blt_set_classify::lifting_prepare_function "
				"nb_needed=" << nb_needed << endl;
		cout << "blt_set_classify::lifting_prepare_function "
				"nb_candidates=" << nb_candidates << endl;
	}

	if (f_v) {
		cout << "blt_set_classify::lifting_prepare_function "
				"before find_free_points" << endl;
	}

	Blt_set_domain->find_free_points(E->starter, starter_size,
		free_point_list, point_idx, nb_free_points,
		verbose_level - 2);

	if (f_v) {
		cout << "blt_set_classify::lifting_prepare_function "
				"There are " << nb_free_points << " free points" << endl;
	}



	col_labels = NEW_lint(nb_candidates);


	Lint_vec_copy(candidates, col_labels, nb_candidates);


	int nb_rows = nb_free_points;
	int nb_cols = nb_candidates;


	if (f_vv) {
		cout << "blt_set_classify::lifting_prepare_function_new candidates: ";
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << " (nb_candidates=" << nb_candidates << ")" << endl;
	}




	if (E->f_lex) {
		int nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens,
			verbose_level - 2);
		if (f_v) {
			cout << "blt_set_classify::lifting_prepare_function_new "
					"after lexorder test nb_candidates before: "
					<< nb_cols_before << " reduced to  " << nb_cols
					<< " (deleted " << nb_cols_before - nb_cols
					<< ")" << endl;
		}
	}

	if (f_vv) {
		cout << "blt_set_classify::lifting_prepare_function_new "
				"after lexorder test" << endl;
		cout << "blt_set_classify::lifting_prepare_function_new "
				"nb_cols=" << nb_cols << endl;
	}

	int *Pts1, *Pts2;

	Pts1 = NEW_int(nb_free_points * 5);
	Pts2 = NEW_int(nb_cols * 5);
	for (i = 0; i < nb_free_points; i++) {
		Blt_set_domain->O->Hyperbolic_pair->unrank_point(
				Pts1 + i * 5, 1,
				free_point_list[i],
				0 /*verbose_level - 1*/);
	}
	for (i = 0; i < nb_cols; i++) {
		Blt_set_domain->O->Hyperbolic_pair->unrank_point(
				Pts2 + i * 5, 1,
				col_labels[i],
				0 /*verbose_level - 1*/);
	}



	Dio = NEW_OBJECT(combinatorics::solvers::diophant);
	Dio->open(nb_rows, nb_cols, verbose_level - 1);
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
	}

	Dio->fill_coefficient_matrix_with(0);
	if (f_vv) {
		cout << "blt_set_classify::lifting_prepare_function_new "
				"initializing Inc" << endl;
	}


	for (i = 0; i < nb_free_points; i++) {
		for (j = 0; j < nb_cols; j++) {
			a = Blt_set_domain->O->Quadratic_form->evaluate_bilinear_form(
					Pts1 + i * 5,
					Pts2 + j * 5, 1);
			if (a == 0) {
				Dio->Aij(i, j) = 1;
			}
		}
	}


	FREE_lint(free_point_list);
	FREE_int(point_idx);
	FREE_int(Pts1);
	FREE_int(Pts2);
	if (f_v) {
		cout << "blt_set_classify::lifting_prepare_function_new "
				"nb_free_points=" << nb_free_points
				<< " nb_candidates=" << nb_candidates << endl;
	}

	if (f_v) {
		cout << "blt_set_classify::lifting_prepare_function_new "
				"done" << endl;
	}
}


void blt_set_classify::report_from_iso(
		isomorph::isomorph &Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify::report_from_iso" << endl;
	}

	data_structures_groups::orbit_transversal *T;

	if (f_v) {
		cout << "blt_set_classify::report_from_iso "
				"before Iso.get_orbit_transversal" << endl;
	}

	Iso.Folding->get_orbit_transversal(
			T, verbose_level);

	if (f_v) {
		cout << "blt_set_classify::report_from_iso "
				"after Iso.get_orbit_transversal" << endl;
	}

	report(T, verbose_level);

	FREE_OBJECT(T);

	if (f_v) {
		cout << "blt_set_classify::report_from_iso done" << endl;
	}
}


void blt_set_classify::report(
		data_structures_groups::orbit_transversal *T,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;

	if (f_v) {
		cout << "blt_set_classify::report" << endl;
	}

	fname = Blt_set_domain->prefix + "_report.tex";

	{
		ofstream ost(fname);

		report2(ost, T, verbose_level);
	}

	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "blt_set_classify::report done" << endl;
	}

}


void blt_set_classify::report2(
		std::ostream &ost,
		data_structures_groups::orbit_transversal *T,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify::report2" << endl;
	}

	int f_book = false;
	int f_title = true;
	string title, author, extra_praeamble;

	int f_toc = false;
	int f_landscape = false;
	int f_12pt = false;
	int f_enlarged_page = true;
	int f_pagenumbers = true;
	other::l1_interfaces::latex_interface L;

	title = "BLT-sets of ${\\cal Q}(4," + std::to_string(q) + ")$";

	author.assign("Orbiter");

	cout << "number of BLT-sets is " << T->nb_orbits << endl;

	L.head(ost, f_book, f_title,
		title,
		author,
		f_toc,
		f_landscape,
		f_12pt,
		f_enlarged_page,
		f_pagenumbers,
		extra_praeamble /* extra_praeamble */);


	int h;
	algebra::ring_theory::longinteger_object go;


	algebra::ring_theory::longinteger_object *Ago;

	Ago = NEW_OBJECTS(algebra::ring_theory::longinteger_object, T->nb_orbits);



	ost << "\\section{Summary}" << endl << endl;
	ost << "There are " << T->nb_orbits
			<< " isomorphism types of BLT-sets." << endl << endl;


	for (h = 0; h < T->nb_orbits; h++) {
		T->Reps[h].group_order(Ago[h]);
	}





	cout << "Computing intersection and plane invariants" << endl;


	geometry::orthogonal_geometry::blt_set_invariants *Inv;

	Inv = NEW_OBJECTS(geometry::orthogonal_geometry::blt_set_invariants, T->nb_orbits);

	for (h = 0; h < T->nb_orbits; h++) {


		if (f_v) {
			cout << "blt_set_classify::report2 looking at "
					"representative h=" << h << endl;
		}

		Inv[h].init(Blt_set_domain, T->Reps[h].data,
				verbose_level);



		Inv[h].compute(verbose_level);

	}


	cout << "Computing intersection and plane invariants done" << endl;

	//f << "\\section{Invariants}" << endl << endl;

	ost << "\\section{The BLT-Sets}" << endl << endl;



	for (h = 0; h < T->nb_orbits; h++) {


		ost << "\\subsection{Isomorphism Type " << h << "}" << endl;
		ost << "\\bigskip" << endl;


		if (T->Reps[h].Stab/*Iso.Reps->stab[h]*/) {
			T->Reps[h].Stab->group_order(go);
			ost << "Stabilizer has order $";
			go.print_not_scientific(ost);
			ost << "$\\\\" << endl;
		}
		else {
			//cout << endl;
		}

		Inv[h].latex(ost, verbose_level);




#if 0
		sims *Stab;

		Stab = T->Reps[h].Stab;

		if (f_v) {
			cout << "blt_set_classify::report computing induced action "
					"on the set (in data)" << endl;
		}
		Iso.induced_action_on_set(Stab, T->Reps[h].data, 0 /*verbose_level*/);

		longinteger_object go1;

		Iso.AA->group_order(go1);
		cout << "action " << Iso.AA->label << " computed, "
				"group order is " << go1 << endl;

		f << "Order of the group that is induced on the object is ";
		f << "$";
		go1.print_not_scientific(f);
		f << "$\\\\" << endl;

		{
			int nb_ancestors;
			nb_ancestors = Iso.UF->count_ancestors();

			f << "Number of ancestors on $" << Iso.level << "$-sets is "
					<< nb_ancestors << ".\\\\" << endl;

			int *orbit_reps;
			int nb_orbits;
			strong_generators *Strong_gens;

			Strong_gens = NEW_OBJECT(strong_generators);
			Strong_gens->init_from_sims(Iso.AA->Sims, 0);


			poset *Poset;

			Poset = NEW_OBJECT(poset);
			Poset->init_subset_lattice(Iso.AA, Iso.AA, Strong_gens,
					verbose_level);


			Poset->orbits_on_k_sets(
				Iso.level, orbit_reps, nb_orbits, verbose_level);

			FREE_OBJECT(Poset);
			f << "Number of orbits on $" << Iso.level << "$-sets is "
					<< nb_orbits << ".\\\\" << endl;
			FREE_int(orbit_reps);
			FREE_OBJECT(Strong_gens);
		}

		schreier Orb;
		//longinteger_object go2;

		Iso.AA->compute_all_point_orbits(Orb, Stab->gens,
				verbose_level - 2);
		f << "With " << Orb.nb_orbits
				<< " orbits on the object\\\\" << endl;

		classify C_ol;

		C_ol.init(Orb.orbit_len, Orb.nb_orbits, false, 0);

		f << "Orbit lengths: $";
		//int_vec_print(f, Orb.orbit_len, Orb.nb_orbits);
		C_ol.print_bare_tex(f, false /* f_reverse */);
		f << "$ \\\\" << endl;
#endif




		T->Reps[h].Strong_gens->print_generators_tex(ost);
		T->Reps[h].Strong_gens->print_generators_for_make_element(ost);

#if 0
		longinteger_object so;
		int i;

		T->Reps[h].Stab->group_order(so);
		f << "Stabilizer of order ";
		so.print_not_scientific(f);
		f << " is generated by:\\\\" << endl;
		for (i = 0; i < T->Reps[h].Stab->gens.len; i++) {

			int *fp, n;

			fp = NEW_int(A->degree);
			n = A->find_fixed_points(T->Reps[h].Stab->gens.ith(i), fp, 0);
			//cout << "with " << n << " fixed points" << endl;
			FREE_int(fp);

			f << "$$ g_{" << i + 1 << "}=" << endl;
			A->element_print_latex(T->Reps[h].Stab->gens.ith(i), f);
			f << "$$" << endl << "with " << n
					<< " fixed points" << endl;
		}
#endif


		string label_txt;
		string label_tex;
		blt_set_with_action *BA;


		label_txt = "BLT_set_q" + std::to_string(q) + "_iso" + std::to_string(h);
		label_tex = "BLT\\_set\\_q" + std::to_string(q) + "\\_iso" + std::to_string(h);


		BA = NEW_OBJECT(blt_set_with_action);
		BA->init_set(
				A, Blt_set_domain_with_action,
				T->Reps[h].data,
				label_txt,
				label_tex,
				T->Reps[h].Strong_gens,
				true /* f_invariants */,
				verbose_level);
		BA->Blt_set_group_properties->print_automorphism_group(ost);

		FREE_OBJECT(BA);
	}


	string label_of_structure_plural;

	label_of_structure_plural.assign("BLT-Sets");

	T->export_data_in_source_code_inside_tex(
			Blt_set_domain->prefix,
			label_of_structure_plural, ost,
			verbose_level);


	L.foot(ost);
	FREE_OBJECTS(Ago);
	FREE_OBJECTS(Inv);

}

// #############################################################################
// global functions:
// #############################################################################



static void blt_set_classify_print(
		std::ostream &ost, int len, long int *S, void *data)
{
	blt_set_classify *Gen = (blt_set_classify *) data;

	//print_vector(ost, S, len);
	Gen->Blt_set_domain->print(ost, S, len);
}


static void blt_set_classify_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	blt_set_classify *BLT = (blt_set_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_early_test_func for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	BLT->Blt_set_domain->early_test_func(
			S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "blt_set_early_test_func done" << endl;
	}
}



}}}







