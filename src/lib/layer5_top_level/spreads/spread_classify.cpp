// spread_classify.cpp
// 
// Anton Betten
// November 17, 2009
//
// moved to TOP_LEVEL: November 2, 2013
// renamed to spread.cpp from translation_plane.cpp: March 25, 2018
// renamed spread_classify.cpp from spread.cpp: Aug 4, 2019
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


static int starter_canonize_callback(long int *Set, int len, int *Elt,
	void *data, int verbose_level);
static int callback_incremental_check_function(
	int len, long int *S,
	void *data, int verbose_level);
static void spread_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);




spread_classify::spread_classify()
{

	Descr = NULL;
	SD = NULL;

	PA = NULL;
	Strong_gens = NULL;


	Mtx = NULL;

	block_size = 0;


	starter_size = 0;
	target_size = 0;

	
	A = NULL;
	A2 = NULL;
	AG = NULL;

	R = NULL;
	Base_case = NULL;

	Starter = NULL;
	Starter_size = 0;
	Starter_Strong_gens = NULL;

	Control = NULL;
	Poset = NULL;
	gen = NULL;

	//std::string prefix;

	Sing = NULL;

	Nb = 0;

	Worker = NULL;
}

spread_classify::~spread_classify()
{
#if 0
	if (A) {
		FREE_OBJECT(A);
	}
#endif
	if (A2) {
		FREE_OBJECT(A2);
	}
#if 0
	if (AG) {
		FREE_OBJECT(AG);
	}
#endif

	if (R) {
		FREE_OBJECT(R);
	}
	if (Base_case) {
		FREE_OBJECT(Base_case);
	}
	if (Starter) {
		FREE_lint(Starter);
	}
	if (Starter_Strong_gens) {
		FREE_OBJECT(Starter_Strong_gens);
	}

#if 1
	if (Sing) {
		FREE_OBJECT(Sing);
	}
#endif
#if 0
	if (Data3) {
		FREE_int(Data3);
	}
#endif
}


void spread_classify::init_basic(
		spread_classify_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_classify::init_basic" << endl;
		cout << "spread_classify::init_basic "
				"verbose_level = " << verbose_level << endl;
	}

	spread_classify::Descr = Descr;


	if (!Descr->f_projective_space) {
		cout << "spread_classify::init_basic please specify "
				"the projective space using -projective_space <string>" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "spread_classify::init_basic searching for projective space object with label " << Descr->projective_space_label << endl;
	}
	PA = Get_object_of_projective_space(Descr->projective_space_label);


	int n;

	n = PA->d;

	if (f_v) {
		cout << "spread_classify::init_basic n = " << n << endl;
	}

	if (!Descr->f_starter_size) {
		cout << "spread_classify::init_basic please specify "
				"the starter_size using -starter_size <int>" << endl;
		exit(1);
	}
	starter_size = Descr->starter_size;

	geometry::spread_domain *SD;

	SD = NEW_OBJECT(geometry::spread_domain);

	if (!Descr->f_k) {
		cout << "spread_classify::init_basic please specify "
				"the dimension over the kernel using -k <int>" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "spread_classify::init_basic before SD->init" << endl;
	}

	SD->init(
			PA->F,
			n, Descr->k,
			verbose_level - 1);

	if (f_v) {
		cout << "spread_classify::init_basic after SD->init" << endl;
	}





	spread_classify::A = PA->A;
	spread_classify::Strong_gens = PA->A->Strong_gens;



	if (!A->is_matrix_group()) {
		cout << "the group must be of matrix_group type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "spread_classify::init_basic" << endl;
	}


#if 0
	if (f_v) {
		cout << "spread_classify::init_basic "
				"before lex_least_base_in_place" << endl;
	}
	A->lex_least_base_in_place(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "spread_classify::init_basic "
				"after lex_least_base_in_place" << endl;
	}
	if (f_v) {
		cout << "spread_classify::init_group "
				"computing lex least base done" << endl;
		cout << "blt_set::init_group base: ";
		lint_vec_print(cout, A->get_base(), A->base_len());
		cout << endl;
	}
#endif



	//degree = SD->degree;
	target_size = SD->spread_size;


	if (f_v) {
		cout << "spread_classify::init_basic q=" << SD->F->q
				<< " target_size = " << target_size << endl;
	}


	if (f_v) {
		cout << "spread_classify::init_basic before init" << endl;
	}
	init(SD, PA, verbose_level);
	if (f_v) {
		cout << "spread_classify::init_basic after init" << endl;
	}


	if (f_v) {
		cout << "spread_classify::init_basic finished" << endl;
	}
}



void spread_classify::init(
		geometry::spread_domain *SD,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;
	
	
	if (f_v) {
		cout << "spread_classify::init" << endl;
		//cout << "k=" << k << endl;
	}


	spread_classify::SD = SD;
	spread_classify::PA = PA;
	spread_classify::A = PA->A;

	//n = A->matrix_group_dimension();
	//spread_classify::k = k;
	Mtx = A->get_matrix_group();
	//F = Mtx->GFq;

	if (A->matrix_group_dimension() != SD->n) {
		cout << "spread_classify::init the dimension of the matrix group is not correct" << endl;
		exit(1);
	}
	if (A->get_matrix_group()->GFq->q != SD->q) {
		cout << "spread_classify::init the matrix group is not over the correct field" << endl;
		exit(1);
	}



	block_size = SD->kC1q;

#if 0
	Control->f_depth = TRUE;
	Control->depth = spread_size;
	if (f_v) {
		cout << "spread_classify::init" << endl;
		cout << "Control:" << endl;
		Control->print();
	}
#endif


	
#if 0
	if (k == (n >> 1)) {
		f_recoordinatize = TRUE;
	}
	else {
		f_recoordinatize = FALSE;
	}

	if (f_v) {
		cout << "spread_classify::init f_recoordinatize = " << f_recoordinatize << endl;
	}
#endif



	
	gen = NEW_OBJECT(poset_classification::poset_classification);



	A2 = NEW_OBJECT(actions::action);
	AG = NEW_OBJECT(induced_actions::action_on_grassmannian);

#if 0
	longinteger_object go;
	A->Sims->group_order(go);
	if (f_v) {
		cout << "spread_classify::init go = " << go <<  endl;
	}
#endif


	if (f_vv) {
		cout << "action A created: ";
		A->print_info();
	}





	if (f_v) {
		cout << "spread_classify::init before AG->init" <<  endl;
	}
	
	AG->init(*A, SD->Grass, 0 /*verbose_level - 2*/);
	
	if (f_v) {
		cout << "spread_classify::init after AG->init" <<  endl;
	}

	if (f_v) {
		cout << "spread_classify::init before "
				"A2->induced_action_on_grassmannian" <<  endl;
	}

	A2->induced_action_on_grassmannian(A, AG, 
		FALSE /*f_induce_action*/, NULL /*sims *old_G */,
		0 /*verbose_level - 2*/);
	
	if (f_v) {
		cout << "spread_classify::init after "
				"A2->induced_action_on_grassmannian" <<  endl;
	}

	if (f_vv) {
		cout << "action A2 created: ";
		A2->print_info();
	}

#if 0
	if (!A->f_has_strong_generators) {
		cout << "action does not have strong generators" << endl;
		exit(1);
	}
#endif



	if (FALSE) {
		int f_print_as_permutation = TRUE;
		int f_offset = FALSE;
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = FALSE;
		
		cout << "printing generators for the group:" << endl;
		A->Strong_gens->gens->print(cout, f_print_as_permutation, 
			f_offset, offset, 
			f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
	}


#if 0
	len = gens->len;
	for (i = 0; i < len; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(gens->ith(i), cout);
		cout << endl;
		if (A2->degree < 150) {
			A2->element_print_as_permutation(gens->ith(i), cout);
			cout << endl;
		}
	}
#endif


#if 0
	if (nb_pts < 50) {
		print_points();
	}



	if (A2->degree < 150) {
		print_elements();
		print_elements_and_points();
	}
#endif


	if (TRUE /*f_v*/) {
		ring_theory::longinteger_object go;
		
		A->Strong_gens->group_order(go);
		cout << "spread_classify::init The order of PGGL(n,q) is " << go << endl;
	}

	
	if (Descr->f_recoordinatize) {
		if (f_v) {
			cout << "spread_classify::init before recoordinatize::init" << endl;
		}
		//char str[1000];
		//string fname_live_points;

		//snprintf(str, sizeof(str), "live_points_q%d", q);
		//fname_live_points.assign(str);

		R = NEW_OBJECT(recoordinatize);
		R->init(SD,  // SD->n, SD->k, SD->F, SD->Grass,
				A, A2,
				TRUE /*f_projective*/, Mtx->f_semilinear,
				callback_incremental_check_function, (void *) this,
				//fname_live_points,
				verbose_level);

		if (f_v) {
			cout << "spread_classify::init before "
					"recoordinatize::compute_starter" << endl;
		}
		R->compute_starter(Starter, Starter_size, 
			Starter_Strong_gens, verbose_level - 2);
		if (f_v) {
			cout << "spread_classify::init after "
					"recoordinatize::compute_starter" << endl;
		}

		ring_theory::longinteger_object go;
		Starter_Strong_gens->group_order(go);
		if (TRUE /*f_v*/) {
			cout << "spread_classify::init The stabilizer of the "
					"first three components has order " << go << endl;
		}

		Nb = R->nb_live_points;
	}
	else {
		if (f_v) {
			cout << "spread_classify::init we are not using "
					"recoordinatization" << endl;
			//exit(1);
		}
		Nb = Combi.generalized_binomial(SD->n, SD->k, SD->q); //R->nCkq; // this makes no sense
	}

	if (f_v) {
		cout << "spread_classify::init Nb = " << Nb << endl;
		cout << "spread_classify::init kn = " << SD->kn << endl;
		cout << "spread_classify::init n = " << SD->n << endl;
		cout << "spread_classify::init k = " << SD->k << endl;
		cout << "spread_classify::init allocating Data1 and Data2" << endl;
	}
	
	//Data3 = NEW_int(n * n);
	

#if 0
	if (k == 2 && is_prime(q)) {
		Sing = NEW_OBJECT(singer_cycle);
		if (f_v) {
			cout << "spread_classify::init "
					"before singer_cycle::init" << endl;
		}
		Sing->init(4, F, A, A2, 0 /*verbose_level*/);
		Sing->init_lines(0 /*verbose_level*/);
	}
#endif

	
#if 1
	if (f_v) {
		cout << "spread_classify::init before init2" << endl;
	}
	init2(verbose_level - 1);
	if (f_v) {
		cout << "spread_classify::init after init2" << endl;
	}
#endif



	if (f_v) {
		cout << "spread_classify::init done" << endl;
	}
}

void spread_classify::init2(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_classify::init2" << endl;
	}

	if (!Descr->f_poset_classification_control) {
		cout << "spread_classify::init2 need -poset_classification_control <options> -end" << endl;
		exit(1);
	}

	poset_classification::poset_classification_control *Control;

	Control = Get_object_of_type_poset_classification_control(Descr->poset_classification_control_label);


	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(A, A2,
			A->Strong_gens,
			verbose_level);
	Poset->add_testing_without_group(
			spread_early_test_func_callback,
				this /* void *data */,
				verbose_level);


	if (Descr->f_recoordinatize) {
		if (f_v) {
			cout << "spread_classify::init2 "
					"f_recoordinatize is TRUE" << endl;
		}
		if (f_v) {
			cout << "spread_classify::init2 "
					"before gen->initialize_with_starter" << endl;
		}

		Base_case = NEW_OBJECT(poset_classification::classification_base_case);

		Base_case->init(Poset,
				Starter_size,
				Starter,
				R->live_points,
				R->nb_live_points,
				Starter_Strong_gens,
				this,
				starter_canonize_callback,
				verbose_level);


		gen->initialize_with_base_case(Control, Poset,
			SD->spread_size,
			Base_case,
			verbose_level - 2);
		if (f_v) {
			cout << "spread_classify::init2 "
					"after gen->initialize_with_starter" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "spread_classify::init2 "
					"f_recoordinatize is FALSE" << endl;
		}
		if (f_v) {
			cout << "spread_classify::init2 "
					"before gen->initialize" << endl;
		}
		gen->initialize_and_allocate_root_node(Control, Poset,
			SD->spread_size,
			verbose_level - 2);
		if (f_v) {
			cout << "spread_classify::init2 "
					"after gen->initialize" << endl;
		}
	}

	//gen->f_allowed_to_show_group_elements = TRUE;


#if 0
	gen->f_print_function = TRUE;
	gen->print_function = callback_spread_print;
	gen->print_function_data = this;
#endif


	prefix.assign(gen->get_problem_label_with_path());

	if (f_v) {
		cout << "spread_classify::init2 done" << endl;
	}
}




void spread_classify::classify_partial_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0;
	orbiter_kernel_system::os_interface Os;


	if (f_v) {
		cout << "spread_classify::classify_partial_spreads" << endl;
	}

	if (gen->get_depth() < starter_size) {
		cout << "spread_classify::classify_partial_spreads gen->depth < starter_size" << endl;
		exit(1);
	}

	gen->get_depth() = starter_size;

	if (f_v) {
		cout << "spread_classify::classify_partial_spreads Control->max_depth=" << gen->get_control()->depth << endl;
	}


	schreier_depth = starter_size; // gen->get_control()->depth;
	
	if (f_v) {
		cout << "spread_classify::classify_partial_spreads calling generator_main" << endl;
	}

	t0 = Os.os_ticks();
	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	int length;
	
	if (f_v) {
		cout << "spread_classify::classify_partial_spreads done with generator_main" << endl;
	}
	length = gen->nb_orbits_at_level(gen->get_control()->depth);
	if (f_v) {
		cout << "spread_classify::compute We found " << length << " orbits on "
			<< gen->get_control()->depth << "-sets of " << SD->k
			<< "-subspaces in PG(" << SD->n - 1 << "," << SD->q << ")"
			<< " satisfying the partial spread condition" << endl;
	}



	if (f_v) {
		cout << "spread_classify::classify_partial_spreads done" << endl;
	}
}

void spread_classify::lifting(
		int orbit_at_level, int level_of_candidates_file,
		int f_lexorder_test, int f_eliminate_graphs_if_possible,
		int &nb_vertices,
		solvers::diophant *&Dio,
		long int *&col_labels,
		int &f_ruled_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "spread_classify::lifting" << endl;
	}


	if (f_v) {
		cout << "spread_classify::lifting "
				"prefix=" << prefix << endl;
		cout << "spread_classify::lifting "
				"f_lexorder_test=" << f_lexorder_test << endl;
		cout << "spread_classify::lifting "
				"orbit_at_level=" << orbit_at_level << endl;
		cout << "spread_classify::lifting "
				"level_of_candidates_file=" << level_of_candidates_file << endl;
	}

	f_ruled_out = FALSE;

	data_structures_groups::orbit_rep *R;



	int max_starter;
	int nb;

	nb_vertices = 0;


	R = NEW_OBJECT(data_structures_groups::orbit_rep);
	if (f_v) {
		cout << "spread_classify::lifting before R->init_from_file" << endl;
	}

	R->init_from_file(A, prefix,
		starter_size, orbit_at_level, level_of_candidates_file,
		spread_early_test_func_callback,
		this /* early_test_func_callback_data */,
		verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::lifting after R->init_from_file" << endl;
	}
	nb = target_size - starter_size;


	if (f_v) {
		cout << "spread_classify::lifting Case "
				<< orbit_at_level << " / " << R->nb_cases
				<< " Read starter : ";
		Lint_vec_print(cout, R->rep, starter_size);
		cout << endl;
	}

	max_starter = R->rep[starter_size - 1];

	if (f_vv) {
		cout << "spread_classify::lifting Case " << orbit_at_level
				<< " / " << R->nb_cases << " max_starter="
				<< max_starter << endl;
		cout << "spread_classify::lifting Case " << orbit_at_level
				<< " / " << R->nb_cases << " Group order="
				<< *R->stab_go << endl;
		cout << "spread_classify::lifting Case " << orbit_at_level
				<< " / " << R->nb_cases << " nb_candidates="
				<< R->nb_candidates << " at level "
				<< starter_size << endl;
	}



	if (f_lexorder_test) {
		int nb_candidates2;

		if (f_v3) {
			cout << "spread_classify::lifting Case " << orbit_at_level
					<< " / " << R->nb_cases
					<< " Before lexorder_test" << endl;
		}
		A->lexorder_test(R->candidates,
			R->nb_candidates, nb_candidates2,
			R->Strong_gens->gens, max_starter, 0 /*verbose_level - 3*/);
		if (f_vv) {
			cout << "spread_classify::lifting "
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
			cout << "spread_classify::lifting "
					"Case " << orbit_at_level << " / "
					<< R->nb_cases << " nb_candidates < nb, "
							"the case is eliminated" << endl;
		}
		FREE_OBJECT(R);
		f_ruled_out = TRUE;

		spread_lifting *SL;

		SL = NEW_OBJECT(spread_lifting);

		if (f_v) {
			cout << "spread_classify::lifting "
					"before SL->init" << endl;
		}
		SL->init(this,
				R,
				Descr->output_prefix,
				//Strong_gens,
				FALSE /* E->f_lex */,
				verbose_level);
		if (f_v) {
			cout << "spread_classify::lifting "
					"after SL->init" << endl;
		}

		SL->create_dummy_graph(verbose_level);


		FREE_OBJECT(SL);

		return;
	}
		//}


	nb_vertices = R->nb_candidates;


	if (!Descr->f_output_prefix) {
		cout << "spread_classify::lifting -output_prefix has not been set" << endl;
	}
	if (f_v) {
		cout << "spread_classify::lifting before "
				"setup_lifting" << endl;
		}
	setup_lifting(
			R,
			Descr->output_prefix,
			Dio, col_labels,
			f_ruled_out,
			verbose_level);
	if (f_v) {
		cout << "spread_classify::lifting after "
				"setup_lifting" << endl;
	}

	FREE_OBJECT(R);
	if (f_v) {
		cout << "spread_classify::lifting done" << endl;
	}
}

void spread_classify::setup_lifting(
		data_structures_groups::orbit_rep *R,
		std::string &output_prefix,
		solvers::diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_v3 = (verbose_level >= 3);
	
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"nb_candidates=" << R->nb_candidates << endl;
	}


	spread_lifting *SL;

	SL = NEW_OBJECT(spread_lifting);

	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before SL->init" << endl;
	}
	SL->init(this,
			R,
			output_prefix,
		//R->rep /* starter */, R->level /* starter_size */,
		//R->orbit_at_level, R->nb_cases,
		//R->candidates, R->nb_candidates,
		//Strong_gens,
		FALSE /* E->f_lex */,
		verbose_level);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after SL->init" << endl;
	}


	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before SL->compute_colors" << endl;
	}

	SL->compute_colors(f_ruled_out, verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after SL->compute_colors" << endl;
	}

	if (f_ruled_out) {
		if (f_v) {
			cout << "spread_classify::setup_lifting "
					"the case is ruled out." << endl;
		}

		SL->create_dummy_graph(verbose_level);


		FREE_OBJECT(SL);
		return;
	}
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before SL->reduce_candidates" << endl;
	}

	SL->reduce_candidates(verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after SL->reduce_candidates" << endl;
	}
	
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before SL->create_system" << endl;
	}

	Dio = SL->create_system(verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after SL->create_system" << endl;
	}

#if 0
	int *col_color;
	int nb_colors;

	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before SL->find_coloring" << endl;
	}
	SL->find_coloring(Dio, 
		col_color, nb_colors, 
		verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after SL->find_coloring" << endl;
	}

	if (f_v3) {
		cout << "col_color=";
		Int_vec_print(cout, col_color, Dio->n);
		cout << endl;
	}
#endif

	data_structures::bitvector *Adj;
	
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before Dio->make_clique_graph_adjacency_matrix" << endl;
	}
	Dio->make_clique_graph_adjacency_matrix(Adj, verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after Dio->make_clique_graph_adjacency_matrix" << endl;
	}


	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before SL->create_graph" << endl;
	}
	SL->create_graph(Adj, verbose_level);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after SL->create_graph" << endl;
	}

	col_labels = SL->col_labels;
	SL->col_labels = NULL;

	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before FREE_OBJECT(SL)" << endl;
	}
	FREE_OBJECT(SL);
	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"before FREE_OBJECT(Adj)" << endl;
	}
	FREE_OBJECT(Adj);
	//FREE_int(col_color);

	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"after SL->create_system" << endl;
	}

	if (f_v) {
		cout << "spread_classify::setup_lifting "
				"done" << endl;
	}
}


#if 0
void spread_classify::lifting_prepare_function_new(
	exact_cover *E, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v3 = (verbose_level >= 3);

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}


	spread_lifting *SL;

	SL = NEW_OBJECT(spread_lifting);

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before SL->init" << endl;
	}
	SL->init(this, E,
		E->starter, E->starter_size,
		starter_case, E->starter_nb_cases,
		candidates, nb_candidates, Strong_gens,
		E->f_lex,
		verbose_level);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->init" << endl;
	}


	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before SL->create_system" << endl;
	}

	Dio = SL->create_system(verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->create_system" << endl;
	}

	int *col_color;
	int nb_colors;

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before SL->find_coloring" << endl;
	}
	SL->find_coloring(Dio,
		col_color, nb_colors,
		verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->find_coloring" << endl;
	}

	if (f_v3) {
		cout << "col_color=";
		Int_vec_print(cout, col_color, Dio->n);
		cout << endl;
	}

	data_structures::bitvector *Adj;

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before Dio->make_clique_graph_adjacency_matrix" << endl;
	}
	Dio->make_clique_graph_adjacency_matrix(Adj, verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after Dio->make_clique_graph_adjacency_matrix" << endl;
	}

	graph_theory::colored_graph *CG;

	CG = NEW_OBJECT(graph_theory::colored_graph);

	char str[1000];
	string label, label_tex;
	snprintf(str, sizeof(str), "graph_%d", starter_case);
	label.assign(str);
	label_tex.assign(str);

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before CG->init_with_point_labels" << endl;
	}
	CG->init_with_point_labels(SL->nb_cols, nb_colors, 1,
		col_color,
		Adj, TRUE /* f_ownership_of_bitvec */,
		SL->col_labels /* point_labels */,
		label, label_tex,
		verbose_level);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after CG->init_with_point_labels" << endl;
	}

	string fname_clique_graph;
	orbiter_kernel_system::file_io Fio;

	fname_clique_graph.assign(E->output_prefix);
	fname_clique_graph.append(str);
	fname_clique_graph.append(".graph");

	CG->save(fname_clique_graph, verbose_level - 1);
	if (f_v) {
		cout << "Written file " << fname_clique_graph
				<< " of size " << Fio.file_size(fname_clique_graph) << endl;
	}

	FREE_OBJECT(CG);

	col_labels = SL->col_labels;
	SL->col_labels = NULL;

	FREE_OBJECT(SL);
	//FREE_uchar(Adj);
	FREE_int(col_color);

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->create_system" << endl;
	}

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"done" << endl;
	}
}

#endif




// #############################################################################
// global functions:
// #############################################################################

#if 0
static void spread_lifting_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	spread_classify *Spread = (spread_classify *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "spread_lifting_early_test_function for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	Spread->early_test_func(S, len,
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "spread_lifting_early_test_function done" << endl;
	}
}

static void spread_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	spread_classify *Spread = (spread_classify *) EC->user_data;

	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	Spread->lifting_prepare_function_new(EC, starter_case,
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level);


	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"after lifting_prepare_function_new" << endl;
	}

	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"nb_rows=" << Dio->m
				<< " nb_cols=" << Dio->n << endl;
	}

	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"done" << endl;
	}
}
#endif




static int starter_canonize_callback(long int *Set, int len,
		int *Elt, void *data, int verbose_level)
// for starter, interface to recoordinatize,
// which uses callback_incremental_check_function
{
	spread_classify *Spread = (spread_classify *) data;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "starter_canonize_callback" << endl;
	}
	Spread->R->do_recoordinatize(Set[0], Set[1], Set[2], verbose_level - 2);
	Spread->A->element_move(Spread->R->Elt, Elt, FALSE);
	if (f_v) {
		cout << "starter_canonize_callback done" << endl;
	}
	if (f_vv) {
		cout << "transporter:" << endl;
		Spread->A->element_print(Elt, cout);
	}
	return TRUE;
}

static int callback_incremental_check_function(
		int len, long int *S, void *data, int verbose_level)
// for recoordinatize
{
	spread_classify *Spread = (spread_classify *) data;
	int ret;

	ret = Spread->SD->incremental_check_function(len, S, verbose_level);
	return ret;
}

static void spread_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	spread_classify *SC = (spread_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_early_test_func_callback for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	SC->SD->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "spread_early_test_func_callback done" << endl;
	}
}

}}}




