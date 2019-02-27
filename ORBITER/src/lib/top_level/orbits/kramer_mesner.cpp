// kramer_mesner.C
// 
// Anton Betten
// April 20, 2009
//
// moved here: Dec 1, 2015
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


kramer_mesner::~kramer_mesner()
{
	if (A) {
		if (f_A_is_allocated) {
			FREE_OBJECT(A);
			}
		A = NULL;
		}
	if (final_A) {
		FREE_OBJECT(final_A);
		final_A = NULL;
		}
	if (gen) {
		FREE_OBJECT(gen);
		gen = NULL;
		}
	if (Surf) {
		FREE_OBJECT(Surf);
		}
}

kramer_mesner::kramer_mesner()
{
	n = 0;
	p = 0;
	q = 0;
	h = 0;

	override_poly = NULL;

	final_A = NULL;
	F = NULL;
	Poset = NULL;
	gen = NULL;
	
	f_linear = FALSE;
	Descr = NULL;
	LG = NULL;
	A = NULL;
	A2 = NULL;
	f_A_is_allocated = FALSE;
	


	
	f_list = FALSE;
	f_arc = FALSE;
	f_surface = FALSE;
	Surf = NULL;
	f_KM = FALSE;

	f_draw_poset = FALSE;
	f_embedded = FALSE;
	f_sideways = FALSE;

	nb_identify = 0;
	Identify_label = NULL;
	Identify_data = NULL;
	Identify_length = NULL;


	//f_subgroup_by_standard_generators = FALSE;
	//f_subgroup_singer = FALSE;


	f_orbits_t = FALSE;
	orbits_t = -1;
	f_orbits_k = FALSE;
	orbits_k = -1;
}



void kramer_mesner::read_arguments(
	int argc, const char **argv,
	int &verbose_level)
{
	int i, j;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}

		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = NEW_OBJECT(linear_group_description);
			i += Descr->read_arguments(
					argc - (i - 1), argv + i, verbose_level);

			cout << "after Descr->read_arguments" << endl;
			}


		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list" << endl;
			}
		else if (strcmp(argv[i], "-arc") == 0) {
			f_arc = TRUE;
			cout << "-arc" << endl;
			}
		else if (strcmp(argv[i], "-surface") == 0) {
			f_surface = TRUE;
			cout << "-surface" << endl;
			}
		else if (strcmp(argv[i], "-KM") == 0) {
			f_KM = TRUE;
			cout << "-KM" << endl;
			}
		else if (strcmp(argv[i], "-t") == 0) {
			f_orbits_t = TRUE;
			orbits_t = atoi(argv[++i]);
			cout << "-t " << orbits_t << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_orbits_k = TRUE;
			orbits_k = atoi(argv[++i]);
			cout << "-k " << orbits_k << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-identify") == 0) {
			if (nb_identify == 0) {
				Identify_label = NEW_pchar(1000);
				Identify_data = NEW_pint(1000);
				Identify_length = NEW_int(1000);
				}
			int data[1000];
			int data_sz = 0;
			cout << "-identify " << endl;
			const char *label = argv[++i];
			cout << "-identify " << label << endl;
			Identify_label[nb_identify] = NEW_char(strlen(label) + 1);
			strcpy(Identify_label[nb_identify], label);
			for (j = 0; ; j++) {
				data[j] = atoi(argv[++i]);
				if (data[j] == -1) {
					break;
					}
				}
			data_sz = j;
			Identify_data[nb_identify] = NEW_int(data_sz);
			Identify_length[nb_identify] = data_sz;
			int_vec_copy(data, Identify_data[nb_identify], data_sz);
			cout << "-identify " << Identify_label[nb_identify] << " ";
			for (j = 0; j < Identify_length[nb_identify]; j++) {
				cout << Identify_data[nb_identify][j] << " ";
				}
			cout << "-1" << endl;
			nb_identify++;
			}
		else {
			cout << "option " << argv[i] << " unrecognized" << endl;
			}
		}
	if (f_orbits_t) {
		cout << "orbits_t=" << orbits_t << endl;
		}
	if (f_orbits_k) {
		cout << "orbits_k=" << orbits_k << endl;
		}
}


void kramer_mesner::init_group(sims *&S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "kramer_mesner::init_group q=" << q << endl;
		}
	is_prime_power(q, p, h);



	if (f_linear) {
		if (f_v) {
			cout << "kramer_mesner::init_group linear group" << endl;
			}
		F = NEW_OBJECT(finite_field);

		F->init(Descr->input_q, 0);
		//F->init_override_polynomial(Descr->input_q, override_poly, 0);
		Descr->F = F;
		q = Descr->input_q;
		//F->init(q, 0);

		//F->init_override_polynomial(q,
		//override_poly, 0 /*verbose_level - 1 */);
	


		LG = NEW_OBJECT(linear_group);

		cout << "kramer_mesner::init_group before LG->init, "
				"creating the group" << endl;

		LG->init(Descr, verbose_level);
	
		cout << "kramer_mesner::init_group after LG->init, "
				"strong generators for the group have been created" << endl;

		A = LG->A_linear;
		A2 = LG->A2;
		S = LG->Strong_gens->create_sims(0 /*verbose_level */);
		

		if (f_v) {
			cout << "kramer_mesner::init_group after "
					"create_linear_group" << endl;
			}
		}
	else {
		cout << "kramer_mesner::init_group other than linear "
				"is not yet implemented" << endl;
		exit(1);
		}
	
	

	if (f_surface) {

		Surf = NEW_OBJECT(surface);

		if (f_v) {
			cout << "kramer_mesner::init_group before Surf->init" << endl;
			}
		Surf->init(F, verbose_level);
		if (f_v) {
			cout << "kramer_mesner::init_group after Surf->init" << endl;
			}
		}
	
	if (f_v) {
		cout << "kramer_mesner::init_group finished" << endl;
		}
	
}

void kramer_mesner::orbits(
		int argc, const char **argv,
		sims *S, int verbose_level)
// the group is in A->strong_generators, A->transversal_length
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int t0;

	t0 = os_ticks();
	
	if (f_v) {
		cout << "kramer_mesner::orbits" << endl;
		cout << "computing orbits up to depth " << orbits_k << endl;
		}
	gen = NEW_OBJECT(poset_classification);
	gen->read_arguments(argc, argv, verbose_level);
	
	if (!f_orbits_t) {
		cout << "kramer_mesner::orbits please specify t" << endl;
		exit(1);
		}
	if (!f_orbits_k) {
		cout << "kramer_mesner::orbits please specify k" << endl;
		exit(1);
		}
	//gen->depth = orbits_depth;
	gen->depth = orbits_k;
	


	if (f_v) {
		cout << "kramer_mesner::orbits preparing "
				"strong generators:" << endl;
		}

	strong_generators *Strong_gens;

	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(S, 0);
	
#if 0
	if (!A->f_has_strong_generators) {
		cout << "kramer_mesner::orbits action has "
				"no strong generators" << endl;
		exit(1);
		}
#endif

	if (f_v) {
		cout << "kramer_mesner::orbits before gen->init:" << endl;
		}

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A2,
			Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "blt_set::init2 before "
				"Poset->add_testing_without_group" << endl;
		}

#if 0
	Poset->add_testing_without_group(
				early_test_func_callback,
				this /* void *data */,
				verbose_level);
#endif

	gen->init(Poset,
		gen->depth /* sz */, verbose_level - 2);


#if 0
	if (f_arc) {
		//gen->f_its_OK_to_not_have_an_early_test_func = TRUE;
		gen->init_check_func(kramer_mesner_test_arc, this);
		//gen->init_incremental_check_func(test_arc, this);
		}

	if (f_surface) {
		gen->init_check_func(kramer_mesner_test_surface, this);
		}
#endif

	int nb_nodes = 100;
	
	gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	
	sprintf(gen->fname_base, "%s", A2->label);
	
	if (f_v) {
		cout << "kramer_mesner::orbits calling "
				"init_root_node:" << endl;
		}

	gen->root[0].init_root_node(gen, gen->verbose_level);

	if (f_v) {
		cout << "kramer_mesner::orbits "
				"init_root_node finished" << endl;
		}

	int schreier_depth = 1000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	
	
	if (f_v) {
		cout << "kramer_mesner::orbits computing orbits" << endl;
		}
	gen->main(t0, schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level);

	if (f_v) {
		cout << "kramer_mesner::orbits computing orbits done" << endl;
		}


	if (f_draw_poset) {
		if (f_v) {
			cout << "before gen->draw_poset" << endl;
			}
		gen->draw_poset(gen->fname_base, gen->depth, 
			0 /* data1 */, f_embedded,
			f_sideways,
			0 /* gen->verbose_level */);
		}

	if (f_v) {
		cout << "kramer_mesner::orbits calling "
				"gen->write_lvl " << orbits_t << endl;
		}
	{
		char fname[1000];
		sprintf(fname, "%s_orbits_%d.txt", A2->label, orbits_t);
		{
			ofstream fp(fname);
			gen->write_lvl(fp, orbits_t, t0,
					FALSE /* f_with_stabilizer_generators */,
					FALSE /* f_long_version */,
					0 /* verbose_level */);
		}
		cout << "kramer_mesner::orbits written file " << fname
				<< " of size " << file_size(fname) << endl;
	}

	if (f_v) {
		cout << "kramer_mesner::orbits calling "
				"gen->write_lvl " << orbits_k << endl;
		}
	{
		char fname[1000];
		sprintf(fname, "%s_orbits_%d.txt", A2->label, orbits_k);
		{
			ofstream fp(fname);
			gen->write_lvl(fp, orbits_k, t0,
					FALSE /* f_with_stabilizer_generators */,
					FALSE /* f_long_version */,
					0 /* verbose_level */);
		}
		cout << "kramer_mesner::orbits written file " << fname
				<< " of size " << file_size(fname) << endl;
	}


	if (f_KM) {
	
		if (f_v) {
			cout << "kramer_mesner::orbits computing "
					"Kramer Mesner matrices" << endl;
			}

		// compute Kramer Mesner matrices
		Vector V;
		int i;
	
		V.m_l(orbits_k);
		for (i = 0; i < orbits_k; i++) {
			V[i].change_to_matrix();
			calc_Kramer_Mesner_matrix_neighboring(gen,
					i, V[i].as_matrix(), verbose_level - 2);
			if (f_v) {
				cout << "kramer_mesner::orbits matrix "
						"level " << i << " computed" << endl;
				}
			if (f_vv) {
				cout << "kramer_mesner::orbits matrix "
						"level " << i << ":" << endl;
				//V[i].as_matrix().print(cout);
				}
			}
	
		matrix Mtk;
		int n, m;
	
		Mtk_from_MM(V, Mtk, orbits_t, orbits_k,
				FALSE, 0, verbose_level - 2);
		m = Mtk.s_m();
		n = Mtk.s_n();
		cout << "kramer_mesner::orbits M_{" << orbits_t << ","
				<< orbits_k << "} has size " << m << " x "
				<< n << ":" << endl;
		Mtk.print(cout);
	
		if (f_v) {
			cout << "kramer_mesner::orbits computing "
					"Kramer Mesner matrices done" << endl;
			}

		}

	if (f_list) {
		int *Orbit_reps;
		int nb_orbits;
		int i;

		gen->get_orbit_representatives(orbits_k,
				nb_orbits, Orbit_reps, 0 /* verbose_level */);

		cout << "We found " << nb_orbits << " orbits at level "
				<< orbits_k << endl;
		int_matrix_print(Orbit_reps, nb_orbits, orbits_k);

		for (i = 0; i < nb_orbits; i++) {
			set_and_stabilizer *SaS;

			cout << "orbit " << i << " / " << nb_orbits << ":" << endl;
			SaS = gen->get_set_and_stabilizer(orbits_k,
					i /* orbit_at_level */, verbose_level);
			action *A_on_set;
			schreier *Orb;
			
			if (f_v) {
				cout << "set_and_stabilizer::rearrange_by_orbits "
					"creating restricted action on the set of lines" << endl;
				}
			A_on_set = SaS->A2->restricted_action(
					SaS->data, SaS->sz, verbose_level);
			Orb = SaS->Strong_gens->orbits_on_points_schreier(
					A_on_set, verbose_level);
			
			cout << "orbits on the set:" << endl;
			Orb->print_and_list_orbits_using_labels(cout, SaS->data);
			FREE_OBJECT(Orb);
			FREE_OBJECT(A_on_set);
			FREE_OBJECT(SaS);
			}
		
		if (f_surface) {
			int *R;
			R = NEW_int(nb_orbits);
			for (i = 0; i < nb_orbits; i++) {
				cout << "Orbit " << i << " / " << nb_orbits << ":" << endl;
				R[i] = Surf->compute_system_in_RREF(orbits_k,
						Orbit_reps + i * orbits_k, verbose_level);
				}
			{
			classify C;
			C.init(R, nb_orbits, FALSE, 0);
			cout << "Rank distribution of all " << nb_orbits
					<< " orbit reps:" << endl;
			C.print_naked(TRUE);
			cout << endl;
			}
			}
		}

	if (nb_identify) {
		int i, j, a, b, idx;
		int *data_out;
		int *Elt;
		
		Elt = NEW_int(A->elt_size_in_int);
		for (i = 0; i < nb_identify; i++) {

			set_and_stabilizer *SaS;


			data_out = NEW_int(Identify_length[i]);
			cout << "identifying " << Identify_label[i] << " : ";
			int_vec_print(cout, Identify_data[i], Identify_length[i]);
			cout << endl;
			idx = gen->trace_set(Identify_data[i],
				Identify_length[i], Identify_length[i],
				data_out, Elt, 
				0 /* verbose_level */);

			cout << "the set belongs to orbit = " << idx
					<< " canonical representative = ";
			int_vec_print(cout, data_out, Identify_length[i]);
			cout << endl;
			cout << "a transporter from the given set to "
					"the canonical set is:" << endl;
			A->element_print_quick(Elt, cout);
			cout << endl;

			cout << "testing the mapping in action "
					<< gen->Poset->A2->label << ":" << endl;
			for (j = 0; j < Identify_length[i]; j++) {
				a = Identify_data[i][j];
				b = gen->Poset->A2->element_image_of(a, Elt, 0);
				cout << a << " -> " << b << endl;
				}

			SaS = gen->get_set_and_stabilizer(Identify_length[i],
					idx /* orbit_at_level */, verbose_level);
			cout << "pulled out set and stabilizer at level "
					<< Identify_length[i] << " for orbit "
					<< idx << ":" << endl;
			cout << "a set with a group of order "
					<< SaS->target_go << endl;

			SaS->apply_to_self_inverse(Elt /* data */, verbose_level);

			cout << "The given set has the following "
					"stabilizer generators:" << endl;
			SaS->Strong_gens->print_generators();
			

			char fname_gens[1000];

			sprintf(fname_gens, "%s_stab_gens_%d_%d.txt",
					A2->label, Identify_length[i], idx);
			cout << "The generators in source code:" << endl;
			SaS->Strong_gens->print_generators_in_source_code();
			SaS->Strong_gens->print_generators_in_source_code_to_file(
					fname_gens);

			cout << "The given set is : ";
			int_vec_print(cout, SaS->data, SaS->sz);
			cout << endl;

			FREE_OBJECT(SaS);
			FREE_int(data_out);
			}
		FREE_int(Elt);		
		}

	//delete gen;
	
}


int kramer_mesner_test_arc(int len, int *S, void *data, int verbose_level)
{
	kramer_mesner *G = (kramer_mesner *) data;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int pt, p1, p2, i, j, rk;
	int *M; // [3 * n]
	int *base_cols;
	int f_OK = TRUE;
	int n;

	if (f_v) {
		cout << "test_arc testing ";
		int_vec_print(cout, S, len);
		cout << endl;
		}
	if (!G->f_linear) {
		cout << "not a linear action" << endl;
		exit(1);
		}
	n = G->LG->vector_space_dimension;
	//n = G->linear_m;
	if (f_v) {
		cout << "test_arc n= " << n << endl;
		}
	if (len < 3) {
		return TRUE;
		}
	M = NEW_int(3 * n);
	base_cols = NEW_int(n);
	pt = S[len - 1];
	for (i = 0; i < len - 1; i++) {
		if (pt == S[i]) {
			f_OK = FALSE;
			goto the_end;
			}
		}
	for (i = 0; i < len - 1; i++) {
		p1 = S[i];
		for (j = i + 1; j < len - 1; j++) {
			p2 = S[j];
			if (f_vv) {
				cout << "i=" << i << " j=" << j << endl;
				}
			G->F->PG_element_unrank_modified(M, 1, n, p1);
			G->F->PG_element_unrank_modified(M + n, 1, n, p2);
			G->F->PG_element_unrank_modified(M + 2 * n, 1, n, pt);
			if (f_vv) {
				print_integer_matrix_width(cout,
						M, 3, n, n, G->F->log10_of_q);
				}
			rk = G->F->Gauss_simple(M, 3, n,
					base_cols, 0 /* verbose_level */);
			if (rk < 3) {
				f_OK = FALSE;
				goto the_end;
				}
			}
		}

the_end:

	
	FREE_int(M);
	FREE_int(base_cols);
	return f_OK;
}

int kramer_mesner_test_surface(int len, int *S, void *data, int verbose_level)
{
	kramer_mesner *KM = (kramer_mesner *) data;
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "test_surface testing ";
		int_vec_print(cout, S, len);
		cout << endl;
		}

	ret = KM->Surf->test(len, S, verbose_level);
	if (f_v) {
		cout << "test_surface returns " << ret << endl;
		}
	return ret;
}


}}
