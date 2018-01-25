// extending.C
// 
// Anton Betten
// Oct 13 2011
//
// based on BLT/extension_data.C
//

#include "orbiter.h"
#include "discreta.h"
#include "translation_plane.h"


#if 0
extending::extending()
{
	null();
}

extending::~extending()
{
	free();
}

void extending::null()
{
	T = NULL;
	S = NULL;
	ORB = NULL;
	Good_orbits = NULL;

	points = NULL;
	adjacency = NULL;

	f_color_satisfied = NULL;
	color_chosen_at_depth = NULL;
	color_frequency = NULL;
	
	CF = NULL;
	CFI = NULL;
	fp_out = NULL;
	
	f_choice_allocated = FALSE;
}

void extending::free()
{
	if (S) FREE_INT(S);
	if (ORB) delete ORB;
	if (Good_orbits) FREE_INT(Good_orbits);
	
	if (points) FREE_INT(points);
	if (adjacency) FREE_INT(adjacency);
	
	if (f_color_satisfied) FREE_INT(f_color_satisfied);
	if (color_chosen_at_depth) FREE_INT(color_chosen_at_depth);
	if (color_frequency) FREE_INT(color_frequency);
	
	if (CF) delete CF;
	if (CFI) delete CFI;
	if (f_choice_allocated) {
		free_choice();
		}
	null();
}

void extending::init(BYTE *label, INT the_case, 
	translation_plane *T, 
	INT starter_size, INT *set, INT target_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT time[5];
	INT i;
	
	if (f_v) {
		cout << "extending::init verbose_level=" << verbose_level << endl;
		}
	time[0] = os_ticks();

	extending::the_case = the_case;
	extending::T = T;
	extending::starter_size = starter_size;
	extending::target_size = target_size;

	S = NEW_INT(target_size);

	for (i = 0; i < starter_size; i++) {
		S[i] = set[i];
		}

	strcpy(extending::label, label);
	if (f_vv) {
		cout << "extending::init " << label << endl;
		cout << "the_case=" << the_case << endl;
		cout << "starter_size=" << starter_size << endl;
		cout << "target_size=" << target_size << endl;
		cout << "starter: ";
		INT_vec_print(cout, S, starter_size);
		cout << endl;
		}


	// set all dt to zero in case we have an early return:
	dt[0] = 0;
	dt[1] = 0;
	dt[2] = 0;
	dt[3] = 0;
	dt[4] = 0;

	if (f_v) {
		cout << "extending::init done" << endl;
		}
}

void extending::init_candidates(INT nb_candidates, INT *candidates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "extending::init_candidates using known list of candidates" << endl;
		}
	nb_points = nb_candidates;
	points = NEW_INT(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = candidates[i];
		}
}

void extending::init_clique_finder(INT verbose_level)
{
	INT target_depth = target_size - starter_size;
	INT f_maxdepth = FALSE;
	INT maxdepth = 0;
	INT print_interval = (1 << 20);

	CF = new clique_finder;
	CF->init(label, nb_points, 
		target_depth, 
		TRUE, adjacency, 
		FALSE, NULL, 
		print_interval, 
		f_maxdepth, maxdepth, 
		FALSE /* f_save_solutions */, 
		verbose_level - 2);
}

void extending::init_callbacks_and_CFI(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "extending::init_callbacks_and_CFI" << endl;
		}
	
	CF->call_back_clique_found = call_back_clique_found;
	CF->call_back_add_point = call_back_add_point;
	CF->call_back_delete_point = call_back_delete_point;

	//CF->call_back_find_candidates = call_back_find_candidates_sophisticated;
	CF->call_back_find_candidates = call_back_find_candidates;

	CF->call_back_is_adjacent = call_back_is_adjacent;
	//CF->call_back_is_viable = NULL;
	//CF->call_back_after_reduction = call_back_after_reduction;
	CF->call_back_after_reduction = NULL;
	

	CFI = new clique_finder_interface;

	CF->call_back_clique_found_data = CFI;
	CFI->clique_data_local = this;
	CFI->clique_data = T;
}

void extending::do_search(INT verbose_level)
{
	INT f_v = TRUE; //(verbose_level >= 1);

	if (f_v) {
		cout << "extending::do_search" << endl;
		}
	
	CF->backtrack_search(0, 0);

	if (f_v) {
		cout << "extending::do_search done" << endl;
		}
	
}

void extending::extend(BYTE *label, INT the_case, 
	translation_plane *T, 
	INT starter_size, INT *set, INT target_size, 
	const BYTE *stab_ascii, 
	INT &search_steps, INT &decision_steps, INT &nb_sol, 
	INT f_lexorder, 
	INT print_interval, 
	INT f_compute_points_only, ofstream *fp_points_out, 
	INT f_use_points, ifstream *fp_points_in, 
	INT f_has_candidates, INT nb_candidates, INT *candidates, 
	INT f_write_graph_file, 
	INT f_draw_graph, 
	INT f_write_tree, INT f_decision_nodes_only, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT time[6];
	INT i;
	

	if (f_v) {
		cout << "extending::extend" << endl;
		}
	
	init(label, the_case, T, starter_size, set, target_size, verbose_level);

	nb_sol = 0;
	search_steps = 0;
	decision_steps = 0;

	time[0] = os_ticks();



	if (f_vv) {
		cout << "f_lexorder=" << f_lexorder << endl;
		cout << "f_compute_points_only=" << f_compute_points_only << endl;
		cout << "f_use_points=" << f_use_points << endl;
		}
	
	group AUT;
	
	if (!f_use_points) {
		AUT.init(T->A);
		
		if (strlen(stab_ascii)) {
			AUT.init_ascii_coding(stab_ascii);
		
			AUT.decode_ascii(FALSE);
		
			// now strong generators are available
		
		
			}
		else {
			//cout << "trivial group" << endl;
			AUT.init_strong_generators_empty_set();	
			}
	
		AUT.schreier_sims(0);

		if (f_vv) {
			cout << "the automorphism group has order ";
			AUT.print_group_order(cout);
			cout << endl;
		
			cout << "and is strongly generated by " << AUT.SG->len << " elements" << endl;
			}

		}


	time[1] = os_ticks();
	
	if (f_compute_points_only) {
		compute_points(&AUT, 
			f_lexorder, 
			fp_points_out, 
			verbose_level - 1);
		return;
		}
	else {
		
		// next, we compute the set of points,
		// points[nb_points]
		// and we compute the color classes:
		// the color is stored in 
		// point_color[nb_points]
		
		if (!setup(&AUT, 
			f_lexorder, 
			f_use_points, fp_points_in, 
			f_has_candidates, nb_candidates, candidates, 
			verbose_level - 1)) {
			return ;
			}
		}

	if (f_v) {
		cout << "extending::extend after setup() nb_points=" << nb_points << endl;
		}
	time[2] = os_ticks();
	
	if (f_vv) {
		cout << "extending::extend computing adjacencies" << endl;
		}
	compute_adjacencies(verbose_level - 2);
	if (f_vv) {
		cout << "extending::extend computing adjacencies done" << endl;
		}
	
	time[3] = os_ticks();


	INT target_depth = target_size - starter_size;
	INT *f_deleted;

	f_deleted = NEW_INT(nb_points);
	for (i = 0; i < nb_points; i++) {
		f_deleted[i] = FALSE;
		}
	degree_test(nb_points, points, adjacency, target_depth, f_deleted);


	
#if 0
	cout << "Graph with " << nb_points << " vertices:" << endl;
	cout << "points:" << endl;
	INT_vec_print(cout, points, nb_points);
	cout << endl;
	cout << "adjacency:" << endl;
	INT_vec_print(cout, adjacency, ((nb_points * (nb_points - 1)) >> 1));
	cout << endl;
	cout << "coloring:" << endl;
	INT_vec_print(cout, point_color, nb_points);
	cout << endl;


	if (nb_points < 60) {
		latex_degree_sequence(nb_points, points, adjacency);
		latex_adjacency_matrix(nb_points, points, adjacency, point_color, f_deleted);
		}

#endif


	// here we initialize the cliquefinder
	// the adjacency list is in adjacency[].
	// adjacency is a vector of length {n \choose 2},
	// indexed by all unordered pairs of points in lexicographic order
	// That is, the indices correspond to pairs like this: 
	// 0 = {0,1}
	// 1 = {0,2}
	// ...
	// n-2 = {0, n-1}
	// n-1 = {1,2}
	// etc.
	// adjacency[i] is TRUE if the edge is present.

#if 0
	INT f_maxdepth = FALSE;
	INT maxdepth = 0;
#endif

	init_clique_finder(verbose_level);

#if 0
	CF = new clique_finder;
	CF->init(label, nb_points, 
		target_depth, adjacency, 
		print_interval, 
		f_maxdepth, maxdepth, 
		verbose_level - 2);
#endif

	if (f_write_graph_file) {
		if (f_v) {
			cout << "writing graph as xml file" << endl;
			}
		BYTE fname[1000];
		INT point_offset = 0;
		INT f_point_labels = TRUE;
	
	
		sprintf(fname, "graph_%s.xml", label);
		ofstream f(fname);
		write_graph(f, label, point_offset, f_point_labels);
		}

#if 0
	if (f_draw_graph) {
		if (f_v) {
			cout << "drawing graph as mp file" << endl;
			}
		INT point_offset = 0;
		INT f_point_labels = FALSE;
		INT xmax = 600;
		INT ymax = 600;
		INT rad = 30;
		INT f_color_bars = TRUE;
#if 0
		INT *f_deleted;

		f_deleted = NEW_INT(nb_points);
	
		for (i = 0; i < nb_points; i++) {
			f_deleted[i] = FALSE;
			}
#endif
		//draw_graph(label, point_offset, f_point_labels, f_color_bars, f_deleted, xmax, ymax, rad);

		//FREE_INT(f_deleted);
		}
#endif

	
	FREE_INT(f_deleted);


	
#if 0
	INT point_labels[] = {  };
	INT suspicous_points[] = { };
	INT nb_suspicous_points = sizeof(suspicous_points) / sizeof(INT);
	CF->init_point_labels(point_labels);
	CF->init_suspicous_points(nb_suspicous_points, suspicous_points);
	CF->print_suspicous_points();
#endif

	init_callbacks_and_CFI(verbose_level);
	
	time[4] = os_ticks();


	// here we start the backtrack search for finding all cliques:

	//INT f_decision_nodes_only = TRUE;
	
	if (f_write_tree) {
		CF->open_tree_file(label, f_decision_nodes_only);
		}



	// now we start the clique finder process:

	do_search(verbose_level);



	if (f_write_tree) {
		CF->close_tree_file();
		}
	
	search_steps = CF->counter;
	decision_steps = CF->decision_step_counter;
	nb_sol = CF->nb_sol;
	
	time[5] = os_ticks();

	
	for (i = 0; i < 5; i++) {
		dt[i] = time[i + 1] - time[i];
		}
	dt_total = time[5] - time[0];
	
	if (f_v) {
		
		cout << "case = " << case_no << " " << label << " ";
		cout << "nb_sol = " << nb_sol << " ";
		cout << "number of points = " << nb_points << " ";
		cout << "search steps = " << search_steps << " ";
		cout << "decision steps = " << decision_steps << " ";

		cout << "time: " << dt_total << "=";
		for (i = 0; i < 5; i++) {
			cout << dt[i];
			if (i < 5 - 1)
				cout << "+";
			}
		cout << " units ";
		time_check_delta(cout, dt_total);


		//time_check(cout, time0);
		cout << " time total: ";
		time_check(cout, t0);
		cout << endl;
		}

	free();
	null();
	T = NULL;
	Aut = NULL;
	//cout << "after extending::free" << endl;
}

INT extending::setup(group *Aut, 
	INT f_lexorder, 
	INT f_use_points, ifstream *fp, 
	INT f_has_candidates, INT nb_candidates, INT *candidates, 
	INT verbose_level)
{
	INT i, q;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT the_case2;
	
	if (f_v) {
		cout << "extending::setup: case " << the_case << " trying to extend the set ";
		INT_vec_print(cout, S, starter_size);
		cout << endl;
		cout << "f_lexorder=" << f_lexorder << endl;
		cout << "f_use_points=" << f_use_points << endl;
		cout << "f_has_candidates=" << f_has_candidates << endl;
		cout << "target size = " << target_size << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}

	extending::Aut = Aut;
	
	q = T->q;
	
	if (f_use_points) {
		if (f_vv) {
			cout << "extending::setup using points from file" << endl;
			}
		*fp >> the_case2;
		*fp >> nb_points;
		points = NEW_INT(nb_points);
		for (i = 0; i < nb_points; i++) {
			*fp >> points[i];
			}
		//O = Gen->O;
		if (the_case2 != the_case) {
			cout << "extending::setup using points from file, case numbers don't match" << endl;
			cout << "extending::setup: " << endl;
			cout << "the_case2 = " << the_case2 << endl;
			cout << "the_case=" << the_case << endl;
			exit(1);
			}
		}
	else if (f_has_candidates) {
		if (f_vv) {
			cout << "extending::setup using known list of candidates" << endl;
			}
		nb_points = nb_candidates;
		points = NEW_INT(nb_points);
		for (i = 0; i < nb_points; i++) {
			points[i] = candidates[i];
			}
		}
	else {
		if (f_vv) {
			cout << "extending::setup computing orbits on points" << endl;
			}
		//O = T->A->subaction->G.matrix_grp->O;
		ORB = new schreier;
		ORB->init(T->A2);
		if (f_v) {
			cout << "extending::setup: computing orbits in action of degree " << T->A2->degree << endl;
			}
		ORB->init_generators(*Aut->SG);
		ORB->compute_all_point_orbits(0);
		if (f_v) {
			cout << "extending::setup: there are " << ORB->nb_orbits << " orbits on points" << endl;
			ORB->print_orbit_length_distribution(cout);
			}
		if (FALSE && f_vv) {
			ORB->print_and_list_orbits(cout);
			}
		Good_orbits = NEW_INT(ORB->nb_orbits);
		//Pts = NEW_INT(A->degree);


		if (f_v) {
			cout << "extending::setup before find_good_orbits" << endl;
			}
		find_good_orbits(*ORB, 
			Good_orbits, Nb_good_orbits, 
			points, nb_points, f_lexorder, verbose_level);
		if (f_v) {
			cout << "extending::setup after find_good_orbits" << endl;
			}
	
#if 0
		j = 0;
		for (i = 0; i < nb_points; i++) {
			if (points[i] <= S[starter_size - 1]) {
				cout << "waring: point[i] <= S[starter_size - 1]" << endl;
				cout << "points[i]=" << points[i] << endl;
				cout << "S[starter_size - 1]=" << S[starter_size - 1] << endl;
				exit(1);
				}
			}
		nb_points = j;
#endif
		}
	

	if (f_v) {
		cout << "extending::setup found " << nb_points << " good points" << endl;
		//INT_vec_print(cout, points, nb_points);
		//cout << endl;
		}
	if (FALSE && f_vv) {
		cout << "extending::setup points: ";
		INT_vec_print(cout, points, nb_points);
		cout << endl;
		}

	if (nb_points == 0) {
		return FALSE;
		}
	
	INT ret;
	
	if (f_v) {
		cout << "extending::setup before init_colors()" << endl;
		}
	ret = init_colors(verbose_level);
	if (f_v) {
		cout << "extending::setup after init_colors()" << endl;
		}
	
	allocate_color_arrays(verbose_level);

	
	return ret;
}



void extending::compute_points(group *Aut, 
	INT f_lexorder, ofstream *fp, INT verbose_level)
{
	INT i, q;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "extending::compute_points: case " << the_case << " trying to extend the set ";
		INT_vec_print(cout, S, starter_size);
		cout << endl;
		cout << "f_lexorder=" << f_lexorder << endl;
		}

	extending::Aut = Aut;
	
	q = T->q;
		
	ORB = new schreier;
	ORB->init(T->A2);
	if (f_v) {
		cout << "extending::compute_points: computing orbits in action of degree " << T->A2->degree << endl;
		}
	ORB->init_generators(*Aut->SG);
	ORB->compute_all_point_orbits(0);
	if (f_v) {
		cout << "compute_points: there are " << ORB->nb_orbits << " orbits on points" << endl;
		ORB->print_orbit_length_distribution(cout);
		}

	Good_orbits = NEW_INT(ORB->nb_orbits);
	//Pts = NEW_INT(A->degree);


	find_good_orbits(*ORB, 
		Good_orbits, Nb_good_orbits, 
		points, nb_points, f_lexorder, verbose_level - 2);
	
#if 0
	j = 0;
	for (i = 0; i < nb_points; i++) {
		if (points[i] <= S[starter_size - 1]) {
			cout << "waring: point[i] <= S[starter_size - 1]" << endl;
			cout << "points[i]=" << points[i] << endl;
			cout << "S[starter_size - 1]=" << S[starter_size - 1] << endl;
			exit(1);
			}
		}
	nb_points = j;
#endif

	if (f_v) {
		cout << "found " << nb_points << " good points" << endl;
		//INT_vec_print(cout, points, nb_points);
		//cout << endl;
		}
	if (f_vv) {
		cout << "points: ";
		INT_vec_print(cout, points, nb_points);
		cout << endl;
		}
	
	*fp << setw(10) << the_case << " " << setw(10) << nb_points << "  ";
	for (i = 0; i < nb_points; i++) {
		*fp << " " << points[i];
		}
	*fp << endl;
}

void extending::find_good_orbits(schreier &Orb, 
	INT *Good_orbits, INT &nb_good_orbits, 
	INT *&Pts, INT &nb_pts, INT f_lexorder, INT verbose_level)
{
	INT h, i, f, l, u, v, pt, pt0, nb_pts1;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 6);
	
	if (f_v) {
		cout << "extending::find_good_orbits" << endl;
		}
	nb_pts = 0;
	nb_good_orbits = 0;
	if (f_v) {
		cout << "extending::find_good_orbits testing " << Orb.nb_orbits << " orbits" << endl;
		}
	for (h = 0; h < Orb.nb_orbits; h++) {
		f = Orb.orbit_first[h];
		l = Orb.orbit_len[h];
		//i = Orb.orbit[f];
		pt0 = INT_MAX;
		for (i = 0; i < l; i++) {
			pt = Orb.orbit[f + i];
			if (pt < pt0) {
				pt0 = pt;
				}
			}

		if (f_lexorder) {
			// the lexicographic test -- implicit fusion
			if (pt0 <= S[starter_size - 1]) {
				if (f_vvv) {
					cout << "orbit " << h << " containing point " << pt0 
						<< " rejected b/c of implicit fusion, "
						"S[starter_size - 1] = " << S[starter_size - 1];
					cout << " : least element in orbit " << pt0;
					cout << " : highest element in starter " << S[starter_size - 1] << endl;
					//INT_vec_print(cout, Orb.orbit + f, l);
					//cout << endl;
					}
				continue;
				}
			}

		
		S[starter_size] = pt0;
		if (f_v && FALSE) {
			cout << "extending::find_good_orbits before candidate_check_func pt0=" << pt0 << endl;
			}
		if (!(*candidate_check_func)(starter_size + 1, S, candidate_check_data, 0)) {
			if (f_vvv) {
				cout << "orbit " << h << " containing point " << i 
					<< " rejected b/c of test function ";
				cout << " : least element in orbit " << pt0;
				//INT_vec_print(cout, Orb.orbit + f, l);
				cout << endl;
				}
			continue;
			}
		nb_pts += l;
		if (f_vvv) {
			cout << "orbit " << h << " of length " << l 
				<< " works, nb_pts = " << nb_pts;
			cout << " : least element in orbit " << pt0 << endl;
			}
		if (FALSE) {
			INT_vec_print(cout, Orb.orbit + f, l);
			cout << endl;
			}
		Good_orbits[nb_good_orbits++] = h;
		
		}
	if (f_v) {
		cout << "found " << nb_good_orbits 
			<< " good orbits, with " << nb_pts << " points" << endl;
		}
	if (f_vvv) {
		cout << "the good orbits are: " << endl;
		cout << "i : Good_orbits[i] : orbit_len[Good_orbits[i]]" << endl;
		for (h = 0; h < nb_good_orbits; h++) {
			l = Orb.orbit_len[Good_orbits[h]];
			cout << h << " : " << Good_orbits[h] << " : " << l << endl;
			}
		}
		



	Pts = NEW_INT(nb_pts);
	if (f_v) {
		cout << "extracting points contained in good orbits:" << endl;
		}

	nb_pts1 = 0;
	for (i = 0; i < nb_good_orbits; i++) {
		h = Good_orbits[i];
		f = Orb.orbit_first[h];
		l = Orb.orbit_len[h];
		if (f_vvv) {
			cout << "adding the " << l << " points of orbit " << h << endl;
			}
		for (u = 0; u < l; u++) {
			v = Orb.orbit[f + u];



			Pts[nb_pts1++] = v;
			}
		if (f_vvv) {
			cout << "now we have " << nb_pts1 << " points" << endl;
			}
		}
	if (nb_pts1 != nb_pts) {
		cout << "nb_pts1 != nb_pts" << endl;
		cout << "nb_pts1=" << nb_pts1 << endl;
		cout << "nb_pts=" << nb_pts << endl;
		exit(1);
		}
		

}

void extending::compute_adjacencies(INT verbose_level)
{
	INT L = (nb_points * (nb_points - 1)) >> 1;
	INT i, j, k, c1, c2;

	adjacency_length = L;
	adjacency = NEW_INT(adjacency_length);
	
	for (k = 0; k < L; k++) {
		k2ij(k, i, j, nb_points);
		
		c1 = point_color[i];
		c2 = point_color[j];
		if (c1 == c2) {
			adjacency[k] = FALSE;
			continue;
			}

		if (T->check_function_pair(points[i], points[j], verbose_level - 1)) {
			adjacency[k] = TRUE;
			}
		else {
			adjacency[k] = FALSE;
			}
		}
}

void extending::clique_found(INT *current_clique, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *C, i;
	
	if (f_v) {
		cout << "extending::clique_found" << endl;
		cout << "target_size=" << target_size << endl;
		}
	//cout << "CF->target_depth=" << CF->target_depth << endl;
	C = NEW_INT(target_size);
	//cout << "1" << endl;
	for (i = 0; i < starter_size; i++) {
		C[i] = S[i];
		}
	//cout << "2" << endl;

	for (i = 0; i < target_size - starter_size; i++) {
		C[starter_size + i] = points[current_clique[i]];
		}
	if (f_v) {
		cout << "solution " << CF->nb_sol;
		if (f_vv) {
			cout << " ";
			INT_set_print(cout, current_clique, CF->target_depth);
			cout << " ";
			INT_set_print(cout, C, target_size);
			}
		cout << endl;
		}
	
	if (fp_out) {
		//cout << "writing to file" << endl;
		*fp_out << case_no << " ";
		for (i = 0; i < starter_size + CF->target_depth; i++) {
			*fp_out << C[i] << " ";
			}
		*fp_out << endl;
		}
	FREE_INT(C);
}

void extending::add_point(INT pt, 
	INT current_clique_size, INT *current_clique, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c;
	
	c = point_color[pt];
	if (f_color_satisfied[c]) {
		cout << "extending::add_point color already satisfied" << endl;
		exit(1);
		}
	if (c != color_chosen_at_depth[current_clique_size]) {
		cout << "extending::add_point c != color_chosen_at_depth[current_clique_size]" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "add_point " << pt << " at depth " << current_clique_size << " color=" << c << endl;
		}
	f_color_satisfied[c] = TRUE;
}

void extending::delete_point(INT pt, 
	INT current_clique_size, INT *current_clique, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c;
	
	c = point_color[pt];
	if (!f_color_satisfied[c]) {
		cout << "extending::delete_point color not satisfied" << endl;
		exit(1);
		}
	f_color_satisfied[c] = FALSE;
	if (f_v) {
		cout << "delete_point " << pt << " at depth " << current_clique_size << " color=" << c << endl;
		}
}

INT extending::find_candidates(INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c, i, j, c0, c0_freq;
	
	reduced_nb_pts = nb_pts;
	for (i = 0; i < nb_colors; i++) {
		color_frequency[i] = 0;
		}
	for (i = 0; i < nb_pts; i++) {
		c = point_color[pt_list[i]];
		if (c < 0 || c >= nb_colors) {
			cout << "extending::find_candidates color is out of range!" << endl;
			cout << "c=" << c << endl;
			cout << "nb_colors=" << nb_colors << endl;
			cout << "i=" << i << endl;
			cout << "pt_list[i]=" << pt_list[i] << endl;
			print_point_colors();
			exit(1);
			}
		color_frequency[c]++;
		}
	c0 = -1;
	c0_freq = 0;
	for (c = 0; c < nb_colors; c++) {
		if (f_color_satisfied[c]) {
			if (color_frequency[c]) {
				cout << "extension_data::find_candidates satisfied color appears with positive frequency" << endl;
				exit(1);
				}
			}
		else {
			if (color_frequency[c] == 0)
				return 0;
			if (c0 == -1) {
				c0 = c;
				c0_freq = color_frequency[c];
				}
			else {
				if (color_frequency[c] < c0_freq) {
					c0 = c;
					c0_freq = color_frequency[c];
					}
				}
			}
		}
	if (f_v) {
		cout << "extending::find_candidates minimal color is " << c0 << " with frequency " << c0_freq << endl;
		}
	j = 0;
	for (i = 0; i < nb_pts; i++) {
		c = point_color[pt_list[i]];
		if (c == c0) {
			candidates[j++] = pt_list[i];
			}
		}
	if (j != c0_freq) {
		cout << "extending::find_candidates j != c0_freq" << endl;
		exit(1);
		}
	color_chosen_at_depth[current_clique_size] = c0;
	return c0_freq;
}

INT extending::is_adjacent(INT pt1, INT pt2, INT verbose_level)
{
	if (T->check_function_pair(points[pt1], points[pt2], verbose_level - 1)) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

void extending::write_graph(ofstream &ost, BYTE *label, 
	INT point_offset, INT f_point_labels)
{
	INT i, j, d, w, sz;
	
	sz = nb_points;
	cout << "extending::write_graph " 
		<< label << " with " << sz << " points, point_offset=" <<  point_offset
		<< endl;
	w = INT_log10(sz);
	cout << "w=" << w << endl;
	ost << "<GRAPH label=\"" << label << "\" num_pts=" << sz 
		<< " num_colors=" << nb_colors 
		<< " point_offset=" <<  point_offset
		<< " f_point_labels=" <<  f_point_labels
		<< ">" 
		<< endl;
	for (i = 0; i < sz; i++) {
		d = 0;
		for (j = 0; j < sz; j++) {
			if (is_adjacent(i, j, 0))
				d++;
			}
		ost << setw(w) << i + point_offset << " " << setw(w) << d << " ";
		for (j = 0; j < sz; j++) {
			if (is_adjacent(i, j, 0)) {
				ost << setw(w) << j + point_offset << " ";
				}
			}
		ost << endl;
	
		
		}
	ost << endl;
	
	//cout << "calling init_column_table_multi" << endl;
	//init_column_table_multi(0 /* verbose_level */);
	
	for (j = 0; j < nb_colors; j++) {
		d = 0;
		for (i = 0; i < sz; i++) {
			if (point_color[i] == j)
				d++;
			}
		ost << setw(w) << j + point_offset << " " << setw(w) << d << " ";
		for (i = 0; i < sz; i++) {
			if (point_color[i] == j)
				ost << setw(w) << i + point_offset << " ";
			}
		ost << endl;
		}
	
	if (f_point_labels) {
		ost << endl;
		for (i = 0; i < sz; i++) {
			ost << setw(w) << i + point_offset << " " 
				<< setw(6) << points[i] << endl;
			}
		}
	//delete_column_table();

	ost << "</GRAPH>" << endl;
	
}

void extending::degree_test(INT nb_points, INT *points, INT *adjacency, 
	INT clique_size, INT *f_deleted)
{
	INT *M;
	INT *D;
	INT i, j, idx, a;

	M = NEW_INT(nb_points * nb_points);
	D = NEW_INT(nb_points);
	for (i = 0; i < nb_points; i++) {
		D[i] = 0;
		}
	for (i = 0; i < nb_points * nb_points; i++) {
		M[i] = 0;
		}
	idx = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			a = adjacency[idx];
			M[i * nb_points + j] = a;
			M[j * nb_points + i] = a;
			idx++;
			}
		}
	for (i = 0; i < nb_points; i++) {
		for (j = 0; j < nb_points; j++) {
			if (M[i * nb_points + j]) {
				D[i]++;
				}
			}
		}
	for (i = 0; i < nb_points; i++) {
		if (D[i] < clique_size - 1) {
			f_deleted[i] = TRUE;
			}
		}

	FREE_INT(M);
	FREE_INT(D);
}

void extending::latex_degree_sequence(INT nb_points, INT *points, INT *adjacency)
{
	INT *M;
	INT *D;
	INT i, j, idx, a;
	INT point_offset = 0;

	M = NEW_INT(nb_points * nb_points);
	D = NEW_INT(nb_points);
	for (i = 0; i < nb_points; i++) {
		D[i] = 0;
		}
	for (i = 0; i < nb_points * nb_points; i++) {
		M[i] = 0;
		}
	idx = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			a = adjacency[idx];
			M[i * nb_points + j] = a;
			M[j * nb_points + i] = a;
			idx++;
			}
		}
	for (i = 0; i < nb_points; i++) {
		for (j = 0; j < nb_points; j++) {
			if (M[i * nb_points + j]) {
				D[i]++;
				}
			}
		}
	cout << "\\begin{array}{|c|*{" << nb_points << "}{c}|}" << endl;
	cout << "\\hline" << endl;
	cout << "i ";
	for (j = 0; j < nb_points; j++) {
		cout << " & " << j + point_offset;
		}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "d(i) ";
	for (j = 0; j < nb_points; j++) {
		cout << " & " << D[j];
		}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\end{array}" << endl;

	FREE_INT(M);
	FREE_INT(D);
}

void extending::latex_adjacency_matrix(INT nb_points, INT *points, 
	INT *adjacency, INT *point_color, INT *f_deleted)
{
	INT *M;
	INT *nb_deleted;
	INT i, j, h1, h2, f1, f2, l1, l2, idx, a, ii, jj, x, y, c;
	INT point_offset = 0;

	cout << "extending::latex_adjacency_matrix" << endl;
	M = NEW_INT(nb_points * nb_points);
	for (i = 0; i < nb_points * nb_points; i++) {
		M[i] = 0;
		}
	idx = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			a = adjacency[idx];
			M[i * nb_points + j] = a;
			M[j * nb_points + i] = a;
			idx++;
			}
		}

	classify C;

	C.init(point_color, nb_points, FALSE, 0);


	nb_deleted = NEW_INT(C.nb_types);
	for (i = 0; i < C.nb_types; i++) {
		nb_deleted[i] = 0;
		}
	for (i = 0; i < nb_points; i++) {
		if (f_deleted[i]) {
			cout << "calling class_of for deleted vertex " << i << endl;
			c = C.class_of(i);
			cout << "class of vertex " << i << " is " << c << endl;
			nb_deleted[c]++;
			}
		}
	


	cout << "\\begin{array}{c";
	for (h2 = 0; h2 < C.nb_types; h2++) {
		cout << "|*{" << C.type_len[h2] - nb_deleted[h2] << "}{c}";
		}
	cout << "|}" << endl;
	cout << " ";
	for (h2 = 0; h2 < C.nb_types; h2++) {
		f2 = C.type_first[h2];
		l2 = C.type_len[h2];
		for (j = 0; j < l2; j++) {
			jj = f2 + j;
			y = C.sorting_perm_inv[jj];
			if (!f_deleted[y]) {
				cout << " & " << setw(3) << y + point_offset;
				}
			}
		} // next h2
	
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\hline" << endl;
	for (h1 = 0; h1 < C.nb_types; h1++) {
		f1 = C.type_first[h1];
		l1 = C.type_len[h1];
		for (i = 0; i < l1; i++) {
			ii = f1 + i;
			x = C.sorting_perm_inv[ii];
			if (f_deleted[x]) {
				continue;
				}
			cout << setw(3) << x + point_offset << " ";
			for (h2 = 0; h2 < C.nb_types; h2++) {
				f2 = C.type_first[h2];
				l2 = C.type_len[h2];
				for (j = 0; j < l2; j++) {
					jj = f2 + j;
					y = C.sorting_perm_inv[jj];
					if (!f_deleted[y]) {
						a = M[x * nb_points + y];
						cout << " & " << setw(3) << a;
						}
					}
				}
			cout << "\\\\" << endl;
			} // next i
		cout << "\\hline" << endl;
		} // next h1
	cout << "\\end{array}" << endl;


#if 0
	cout << "\\begin{array}{c|*{" << nb_points << "}{c}}" << endl;
	cout << " ";
	for (j = 0; j < nb_points; j++) {
		cout << " & " << j;
		}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;

	for (i = 0; i < nb_points; i++) {
		cout << i << " ";
		for (j = 0; j < nb_points; j++) {
			cout << " & " << M[i * nb_points + j];
			}
		cout << "\\\\" << endl;
		}
	cout << "\\end{array}" << endl;
#endif

	FREE_INT(M);
	FREE_INT(nb_deleted);
}

void extending::after_reduction(clique_finder *CF, 
		INT depth, INT nb_live_points, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "extending::after_reduction" << endl;
		cout << "depth=" << depth << " nb_live_points=" << nb_live_points << endl;
		cout << "current clique: ";
		for (i = 0; i < depth; i++) {
			cout << CF->current_clique[i] << " ";
			}
		cout << endl;
		cout << "live points: ";
		for (i = 0; i < nb_live_points; i++) {
			cout << CF->pt_list[i] << " ";
			}
		cout << endl;
		}
#if 0
	if (depth <= 1) {
		INT *f_deleted;

		f_deleted = NEW_INT(nb_points);
		for (i = 0; i < nb_points; i++) {
			f_deleted[i] = TRUE;
			}
		for (i = 0; i < nb_live_points; i++) {
			f_deleted[CF->pt_list[i]] = FALSE;
			}

		INT point_offset = 0;
		INT f_point_labels = FALSE;
		INT xmax = 1000;
		INT ymax = 1000;
		INT rad = 30;
		INT f_color_bars = TRUE;

		BYTE label[1000];

		sprintf(label, "%s_cf", CF->label);
		for (i = 0; i < depth; i++) {
			sprintf(label + strlen(label), "_%ld", CF->current_clique[i]);
			}

		draw_graph(label, point_offset, f_point_labels, f_color_bars, f_deleted, xmax, ymax, rad);

		FREE_INT(f_deleted);
		}
#endif
	
}

INT extending::init_colors(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 6);
	INT i, j, h, q, a, b, idx, offset;
	INT k, n;
	INT *v;
	
	if (f_v) {
		cout << "extending::init_colors" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "target_size = " << target_size << endl;
		}
	q = T->q;
	k = T->k;
	n = T->n;
	
	N = i_power_j(q, k); // = order = target_size - 1

	if (f_v) {
		cout << "N=" << N << endl;
		}
	if (N != T->order) {
		cout << "extending::init_colors N != T->order" << endl;
		cout << "N=" << N << endl;
		cout << "T->order=" << T->order << endl;
		exit(1);
		}

	if (f_v) {
		cout << "extending::init_colors allocating things:" << endl;
		}
	v = NEW_INT(n);
	open_colors = NEW_INT(N);
	open_colors_inv = NEW_INT(N);
	point_color = NEW_INT(nb_points);
	solution_point_color = NEW_INT(target_size);
	
	for (h = 0; h < starter_size; h++) {
		if (f_vv) {
			cout << "extending::init_colors color of start element " << h << endl;
			}
		a = S[h];
		T->Grass->unrank_INT(a, 0);
		if (h == 1) {
			if (f_vvv) {
				cout << "extending::init_colors starter point " << setw(4) << 1 << " pt=" << a << ":" << endl;
				INT_matrix_print(T->Grass->M, k, n);
				cout << "has no color" << endl;
				}
			b = -1;
			}
		else {
			AG_element_rank(q, T->Grass->M + k, 1, k, b);
			if (f_vvv) {
				cout << "extending::init_colors starter point " << setw(4) << h << " pt=" << a << ":" << endl;
				INT_matrix_print(T->Grass->M, k, n);
				cout << "color=" << b << endl;
				}
			}
		solution_point_color[h] = b;
		}
	if (f_vv) {
		cout << "extending::init_colors solution_point_color:" << endl;
		INT_vec_print(cout, solution_point_color, starter_size);
		cout << endl;
		}
	
	for (h = 0; h < nb_points; h++) {
		a = points[h];
		T->Grass->unrank_INT(a, 0);
		AG_element_rank(q, T->Grass->M + k, 1, k, b);
		if (f_vvv) {
			cout << setw(4) << h << " pt=" << a << ":" << endl;
			INT_matrix_print(T->Grass->M, k, n);
			cout << "color=" << b << " = ";
			T->Grass->unrank_INT(a, 0);
			INT_vec_print(cout, T->Grass->M + k, k);
			cout << endl;
			}
		point_color[h] = b;
		}
	if (f_vvv) {
		cout << "extending::init_colors raw point colors:" << endl;
		INT_vec_print(cout, point_color, nb_points);
		cout << endl;
		}
	nb_colors = N - (starter_size - 1);
	if (f_v) {
		cout << "extending::init_colors nb_colors=" << nb_colors << endl;
		}
	j = 0;
	for (i = 0; i < N; i++) {
		if (INT_vec_search_linear(solution_point_color, starter_size, i, idx)) {
			continue;
			}
		open_colors[j++] = i;
		}
	if (j != nb_colors) {
		cout << "extending::init_colors j != nb_colors" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "extending::init_colors The open colors are:" << endl;
		INT_vec_print(cout, open_colors, nb_colors);
		cout << endl;
		}
	for (; j < N; j++) {
		if (j > nb_colors) {
			offset = 1;
			}
		else {
			offset = 0;
			}
		open_colors[j] = solution_point_color[j - nb_colors + offset];
		}
	if (f_vv) {
		cout << "extending::init_colors The open colors are (filled up to be a permutation):" << endl;
		INT_vec_print(cout, open_colors, N);
		cout << endl;
		}
	for (i = 0; i < N; i++) {
		open_colors_inv[i] = -1;
		}
	
	for (i = 0; i < N; i++) {
		a = open_colors[i];
		if (open_colors_inv[a] != -1) {
			cout << "The open colors are (filled up to be a permutation):" << endl;
			INT_vec_print(cout, open_colors, N);
			cout << endl;
			cout << "open_colors is not a permutation" << endl;
			exit(1);
			}
		open_colors_inv[a] = i;
		}
	if (f_vvv) {
		cout << "I : open colors[i] : open colors inv[i]" << endl;
		for (i = 0; i < N; i++) {
			if (i == nb_colors) {
				cout << "======" << endl;
				}
			AG_element_unrank(q, v, 1, k, open_colors[i]);
			cout << setw(4) << i << " : " << setw(4) << open_colors[i] << " : " << setw(4) << open_colors_inv[i] << " : ";
			INT_vec_print(cout, v, k);
			cout << endl;
			}
		}

	// we replace the open colors by their position in open_color,
	// so that the open colors are labeled 0, ..., nb_colors - 1
	for (h = 0; h < nb_points; h++) {
		a = point_color[h];
		b = open_colors_inv[a];
		if (b >= nb_colors) {
			cout << "extending::init_colors point " << h << " has an illegal color" << endl;
			cout << "point_color[h] = " << a << endl;
			cout << "nb_colors=" << nb_colors << endl;
			cout << "starter:" << endl;
			INT_vec_print(cout, S, starter_size);
			cout << endl;
			cout << "points:" << endl;
			INT_vec_print(cout, points, nb_points);
			cout << endl;
			exit(1);
			}
		point_color[h] = b;
		}
	if (f_vvv) {
		cout << "extending::init_colors point colors after relabeling:" << endl;
		INT_vec_print(cout, point_color, nb_points);
		cout << endl;
		}
	if (f_vvv) {
		cout << "extending::init_colors point colors:" << endl;
		print_point_colors();
		}

	classify C;

	C.init(point_color, nb_points, TRUE, 0);
	if (f_v) {
		cout << "extending::init_colors sizes of color classes:" << endl;
		C.print(FALSE /*f_backwards*/);
		}

	INT minimal_type;
	INT minimal_type_multiplicity;
	INT f, l;

	f = C.second_type_first[0];
	l = C.second_type_len[0];
	minimal_type = C.second_sorting_perm_inv[f + 0];
	minimal_type_multiplicity = C.type_len[minimal_type];
	if (f_v) {
		cout << "extending::init_colors minimal type is " << minimal_type << endl;
		cout << "extending::init_colors minimal_type_multiplicity " << minimal_type_multiplicity << endl;
		}

	FREE_INT(v);
	
	if (minimal_type_multiplicity == 0) {
		return FALSE;
		}
		
	allocate_choice(); // for find_candidates_sophisticated

	if (f_v) {
		cout << "extending::init_colors done" << endl;
		}
	
	return TRUE;
}

void extending::allocate_color_arrays(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "extending::allocate_color_arrays" << endl;
		}
	f_color_satisfied = NEW_INT(nb_colors);
	color_chosen_at_depth = NEW_INT(nb_colors);
	color_frequency = NEW_INT(nb_colors);
	
	for (i = 0; i < nb_colors; i++) {
		f_color_satisfied[i] = FALSE;
		}
	if (f_v) {
		cout << "extending::allocate_color_arrays done" << endl;
		}
	
}


void extending::allocate_choice()
{
	f_choice_allocated = TRUE;
	remaining_colors = NEW_INT(N);
	remaining_color_idx = NEW_INT(N);
	live_points_by_color = NEW_INT(N * nb_points);
	f_live_points_by_color_active = NEW_INT(N * nb_points);
	nb_live_points_by_color = NEW_INT(N);
	nb_live_points_by_color_active = NEW_INT(N);
	buddy_table = NEW_INT(nb_points * N);
	f_point_is_active = NEW_INT(nb_points);
}

void extending::free_choice()
{
	f_choice_allocated = FALSE;
	FREE_INT(remaining_colors);
	FREE_INT(remaining_color_idx);
	FREE_INT(live_points_by_color);
	FREE_INT(f_live_points_by_color_active);
	FREE_INT(nb_live_points_by_color);
	FREE_INT(nb_live_points_by_color_active);
	FREE_INT(buddy_table);
	FREE_INT(f_point_is_active);
}

void extending::print_point_colors()
{
	INT u;
	cout << "u : point_color[u]" << endl;
	for (u = 0; u < nb_points; u++) {
		cout << setw(3) << u << " : " << setw(3) << point_color[u] << endl;
		}
}



INT extending::find_candidates_sophisticated(INT current_clique_size, 
	INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level)
{
	//verbose_level = 2;

	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, a, c, rc, ca, b, bb, idx, a2;
	INT f_point_set_has_changed;

	if (f_v) {
		cout << "extending::find_candidates_sophisticated" << endl;
		}	

	nb_remaining_colors = 0;
	for (i = 0; i < nb_colors; i++) {
		remaining_color_idx[i] = -1;
		}
	for (i = 0; i < nb_colors; i++) {
		if (!f_color_satisfied[i]) {
			remaining_colors[nb_remaining_colors] = i;
			remaining_color_idx[i] = nb_remaining_colors;
			nb_remaining_colors++;
			}
		}
	if (f_vv) {
		cout << "The " << nb_remaining_colors << " remaining colors are:" << endl;
		INT_vec_print(cout, remaining_colors, nb_remaining_colors);
		cout << endl;
		}
	for (i = 0; i < nb_remaining_colors; i++) {
		nb_live_points_by_color[i] = 0;
		}
	for (i = 0; i < nb_pts; i++) {
		a = pt_list[i];
		c = point_color[a];
		rc = remaining_color_idx[c];
		if (f_vv) {
			cout << "The " << i << "-th point " << a << " has color " << c << " which is remaining color " << rc << endl;
			}
		live_points_by_color[rc * nb_points + nb_live_points_by_color[rc]] = i;
		nb_live_points_by_color[rc]++;
		}
	if (f_vv) {
		cout << "the points by remaining colors are:" << endl;
		for (c = 0; c < nb_remaining_colors; c++) {
			cout << setw(3) << c << " : " << setw(3) << nb_live_points_by_color[c] << " : ";
			INT_vec_print(cout, live_points_by_color + c * nb_points, nb_live_points_by_color[c]);
			cout << endl;
			}
		}

	for (i = 0; i < nb_pts; i++) {
		f_point_is_active[i] = TRUE;
		}
	nb_active_points = nb_pts;
	for (c = 0; c < nb_remaining_colors; c++) {
		for (j = 0; j < nb_live_points_by_color[c]; j++) {
			f_live_points_by_color_active[c * nb_points + j] = TRUE;
			}
		nb_live_points_by_color_active[c] = nb_live_points_by_color[c];
		}

	if (f_vv) {
		cout << "trying to establish buddy table:" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_remaining_colors; j++) {
			buddy_table[i * nb_remaining_colors + j] = -1;
			}
		}
	while (TRUE) {
		f_point_set_has_changed = FALSE;
		
		if (f_v) {
			cout << "buddy table before pass:" << endl;
			INT_matrix_print(buddy_table, nb_pts, nb_remaining_colors);
			}
		for (i = 0; i < nb_pts; i++) {
			if (!f_point_is_active[i]) {
				continue; // don't worry about dead points
				}
			a = pt_list[i];
			ca = remaining_color_idx[point_color[a]];
			for (c = 0; c < nb_remaining_colors; c++) {
				if (c == ca) {
					continue;
					}
				b = buddy_table[i * nb_remaining_colors + c];
				if (b >= 0 && f_live_points_by_color_active[c * nb_points + b]) {
					// we are good
					continue;
					}
				if (f_vv) {
					if (b >= 0) {
						cout << "point " << i << " need to find new buddy replacing " << live_points_by_color[c * nb_points + b] << endl;
						}
					}
				// we need to find a buddy:
				for (bb = b + 1; bb < nb_live_points_by_color[c]; bb++) {
					if (!f_live_points_by_color_active[c * nb_points + bb]) { 
						continue;
						}
					idx = live_points_by_color[c * nb_points + bb];
					a2 = pt_list[idx];
					// test if a2 could be the buddy of a
					if (!CF->s_ij(a, a2)) {
						continue; // no, not adjacent
						}
					// yes, we found a new buddy:
					buddy_table[i * nb_remaining_colors + c] = bb;
					break;
					}
				if (bb == nb_live_points_by_color[c]) {
					// The point is dead, 
					// since it has no active live point in color class bb.
					// We eliminate the point.
					
					f_point_is_active[i] = FALSE;
					nb_active_points--;
					f_point_set_has_changed = TRUE;
					if (f_vv) {
						cout << "point " << i << " was eliminated, remaining number of active points is " << nb_active_points << endl;
						}
					if (nb_live_points_by_color_active[ca] == 1) {
						if (f_vv) {
							cout << "The last active point in color class " << ca << " was eliminated. We are dead" << endl;
							}
						return 0;
						}
					nb_live_points_by_color_active[ca]--;
					if (!INT_vec_search(live_points_by_color + ca * nb_points, nb_live_points_by_color[ca], i, idx)) {
						cout << "could not find point a in its color class, something is wrong" << endl;
						exit(1);
						}
					f_live_points_by_color_active[ca * nb_points + idx] = FALSE;
					}

				if (!f_point_is_active[i]) {
					// we don't need to continue with a dead point
					break;
					}
				// otherwise, keep checking the buddy table
				} // for c
			
			} // for i
		if (!f_point_set_has_changed) {
			// buddy table is up-to-date and OK
			break;
			}
		// otherwise, buddy table needs to be checked again
		} // while
	
	if (f_vv) {
		cout << "buddy table is OK" << endl;
		if (f_v) {
			cout << "buddy table:" << endl;
			INT_matrix_print(buddy_table, nb_pts, nb_remaining_colors);
			}
		cout << "The " << nb_active_points << " active points are:" << endl;
		for (i = 0; i < nb_pts; i++) {
			if (!f_point_is_active[i]) {
				continue; // don't worry about dead points
				}
			a = pt_list[i];
			cout << i << " (" << a << ") ";
			}
		cout << endl;
		
		for (i = 0; i < nb_pts; i++) {
			if (!f_point_is_active[i]) {
				continue; // don't worry about dead points
				}
			cout << setw(3) << i << " : ";
			a = pt_list[i];
			ca = remaining_color_idx[point_color[a]];
			for (c = 0; c < nb_remaining_colors; c++) {
				if (c == ca) {
					cout << "*** ";
					continue;
					}
				b = buddy_table[i * nb_remaining_colors + c];
				idx = live_points_by_color[c * nb_points + b];
				cout << setw(3) << idx << " ";
				if (!f_point_is_active[idx]) {
					cout << endl << "The buddy is not active, something is wrong" << endl;
					exit(1);
					}
				a2 = pt_list[idx];
				if (!CF->s_ij(a, a2)) {
					cout << "a and a2 are not adjacent, something is wrong" << endl;
					exit(1);
					}
				} // next c
			cout << endl;
			} // next i
		}
	INT c0, c0_freq;
	
	c0 = -1;
	c0_freq = 0;
	for (c = 0; c < nb_remaining_colors; c++) {
		if (nb_live_points_by_color_active[c] == 0) {
			cout << "nb_live_points_by_color_active[c] == 0, something is wrong" << endl;
			}
		if (c0 == -1) {
			c0 = c;
			c0_freq = nb_live_points_by_color_active[c];
			}
		else {
			if (nb_live_points_by_color_active[c] < c0_freq) {
				c0 = c;
				c0_freq = nb_live_points_by_color_active[c];
				}
			}
		}
	if (f_v) {
		cout << "extending::find_candidates_sophisticated minimal color is " << remaining_colors[c0] << " with frequency " << c0_freq << endl;
		}
	j = 0;
	for (i = 0; i < nb_live_points_by_color[c0]; i++) {
		idx = live_points_by_color[c0 * nb_points + i];
		if (!f_live_points_by_color_active[c0 * nb_points + i]) {
			continue;
			}
		candidates[j++] = pt_list[idx];
		}
	if (j != c0_freq) {
		cout << "extending::find_candidates_sophisticated j != c0_freq" << endl;
		exit(1);
		}
	color_chosen_at_depth[current_clique_size] = remaining_colors[c0];

	reduced_nb_pts = 0;
	for (i = 0; i < nb_pts; i++) {
		if (!f_point_is_active[i]) {
			continue; // don't worry about dead points
			}
		a = pt_list[i];
		if (reduced_nb_pts != i) {
			b = pt_list[reduced_nb_pts];
			pt_list[reduced_nb_pts] = a;
			pt_list[i] = b;
			if (pt_list_inv) {
				pt_list_inv[a] = reduced_nb_pts;
				pt_list_inv[b] = i;
				}
			}
		reduced_nb_pts++;
		}
	if (reduced_nb_pts != nb_active_points) {
		cout << "reduced_nb_pts != nb_active_points" << endl;
		exit(1);
		}
	return c0_freq;

}


void call_back_clique_found(clique_finder *CF, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//cout << "extending.C: call_back_clique_found" << endl;
	//cout << "CF->call_back_clique_found_data=" << CF->call_back_clique_found_data << endl;
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;
	extending *E = (extending *) CFI->clique_data_local;
	E->CF = CF;
	return E->clique_found(CF->current_clique, verbose_level);
}

void call_back_add_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;
	extending *E = (extending *) CFI->clique_data_local;
	E->add_point(pt, current_clique_size, current_clique, verbose_level);
}

void call_back_delete_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;
	extending *E = (extending *) CFI->clique_data_local;
	E->delete_point(pt, current_clique_size, current_clique, verbose_level);
}

INT call_back_find_candidates(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;
	extending *E = (extending *) CFI->clique_data_local;
	INT ret;
	
	ret = E->find_candidates(current_clique_size, current_clique, 
		nb_pts, reduced_nb_pts, pt_list, pt_list_inv, candidates, verbose_level);
	
	return ret;
}

INT call_back_find_candidates_sophisticated(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;
	extending *E = (extending *) CFI->clique_data_local;
	INT ret;
	
	ret = E->find_candidates_sophisticated(current_clique_size, current_clique, 
		nb_pts, reduced_nb_pts, pt_list, pt_list_inv, candidates, verbose_level);
	
	return ret;
}

INT call_back_is_adjacent(clique_finder *CF, 
	INT pt1, INT pt2, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	extending *E = (extending *) CF->call_back_clique_found_data;
	return E->is_adjacent(pt1, pt2, verbose_level);
}

void call_back_after_reduction(clique_finder *CF, 
		INT depth, INT nb_live_points, INT verbose_level)
{
	//verbose_level = 2;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "call_back_after_reduction" << endl;
		}
	extending *E = (extending *) CF->call_back_clique_found_data;
	E->after_reduction(CF, depth, nb_live_points, verbose_level);
}





INT clique_setup(INT *starter, INT starter_size, 
	INT *point_set, INT point_set_size, 
	INT target_size, 
	void *data, 
	void *&data_local, INT verbose_level)
{
	//verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	translation_plane *T = (translation_plane *) data; 
	extending *E;
	INT i;
	
	if (f_v){
		cout << "clique_setup" << endl;
		}
	E = new extending;
	E->starter_size = starter_size;
	E->target_size = target_size;
	E->T = T;
	
	if (f_v) {
		cout << "copying starter:" << endl;
		}
	E->S = NEW_INT(target_size);
	for (i = 0; i < starter_size; i++) {
		E->S[i] = starter[i];
		}

	if (f_v) {
		cout << "copying points" << endl;
		}
	E->nb_points = point_set_size;
	E->points = NEW_INT(point_set_size);
	for (i = 0; i < point_set_size; i++) {
		E->points[i] = point_set[i];
		}

	INT ret;
	
	if (f_v) {
		cout << "clique_setup before init_colors()" << endl;
		}
	ret = E->init_colors(verbose_level - 1);
	if (f_v) {
		cout << "clique_setup after init_colors()" << endl;
		}
	
	E->f_color_satisfied = NEW_INT(E->nb_colors);
	E->color_chosen_at_depth = NEW_INT(E->nb_colors);
	E->color_frequency = NEW_INT(E->nb_colors);
	
	for (i = 0; i < E->nb_colors; i++) {
		E->f_color_satisfied[i] = FALSE;
		}
	

	


	data_local = E;
	if (f_v) {
		cout << "clique_setup done" << endl;
		}
	return ret;
}



void clique_cleanup(void *data, void *data_local, INT verbose_level)
{
	extending *E = (extending *) data_local;

	E->CF = NULL;
	//cout << "clique_cleanup" << endl;
	delete E;
}

void translation_plane_do_extend(translation_plane *T, INT starter_size, 
	INT r, INT m, INT f_casenumbers, 
	INT f_lexorder,
	INT print_interval,  
	INT f_compute_points_only, INT f_use_points, 
	INT f_read_candidates_file, const BYTE *candidates_fname, 
	INT f_mem_dump_by_size, INT f_mem_dump, 
	INT f_write_graph_file, 
	INT f_draw_graph, 
	INT f_write_tree, INT f_decision_nodes_only, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i, the_case;
	BYTE fname_points[1000];
	ofstream *fp_points_out;
	ifstream *fp_points_in;
	BYTE fname[1000];
	BYTE label[1000];
	INT order = T->order;
	
	if (f_v) {
		cout << "translation_plane::do_extend r=" << r << " f_lexorder=" << f_lexorder << endl;
		cout << "f_casenumbers=" << f_casenumbers << endl;
		cout << "f_lexorder=" << f_lexorder << endl;
		cout << "print_interval=" << print_interval << endl;
		cout << "f_compute_points_only=" << f_compute_points_only << endl;
		cout << "f_use_points=" << f_use_points << endl;
		cout << "f_read_candidates_file=" << f_read_candidates_file << endl;
		cout << "f_write_graph_file=" << f_write_graph_file << endl;
		cout << "f_draw_graph=" << f_draw_graph << endl;
		cout << "f_write_tree=" << f_write_tree << endl;
		cout << "f_decision_nodes_only=" << f_decision_nodes_only << endl;
		}
	if (f_compute_points_only) {
		sprintf(fname_points, "spread_%ld_%ld_%ld_%ld_points.txt", order, starter_size, r, m);
		fp_points_out = new ofstream;
		fp_points_out->open(fname_points);
		}
	if (f_use_points) {
		sprintf(fname_points, "spread_%ld_%ld_%ld_%ld_points.txt", order, starter_size, r, m);
		fp_points_in = new ifstream;
		fp_points_in->open(fname_points);
		}
	if (f_casenumbers) {
		sprintf(fname, "input_%ld.txt", r);
		}
	else {
		sprintf(fname, "spread_%ld_lvl_%ld", order, starter_size);
		}
	
	if (f_v) {
		cout << "translation_plane::do_extend reading file " << fname /*gen->extend_fname*/ << endl;
		}
	
	data_file D;

	D.read(fname, f_casenumbers, verbose_level);

	if (f_read_candidates_file) {
		if (f_v) {
			cout << "translation_plane::do_extend reading candidates file" << endl;
			}
		D.read_candidates(candidates_fname, verbose_level);
		if (f_v) {
			cout << "translation_plane::do_extend reading candidates file done" << endl;
			}
		}


	for (i = 0; i < D.nb_cases; i++) {
		if (D.set_sizes[i] != starter_size) {
			cout << "D.set_sizes[i] != starter_size" << endl;
			exit(1);
			}
		}
	
	
	INT search_steps, decision_steps, nb_sol;
	ofstream *fp_out, *fp_summary, *fp_success;
	BYTE fname_out[1000];
	BYTE fname_summary[1000];
	BYTE fname_success[1000];
	
	
	sprintf(fname_out, "extend_spread_%ld_%ld_%ld_%ld.txt", 
		order, starter_size, r, m);
	sprintf(fname_summary, "extend_spread_%ld_%ld_%ld_%ld.summary", 
		order, starter_size, r, m);
	sprintf(fname_success, "extend_spread_%ld_%ld_%ld_%ld.success", 
		order, starter_size, r, m);
	
	if (f_compute_points_only) {
		fp_out = NULL;
		fp_summary = NULL;
		fp_success = NULL;
		}
	else {
		fp_out = new ofstream;
		fp_summary = new ofstream;
	
		fp_out->open(fname_out);
		fp_summary->open(fname_summary);
		}
	
	if (f_v) {
		cout << "translation_plane_do_extend extending all " << D.nb_cases << " solutions congruent to " 
			<< r << " mod " << m << endl;
		}
	
	for (i = 0; i < D.nb_cases; i++) {
		if (f_casenumbers || (i % m) == r) {
			if (f_vvv && !f_compute_points_only) {
				cout << "##################################################################################################" << endl;
				}
			if (f_casenumbers) {
				the_case = D.casenumbers[i];
				}
			else {
				the_case = i;
				}
			if (f_vv) {
				cout << "translation_plane_do_extend  extending solution " 
					<< the_case << " / " << D.nb_cases << " : ";
				INT_vec_print(cout, D.sets[i], starter_size);
				cout << " : ago = " << D.Ago_ascii[i];
				cout << endl;
				}
			{
			extending E;
			INT target_size = order + 1;
			
			E.f_candidate_check_func = TRUE;
			E.candidate_check_func = translation_plane_check_conditions;
			E.candidate_check_data = (void *) T;


			E.fp_out = fp_out;
			E.fp_summary = fp_summary;
			
			sprintf(label, "spread_%ld_%ld_%ld", order, starter_size, the_case);

			E.case_no = the_case;
			
			// here we call extension_data::extend()
			// to perform the clique finding

			INT f_has_candidates = FALSE;
			INT nb_candidates = 0;
			INT *candidates = NULL;

			if (f_read_candidates_file) {
				f_has_candidates = TRUE;
				nb_candidates = D.nb_candidates[i];
				candidates = D.candidates[i];
				}
			E.extend(label, the_case, T, 
				starter_size, D.sets[i], target_size, 
				D.Aut_ascii[i], 
				search_steps, decision_steps, nb_sol, 
				f_lexorder, 
				print_interval, 
				f_compute_points_only, fp_points_out, 
				f_use_points, fp_points_in, 
				f_has_candidates, nb_candidates, candidates, 
				f_write_graph_file, 
				f_draw_graph, 
				f_write_tree, f_decision_nodes_only, 
				verbose_level - 3);

			
			if (!f_compute_points_only) {
				*E.fp_summary << the_case << " " 
					<< nb_sol << " " 
					<< search_steps << " "
					<< decision_steps << " "
					<< E.nb_points << " "
					<< E.dt[0] << " "
					<< E.dt[1] << " "
					<< E.dt[2] << " "
					<< E.dt[3] << " "
					<< E.dt[4] << " "
					<< E.dt_total << endl;
				}
			} // delete E
			
			if (f_mem_dump_by_size) {
				registry_dump_sorted_by_size();
				}
			if (f_mem_dump) {
				registry_dump();
				}
			
			}
		}
	if (!f_compute_points_only) {
		*fp_summary << "-1 ";
		time_check(*fp_summary, t0);
		*fp_summary << endl;
		*fp_out << "-1 ";
		time_check(*fp_out, t0);
		*fp_out << endl;
		fp_out->close();
		fp_summary->close();

		fp_success = new ofstream;
		fp_success->open(fname_success);
		*fp_success << "job completed successfully" << endl;
		fp_success->close();
		delete fp_success;
		}

	if (f_compute_points_only) {
		*fp_points_out << "-1" << endl;
		fp_points_out->close();
		delete fp_points_out;
		cout << "written file " << fname_points 
			<< " of size " << file_size(fname_points) << endl;
		print_line_of_number_signs();
		}
	if (f_use_points) {
		fp_points_in->close();
		delete fp_points_in;
		}
	
	if (!f_compute_points_only) {
		delete fp_out;
		delete fp_summary;
		}
	
	
}

void translation_plane_extend_simple(translation_plane *T, INT starter_size, 
	INT *starter, 
	INT f_lexorder,
	INT f_write_graph_file, 
	INT f_draw_graph, 
	INT f_write_tree, INT f_decision_nodes_only, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT i, the_case;
	BYTE prefix[1000];
	INT order = T->order;
	
	if (f_v) {
		cout << "translation_plane_extend_simple" << endl;
		cout << "f_lexorder=" << f_lexorder << endl;
		cout << "f_write_graph_file=" << f_write_graph_file << endl;
		cout << "f_draw_graph=" << f_draw_graph << endl;
		cout << "f_write_tree=" << f_write_tree << endl;
		cout << "f_decision_nodes_only=" << f_decision_nodes_only << endl;
		}
	
	INT search_steps, decision_steps, nb_sol;
	ofstream *fp_out, *fp_summary, *fp_success;
	BYTE fname_out[1000];
	BYTE fname_summary[1000];
	BYTE fname_success[1000];
	
	
	prefix[0] = 0;
	for (i = 0; i < starter_size; i++) {
		sprintf(prefix + strlen(prefix), "%ld", starter[i]);
		if (i < starter_size - 1) {
			strcat(prefix, "_");
			}
		}
	sprintf(fname_out, "extend_%s.txt", prefix);
	sprintf(fname_summary, "extend_%s.summary", prefix);
	sprintf(fname_success, "extend_%s.success", prefix);
	
	fp_out = new ofstream;
	fp_summary = new ofstream;
	
	fp_out->open(fname_out);
	fp_summary->open(fname_summary);
	
	if (f_v) {
		cout << "translation_plane_extend_simple" << endl;
		}
	
	the_case = 0;
			{
			extending E;
			INT target_size = order + 1;
			INT print_interval = ONE_MILLION;
			
			E.f_candidate_check_func = TRUE;
			E.candidate_check_func = translation_plane_check_conditions;
			E.candidate_check_data = (void *) T;


			E.fp_out = fp_out;
			E.fp_summary = fp_summary;
			

			E.case_no = the_case;
			
			// here we call extension_data::extend()
			// to perform the clique finding

			INT f_has_candidates = FALSE;
			INT nb_candidates = 0;
			INT *candidates = NULL;

			E.extend(prefix, the_case, T, 
				starter_size, starter, target_size, 
				"" /*D.Aut_ascii[i]*/, 
				search_steps, decision_steps, nb_sol, 
				f_lexorder, 
				print_interval, 
				FALSE /*f_compute_points_only*/, NULL /*fp_points_out*/, 
				FALSE /*f_use_points*/, NULL /*fp_points_in*/, 
				f_has_candidates, nb_candidates, candidates, 
				f_write_graph_file, 
				f_draw_graph, 
				f_write_tree, f_decision_nodes_only, 
				verbose_level - 3);

			
				*E.fp_summary << the_case << " " 
					<< nb_sol << " " 
					<< search_steps << " "
					<< decision_steps << " "
					<< E.nb_points << " "
					<< E.dt[0] << " "
					<< E.dt[1] << " "
					<< E.dt[2] << " "
					<< E.dt[3] << " "
					<< E.dt[4] << " "
					<< E.dt_total << endl;
			} // delete E
			
			
		*fp_summary << "-1 ";
		time_check(*fp_summary, t0);
		*fp_summary << endl;
		*fp_out << "-1 ";
		time_check(*fp_out, t0);
		*fp_out << endl;
		fp_out->close();
		fp_summary->close();

		fp_success = new ofstream;
		fp_success->open(fname_success);
		*fp_success << "job completed successfully" << endl;
		fp_success->close();
		delete fp_success;

		delete fp_out;
		delete fp_summary;
	
	
}
#endif


#if 0
void translation_plane_init_clique(translation_plane *T, generator *gen, INT clique_level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	clique_finder_interface *CFI;

	if (f_v) {
		cout << "translation_plane_init_clique" << endl;
		}
	CFI = new clique_finder_interface;
	gen->CFI = CFI;

	CFI->clique_level = clique_level;
	CFI->f_clique_setup_func = TRUE;
	CFI->clique_setup_func = clique_setup;
	CFI->clique_data = T;
	CFI->clique_cleanup_func = clique_cleanup;

	CFI->call_back_clique_found = call_back_clique_found;
	CFI->call_back_add_point = call_back_add_point;
	CFI->call_back_delete_point = call_back_delete_point;
	CFI->call_back_find_candidates = call_back_find_candidates;
	//CFI->call_back_clique_found_data will be set to 
	//CFI->clique_data_local in oracle_downstep.C
}

#endif





