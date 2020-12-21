// classify_double_sixes.cpp
// 
// Anton Betten
//
// October 10, 2017
//
// based on surface_classify_wedge.cpp started September 2, 2016
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

classify_double_sixes::classify_double_sixes()
{
	null();
}

classify_double_sixes::~classify_double_sixes()
{
	freeself();
}

void classify_double_sixes::null()
{
	q = 0;
	F = NULL;
	Surf_A = NULL;
	Surf = NULL;

	LG = NULL;

	A2 = NULL;
	AW = NULL;
	Elt0 = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	Surf = NULL;
	SG_line_stab = NULL;
	Neighbors = NULL;
	Neighbor_to_line = NULL;
	Neighbor_to_klein = NULL;
	//Line_to_neighbor = NULL;
	Stab = NULL;
	stab_gens = NULL;
	orbit = NULL;
	line_to_orbit = NULL;
	orbit_to_line = NULL;
	Pts_klein = NULL;
	Pts_wedge = NULL;
	Pts_wedge_to_line = NULL;
	line_to_pts_wedge = NULL;
	A_on_neighbors = NULL;
	Control = NULL;
	Poset = NULL;
	Five_plus_one = NULL;
	u = NULL;
	v = NULL;
	w = NULL;
	u1 = NULL;
	v1 = NULL;
	len = 0;
	Idx = NULL;
	nb = 0;
	Po = NULL;

	Pts_for_partial_ovoid_test = NULL;

	Flag_orbits = NULL;

	Double_sixes = NULL;

}

void classify_double_sixes::freeself()
{
	if (Elt0) {
		FREE_int(Elt0);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Elt4) {
		FREE_int(Elt4);
	}
	if (Idx) {
		FREE_int(Idx);
	}
	if (Po) {
		FREE_int(Po);
	}
	if (Pts_for_partial_ovoid_test) {
		FREE_int(Pts_for_partial_ovoid_test);
	}
	if (Flag_orbits) {
		FREE_OBJECT(Flag_orbits);
	}
	if (Double_sixes) {
		FREE_OBJECT(Double_sixes);
	}
	null();
}

void classify_double_sixes::init(
	surface_with_action *Surf_A, linear_group *LG,
	poset_classification_control *Control,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_double_sixes::init" << endl;
		}
	classify_double_sixes::Surf_A = Surf_A;
	classify_double_sixes::LG = LG;
	F = Surf_A->F;
	q = F->q;
	A = Surf_A->A;
	Surf = Surf_A->Surf;
	
	
	u = NEW_int(6);
	v = NEW_int(6);
	w = NEW_int(6);
	u1 = NEW_int(6);
	v1 = NEW_int(6);
	


	A = LG->A_linear;
	A2 = LG->A2;

	if (A2->type_G != action_on_wedge_product_t) {
		cout << "classify_double_sixes::init group must "
				"act in wedge action" << endl;
		exit(1);
	}

	AW = A2->G.AW;

	Elt0 = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);

	pt0_line = 0; // pt0 = the line spanned by 1000, 0100 
		// (we call it point because it is a point on the Klein quadric)
	pt0_wedge = 0; // in wedge coordinates 100000
	pt0_klein = 0; // in klein coordinates 100000


	if (f_v) {
		cout << "classify_double_sixes::init before "
				"SG_line_stab->generators_for_parabolic_subgroup" << endl;
	}


	SG_line_stab = NEW_OBJECT(strong_generators);
	SG_line_stab->generators_for_parabolic_subgroup(A, 
		A->G.matrix_grp, 2, verbose_level - 1);

	if (f_v) {
		cout << "classify_double_sixes::init after "
				"SG_line_stab->generators_for_parabolic_subgroup" << endl;
	}



	if (f_v) {
		cout << "classify_double_sixes::init "
				"before compute_neighbors" << endl;
	}
	compute_neighbors(verbose_level - 1);
	{
		spreadsheet *Sp;
		make_spreadsheet_of_neighbors(Sp, 0 /* verbose_level */);
		FREE_OBJECT(Sp);
	}
	if (f_v) {
		cout << "classify_double_sixes::init "
				"after compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
		cout << "Neighbors=";
		lint_vec_print(cout, Neighbors, nb_neighbors);
		cout << endl;
	}





	if (f_v) {
		cout << "classify_double_sixes::init "
				"computing restricted action on neighbors" << endl;
	}

	A_on_neighbors = NEW_OBJECT(action);
	A_on_neighbors = A2->create_induced_action_by_restriction(
		NULL,
		nb_neighbors, Neighbors, 
		FALSE /* f_induce_action */,
		0 /* verbose_level */);

	if (f_v) {
		cout << "classify_double_sixes::init "
				"restricted action on neighbors "
				"has been computed" << endl;
	}


	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A_on_neighbors,
			SG_line_stab,
			verbose_level);

	if (f_v) {
		cout << "classify_double_sixes::init before "
				"Poset->add_testing_without_group" << endl;
	}
	Pts_for_partial_ovoid_test = NEW_int(5 * 6);
	Poset->add_testing_without_group(
			callback_partial_ovoid_test_early,
				this /* void *data */,
				verbose_level);

	Control->f_depth = TRUE;
	Control->depth = 5;


	if (f_v) {
		cout << "classify_double_sixes::init "
				"before Five_plus_one->init" << endl;
	}
	Five_plus_one = NEW_OBJECT(poset_classification);

	Five_plus_one->initialize_and_allocate_root_node(Control, Poset,
		5 /* sz */, 
		verbose_level - 1);
	if (f_v) {
		cout << "classify_double_sixes::init "
				"after Five_plus_one->init" << endl;
	}


	//Five_plus_one->init_check_func(callback_partial_ovoid_test,
	//	(void *)this /* candidate_check_data */);


	//Five_plus_one->f_print_function = TRUE;
	//Five_plus_one->print_function = callback_print_set;
	//Five_plus_one->print_function_data = (void *) this;


	if (f_v) {
		cout << "classify_double_sixes::init done" << endl;
	}
}


void classify_double_sixes::compute_neighbors(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, a, b, c;
	sorting Sorting;

	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors" << endl;
	}

	nb_neighbors = (long int) (q + 1) * q * (q + 1);
	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
	}
	Neighbors = NEW_lint(nb_neighbors);
	Neighbor_to_line = NEW_lint(nb_neighbors);
	Neighbor_to_klein = NEW_lint(nb_neighbors);
	
	int sz;

	// At first, we get the neighbors
	// as points on the Klein quadric:
	// Later, we will change them to wedge ranks:

	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"before Surf->O->perp" << endl;
		}
	Surf->O->perp(0, Neighbors, sz, verbose_level - 3);
	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"after Surf->O->perp" << endl;

		//cout << "Neighbors:" << endl;
		//lint_matrix_print(Neighbors, (sz + 9) / 10, 10);
	}
	
	if (sz != nb_neighbors) {
		cout << "classify_double_sixes::compute_neighbors "
				"sz != nb_neighbors" << endl;
		cout << "sz = " << sz << endl;
		cout << "nb_neighbors = " << nb_neighbors << endl;
		exit(1);
	}
	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
	}
	
	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"allocating Line_to_neighbor, "
				"Surf->nb_lines_PG_3=" << Surf->nb_lines_PG_3 << endl;
	}

#if 0
	Line_to_neighbor = NEW_lint(Surf->nb_lines_PG_3);
	for (i = 0; i < Surf->nb_lines_PG_3; i++) {
		Line_to_neighbor[i] = -1;
	}
#endif


	// Convert Neighbors from points
	// on the Klein quadric to wedge points:
	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"before Surf->klein_to_wedge_vec" << endl;
	}
	Surf->klein_to_wedge_vec(Neighbors, Neighbors, nb_neighbors);

	// Sort the set Neighbors:
	Sorting.lint_vec_heapsort(Neighbors, nb_neighbors);




	// Establish the bijection between Neighbors and Lines in PG(3,q) 
	// by going through the Klein correspondence.
	// It is important that this be done after we sort Neighbors.
	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"Establish the bijection between Neighbors and Lines in "
				"PG(3,q), nb_neighbors=" << nb_neighbors << endl;
	}
	int N100;

	N100 = nb_neighbors / 100 + 1;

	for (i = 0; i < nb_neighbors; i++) {
		if ((i % N100) == 0) {
			cout << "classify_double_sixes::compute_neighbors i=" << i << " / "
					<< nb_neighbors << " at "
					<< (double)i * 100. / nb_neighbors << "%" << endl;
		}
		a = Neighbors[i];
		AW->unrank_point(w, a);
		Surf->wedge_to_klein(w, v);
		if (FALSE) {
			cout << i << " : ";
			int_vec_print(cout, v, 6);
			cout << endl;
		}
		b = Surf->O->rank_point(v, 1, 0 /* verbose_level*/);
		if (FALSE) {
			cout << " : " << b;
			cout << endl;
		}
		c = Surf->Klein->point_on_quadric_to_line(b, 0 /* verbose_level*/);
		if (FALSE) {
			cout << " : " << c << endl;
			cout << endl;
		}
		Neighbor_to_line[i] = c;
		//Line_to_neighbor[c] = i;
		}

	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors "
				"before int_vec_apply" << endl;
	}
	for (i = 0; i < nb_neighbors; i++) {
		Neighbor_to_klein[i] = Surf->Klein->line_to_point_on_quadric(
				Neighbor_to_line[i], 0 /* verbose_level*/);
	}
#if 0
	lint_vec_apply(Neighbor_to_line,
			Surf->Klein->Line_to_point_on_quadric,
			Neighbor_to_klein, nb_neighbors);
#endif


	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors done" << endl;
	}
}

void classify_double_sixes::make_spreadsheet_of_neighbors(
	spreadsheet *&Sp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char str[1000];
	string fname_csv;

	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_of_neighbors" << endl;
	}

	sprintf(str, "neighbors_%d.csv", q);
	fname_csv.assign(str);
	

	Surf->make_spreadsheet_of_lines_in_three_kinds(Sp, 
		Neighbors, Neighbor_to_line,
		Neighbor_to_klein, nb_neighbors, 0 /* verbose_level */);

	if (f_v) {
		cout << "before Sp->save " << fname_csv << endl;
	}
	Sp->save(fname_csv, verbose_level);
	if (f_v) {
		cout << "after Sp->save " << fname_csv << endl;
	}





	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_of_neighbors done" << endl;
	}
}

void classify_double_sixes::classify_partial_ovoids(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = 10000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	os_interface Os;
	int t0 = Os.os_ticks();


	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids" << endl;
	}
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids "
				"nb_neighbors = " << nb_neighbors << endl;
		cout << "Neighbors=";
		lint_vec_print(cout, Neighbors, nb_neighbors);
		cout << endl;
	}
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids "
				"classifying starter" << endl;
	}
	Five_plus_one->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids "
				"classifying starter done" << endl;
	}
	
	if (q < 20) {
		{
			spreadsheet *Sp;
			Five_plus_one->make_spreadsheet_of_orbit_reps(Sp, 5);
			char str[1000];
			string fname_csv;
			sprintf(str, "fiveplusone_%d.csv", q);
			fname_csv.assign(str);
			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
	}
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids done" << endl;
	}
}

void classify_double_sixes::report(std::ostream &ost,
		layered_graph_draw_options *draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_double_sixes::report" << endl;
	}


	if (f_v) {
		cout << "classify_double_sixes::report reporting groups" << endl;
	}
	//ost << "\\section*{The groups}" << endl;
	ost << "\\section*{The semilinear group}" << endl;
	A->report(ost, A->f_has_sims, A->Sims, A->f_has_strong_generators, A->Strong_gens, draw_options, verbose_level);
	A->latex_all_points(ost);

	if (f_v) {
		cout << "classify_double_sixes::report reporting orthogonal group" << endl;
	}
	ost << "\\section*{The orthogonal group}" << endl;
	A2->report(ost, A2->f_has_sims, A2->Sims, A2->f_has_strong_generators, A2->Strong_gens, draw_options, verbose_level);
	if (A2->degree < 100) {
		A2->latex_all_points(ost);
	}

	if (f_v) {
		cout << "classify_double_sixes::report reporting line stabilizer" << endl;
	}
	ost << "\\section*{The group stabilizing the fixed line}" << endl;
	A_on_neighbors->report(ost, A_on_neighbors->f_has_sims, A_on_neighbors->Sims,
			A_on_neighbors->f_has_strong_generators, A_on_neighbors->Strong_gens, draw_options, verbose_level);
	A_on_neighbors->latex_all_points(ost);

	ost << "{\\small\\arraycolsep=2pt" << endl;
	SG_line_stab->print_generators_tex(ost);
	ost << "}" << endl;

	if (f_v) {
		cout << "classify_double_sixes::report before Five_plus_one->report" << endl;
	}
	ost << "\\section*{The classification of five-plus-ones}" << endl;
	Five_plus_one->report(ost, verbose_level);


	if (f_v) {
		cout << "classify_double_sixes::report done" << endl;
	}
}

void classify_double_sixes::partial_ovoid_test_early(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	int u[6];
	int v[6];
	int fxy;
	int f_OK;

	if (f_v) {
		cout << "classify_double_sixes::partial_ovoid_test_early checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}

	if (len > 5) {
		cout << "classify_double_sixes::partial_ovoid_test_early len > 5" << endl;
		exit(1);
	}
	for (i = 0; i < len; i++) {
		AW->unrank_point(u, Neighbors[S[i]]);
		Surf->wedge_to_klein(u, Pts_for_partial_ovoid_test + i * 6);
	}

	if (len == 0) {
		lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "classify_double_sixes::partial_ovoid_test_early "
					"before testing" << endl;
		}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "classify_double_sixes::partial_ovoid_test_early "
						"testing " << j << " / "
						<< nb_candidates << endl;
			}

			AW->unrank_point(u, Neighbors[candidates[j]]);
			Surf->wedge_to_klein(u, v);

			f_OK = TRUE;
			for (i = 0; i < len; i++) {
				fxy = Surf->O->evaluate_bilinear_form(
						Pts_for_partial_ovoid_test + i * 6, v, 1);

				if (fxy == 0) {
					f_OK = FALSE;
					break;
				}
			}
			if (f_OK) {
				good_candidates[nb_good_candidates++] =
						candidates[j];
			}
		} // next j
	} // else
}



void classify_double_sixes::test_orbits(int verbose_level)
{
	//verbose_level += 2;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 2);
	int i, r;
	long int S[5];
	long int S2[6];
	
	if (f_v) {
		cout << "classify_double_sixes::test_orbits" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	len = Five_plus_one->nb_orbits_at_level(5);

	if (f_v) {
		cout << "classify_double_sixes::test_orbits testing "
				<< len << " orbits of 5-sets of lines:" << endl;
	}
	nb = 0;
	Idx = NEW_int(len);
	for (i = 0; i < len; i++) {
		if (f_vv || ((i % 1000) == 0)) {
			cout << "classify_double_sixes::test_orbits orbit "
				<< i << " / " << len << ":" << endl;
		}
		Five_plus_one->get_set_by_level(5, i, S);
		if (f_vv) {
			cout << "set: ";
			lint_vec_print(cout, S, 5);
			cout << endl;
		}


		lint_vec_apply(S, Neighbor_to_line, S2, 5);
		S2[5] = pt0_line;
		if (f_vv) {
			cout << "5+1 lines = ";
			lint_vec_print(cout, S2, 6);
			cout << endl;
		}

#if 1
		if (f_vv) {
			Surf->Gr->print_set(S2, 6);
		}
#endif

		r = Surf->rank_of_system(6,
				S2, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "classify_double_sixes::test_orbits orbit "
					<< i << " / " << len
					<< " has rank = " << r << endl;
		}
		if (r == 19) {
			Idx[nb++] = i;
		}
	}

	if (f_v) {
		cout << "classify_double_sixes::test_orbits we found "
				<< nb << " / " << len
				<< " orbits where the rank is 19" << endl;
		cout << "Idx=";
		int_vec_print(cout, Idx, nb);
		cout << endl;
	}
	if (f_v) {
		cout << "classify_double_sixes::test_orbits done" << endl;
	}
}

void classify_double_sixes::make_spreadsheet_of_fiveplusone_configurations(
	spreadsheet *&Sp,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int nb_orbits;
	int i, k;
	int *Stab_order;
	int *Len;
	char **Transporter;
	char **Text;
	long int *rep;
	long int *lines;
	int *data;
	longinteger_object go;
	longinteger_object len;
	string fname_csv;
	char str[1000];
	

	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_"
				"of_fiveplusone_configurations" << endl;
	}
	sprintf(str, "fiveplusone19_%d.csv", q);
	fname_csv.assign(str);

	k = 5;

	//nb_orbits = Five_plus_one->nb_orbits_at_level(k);
	rep = NEW_lint(k);
	lines = NEW_lint(k);
	Stab_order = NEW_int(nb);
	Len = NEW_int(nb);
	Transporter = NEW_pchar(nb);
	Text = NEW_pchar(nb);
	data = NEW_int(A->make_element_size);

	for (i = 0; i < nb; i++) {
		Five_plus_one->get_set_by_level(k, Idx[i], rep);
		lint_vec_apply(rep, Neighbor_to_line, lines, k);
		Five_plus_one->get_stabilizer_order(k, Idx[i], go);
		Five_plus_one->orbit_length(Idx[i], k, len);
		Stab_order[i] = go.as_int();
		Len[i] = len.as_int();
	}
	for (i = 0; i < nb; i++) {
		Five_plus_one->get_set_by_level(k, Idx[i], rep);
		lint_vec_apply(rep, Neighbor_to_line, lines, k);

		lint_vec_print_to_str(str, lines, k);

		Text[i] = NEW_char(strlen(str) + 1);
		strcpy(Text[i], str);
	}

#if 0
	if (f_with_fusion) {
		for (i = 0; i < nb; i++) {
			if (Fusion[i] == -2) {
				str[0] = 0;
				strcat(str, "\"N/A\"");
			}
			else {
				A->element_code_for_make_element(transporter->ith(i), data);


				int_vec_print_to_str(str, data, A->make_element_size);

			}
			Transporter[i] = NEW_char(strlen(str) + 1);
			strcpy(Transporter[i], str);
		}
	}
#endif


	Sp = NEW_OBJECT(spreadsheet);
#if 0
	if (f_with_fusion) {
		Sp->init_empty_table(nb + 1, 7);
	}
	else {
		Sp->init_empty_table(nb + 1, 5);
	}
#endif
	Sp->init_empty_table(nb + 1, 5);
	Sp->fill_column_with_row_index(0, "Orbit");
	Sp->fill_column_with_int(1, Idx, "Idx");
	Sp->fill_column_with_text(2, (const char **) Text, "Rep");
	Sp->fill_column_with_int(3, Stab_order, "Stab_order");
	Sp->fill_column_with_int(4, Len, "Orbit_length");
#if 0
	if (f_with_fusion) {
		Sp->fill_column_with_int(5, Fusion, "Fusion");
		Sp->fill_column_with_text(6,
				(const char **) Transporter, "Transporter");
	}
#endif
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;

	FREE_lint(rep);
	FREE_lint(lines);
	FREE_int(Stab_order);
	FREE_int(Len);
	for (i = 0; i < nb; i++) {
		FREE_char(Text[i]);
	}
	FREE_pchar(Text);
	for (i = 0; i < nb; i++) {
		FREE_char(Transporter[i]);
	}
	FREE_pchar(Transporter);
	FREE_int(data);
	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_"
				"of_fiveplusone_configurations done" << endl;
	}
}


void classify_double_sixes::identify_five_plus_one(
	long int *five_lines, long int transversal_line,
	long int *five_lines_out_as_neighbors, int &orbit_index,
	int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int W1[5];
	long int W2[5];
	long int N1[5];
	sorting Sorting;

	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one" << endl;
		cout << "classify_double_sixes::identify_five_plus_one "
				"transversal_line=" << transversal_line << endl;
		cout << "classify_double_sixes::identify_five_plus_one "
				"five_lines=";
		lint_vec_print(cout, five_lines, 5);
		cout << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	
	Surf->line_to_wedge_vec(five_lines, W1, 5);
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one W1=";
		lint_vec_print(cout, W1, 5);
		cout << endl;
	}


	A->make_element_which_moves_a_line_in_PG3q(
		Surf->Gr,
		transversal_line,
		Elt0,
		0 /* verbose_level */);
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one "
				"element which moves transversal line:" << endl;
		A->element_print(Elt0, cout);
	}


	A2->map_a_set(
			W1,
			W2,
			5,
			Elt0, 0 /* verbose_level */);
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one W2=";
		lint_vec_print(cout, W2, 5);
		cout << endl;
	}

	Sorting.lint_vec_search_vec(Neighbors, nb_neighbors,
			W2, 5, N1);

	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one "
				"tracing the set N1=";
		lint_vec_print(cout, N1, 5);
		cout << endl;
	}
	orbit_index = Five_plus_one->trace_set(
			N1, 5, 5,
			five_lines_out_as_neighbors,
			Elt1,
			verbose_level - 2);
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one "
				"orbit_index = " << orbit_index << endl;
	}
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one "
				"element which moves neighbor set:" << endl;
		A->element_print(Elt1, cout);
	}

	
	A->element_mult(Elt0, Elt1, transporter, 0);
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one "
				"element which moves five_plus_one:" << endl;
		A->element_print(transporter, cout);
	}
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one "
				"done" << endl;
	}
}

void classify_double_sixes::classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_double_sixes::classify" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::classify "
				"before downstep" << endl;
	}
	downstep(verbose_level);
	if (f_v) {
		cout << "classify_double_sixes::classify "
				"after downstep" << endl;
		cout << "we found " << Flag_orbits->nb_flag_orbits
				<< " flag orbits out of "
				<< Five_plus_one->nb_orbits_at_level(5)
				<< " orbits" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::classify "
				"before upstep" << endl;
	}
	upstep(verbose_level);
	if (f_v) {
		cout << "classify_double_sixes::classify "
				"after upstep" << endl;
		cout << "we found " << Double_sixes->nb_orbits
				<< " double sixes out of "
				<< Flag_orbits->nb_flag_orbits
				<< " flag orbits" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::classify done" << endl;
	}
}


void classify_double_sixes::downstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f, i, nb_orbits, nb_flag_orbits, c;

	if (f_v) {
		cout << "classify_double_sixes::downstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"before test_orbits" << endl;
	}
	test_orbits(verbose_level - 1);
	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"after test_orbits" << endl;
		cout << "Idx=";
		int_vec_print(cout, Idx, nb);
		cout << endl;
	}



	nb_orbits = Five_plus_one->nb_orbits_at_level(5);
	
	Flag_orbits = NEW_OBJECT(flag_orbits);
	Flag_orbits->init(A, A2,
		nb_orbits /* nb_primary_orbits_lower */,
		5 + 6 + 12 /* pt_representation_sz */,
		nb,
		1 /* upper_bound_for_number_of_traces */, // ToDo
		NULL /* void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level) */,
		NULL /* void (*func_latex_report_trace)(std::ostream &ost, void *trace_result, void *data, int verbose_level)*/,
		NULL /* void *free_received_trace_data */,
		verbose_level);

	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"initializing flag orbits" << endl;
	}

	int f_process = FALSE;
	int nb_100 = 1;

	if (nb > 1000) {
		f_process = TRUE;
		nb_100 = nb / 100 + 1;
	}

	nb_flag_orbits = 0;
	for (f = 0; f < nb; f++) {

		i = Idx[f];
		if (f_v) {
			cout << "classify_double_sixes::downstep "
					"orbit " << f << " / " << nb
					<< " with rank = 19 is orbit "
					<< i << " / " << nb_orbits << endl;
		}
		if (f_process) {
			if ((f % nb_100) == 0) {
				cout << "classify_double_sixes::downstep orbit "
					<< i << " / " << nb_orbits << ", progress at " << f / nb_100 << "%" << endl;
			}
		}

		set_and_stabilizer *R;
		longinteger_object ol;
		longinteger_object go;
		long int dataset[23];

		R = Five_plus_one->get_set_and_stabilizer(
				5 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);

		Five_plus_one->orbit_length(
				i /* node */, 5 /* level */, ol);

		R->Strong_gens->group_order(go);

		lint_vec_copy(R->data, dataset, 5);

		lint_vec_apply(dataset,
				Neighbor_to_line, dataset + 5, 5);
		
		dataset[10] = pt0_line;

		long int double_six[12];
		if (f_vv) {
			cout << "5+1 lines = ";
			lint_vec_print(cout, dataset + 5, 6);
			cout << endl;
		}

		if (f_vv) {
			cout << "classify_double_sixes::downstep before "
					"create_double_six_from_five_lines_with_"
					"a_common_transversal" << endl;
		}

		c = Surf_A->create_double_six_from_five_lines_with_a_common_transversal(
				dataset + 5, pt0_line, double_six,
				0 /*verbose_level*/);
		
		if (c) {

			if (f_vv) {
				cout << "The starter configuration is good, "
						"a double six has been computed:" << endl;
				lint_matrix_print(double_six, 2, 6);
			}

			lint_vec_copy(double_six, dataset + 11, 12);


			Flag_orbits->Flag_orbit_node[nb_flag_orbits].init(
				Flag_orbits,
				nb_flag_orbits /* flag_orbit_index */,
				i /* downstep_primary_orbit */,
				0 /* downstep_secondary_orbit */,
				ol.as_int() /* downstep_orbit_len */,
				FALSE /* f_long_orbit */,
				dataset /* int *pt_representation */,
				R->Strong_gens,
				verbose_level - 2);
			R->Strong_gens = NULL;

			if (f_vv) {
				cout << "orbit " << f << " / " << nb
					<< " with rank = 19 is orbit " << i
					<< " / " << nb_orbits << ", stab order "
					<< go << endl;
			}
			nb_flag_orbits++;
		}
		else {
			if (f_vv) {
				cout << "classify_double_sixes::downstep "
						"orbit " << f << " / " << nb
						<< " with rank = 19 does not yield a "
						"double six, skipping" << endl;
			}
		}


		FREE_OBJECT(R);
	}

	Flag_orbits->nb_flag_orbits = nb_flag_orbits;


	Po = NEW_int(nb_flag_orbits);
	for (f = 0; f < nb_flag_orbits; f++) {
		Po[f] = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
	}
	if (f_v) {
		cout << "classify_double_sixes::downstep we found "
			<< nb_flag_orbits << " flag orbits out of "
			<< nb_orbits << " orbits" << endl;
	}
	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"initializing flag orbits done" << endl;
	}
}


void classify_double_sixes::upstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, k, i0;
	int f, po, so;
	int *f_processed;
	int nb_processed;
	sorting Sorting;

	if (f_v) {
		cout << "classify_double_sixes::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	f_processed = NEW_int(Flag_orbits->nb_flag_orbits);
	int_vec_zero(f_processed, Flag_orbits->nb_flag_orbits);
	nb_processed = 0;

	Double_sixes = NEW_OBJECT(classification_step);

	longinteger_object go;
	A->group_order(go);

	Double_sixes->init(A, A2,
			Flag_orbits->nb_flag_orbits, 12, go,
			verbose_level);


	if (f_v) {
		cout << "flag orbit : downstep_primary_orbit" << endl;
		cout << "f : po" << endl;
		for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {
			po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
			cout << f << " : " << po << endl;
		}
	}
	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		double progress;
		long int dataset[23];
		
		if (f_processed[f]) {
			continue;
		}

		progress = ((double)nb_processed * 100. ) /
				(double) Flag_orbits->nb_flag_orbits;

		if (f_v) {
			cout << "classify_double_sixes::upstep "
				"Defining n e w orbit "
				<< Flag_orbits->nb_primary_orbits_upper
				<< " from flag orbit " << f << " / "
				<< Flag_orbits->nb_flag_orbits
				<< " progress=" << progress << "%" << endl;
		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit
			= Flag_orbits->nb_primary_orbits_upper;
		

		if (Flag_orbits->pt_representation_sz != 23) {
			cout << "Flag_orbits->pt_representation_sz != 23" << endl;
			exit(1);
		}
		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
		}
		lint_vec_copy(Flag_orbits->Pt + f * 23, dataset, 23);




		vector_ge *coset_reps;
		int nb_coset_reps;
		
		coset_reps = NEW_OBJECT(vector_ge);
		coset_reps->init(Surf_A->A, verbose_level - 2);
		coset_reps->allocate(12, verbose_level - 2);


		strong_generators *S;
		longinteger_object go;
		long int double_six[12];

		lint_vec_copy(dataset + 11, double_six, 12);

		if (f_v) {
			cout << "double six:";
			lint_vec_print(cout, double_six, 12);
			cout << endl;
		}
		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
		S->group_order(go);
		if (f_v) {
			cout << "po=" << po << " so=" << so
					<< " go=" << go << endl;
		}

		nb_coset_reps = 0;
		for (i = 0; i < 2; i++) {
			for (j = 0; j < 6; j++) {
			
				if (f_v) {
					cout << "i=" << i << " j=" << j << endl;
				}
				long int transversal_line;
				long int five_lines[5];
				//int five_lines_in_wedge[5];
				long int five_lines_out_as_neighbors[5];
				int orbit_index;
				int f2;
				
				transversal_line = double_six[i * 6 + j];
				i0 = 1 - i;
				k = 0;
				for (h = 0; h < 6; h++) {
					if (h == j) {
						continue;
					}
					five_lines[k++] = double_six[i0 * 6 + h];
				}

				//int_vec_apply(five_lines,
				//Line_to_neighbor, five_lines_in_wedge, 5);
				
				if (f_v) {
					cout << "transversal_line = "
							<< transversal_line << " five_lines=";
					lint_vec_print(cout, five_lines, 5);
					cout << endl;
				}
				identify_five_plus_one(five_lines, transversal_line, 
					five_lines_out_as_neighbors, orbit_index, 
					Elt3 /* transporter */, verbose_level - 2);

				if (f_v) {
					cout << "We found a transporter:" << endl;
					A->element_print_quick(Elt3, cout);
				}

				if (!Sorting.int_vec_search(Po, Flag_orbits->nb_flag_orbits,
						orbit_index, f2)) {
					cout << "cannot find orbit " << orbit_index
							<< " in Po" << endl;
					cout << "Po=";
					int_vec_print(cout, Po, Flag_orbits->nb_flag_orbits);
					cout << endl;
					exit(1);
				}

				if (Flag_orbits->Flag_orbit_node[f2].downstep_primary_orbit
						!= orbit_index) {
					cout << "Flag_orbits->Flag_orbit_node[f2].downstep_"
							"primary_orbit != orbit_index" << endl;
					exit(1);
				}





		
				if (f2 == f) {
					if (f_v) {
						cout << "We found an automorphism of "
								"the double six:" << endl;
						A->element_print_quick(Elt3, cout);
						cout << endl;
					}
					A->element_move(Elt3, coset_reps->ith(nb_coset_reps), 0);
					nb_coset_reps++;
					//S->add_single_generator(Elt3,
					//2 /* group_index */, verbose_level - 2);
				}
				else {
					if (f_v) {
						cout << "We are identifying flag orbit "
								<< f2 << " with flag orbit " << f << endl;
					}
					if (!f_processed[f2]) {
						Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
							= Flag_orbits->nb_primary_orbits_upper;
						Flag_orbits->Flag_orbit_node[f2].f_fusion_node
							= TRUE;
						Flag_orbits->Flag_orbit_node[f2].fusion_with
							= f;
						Flag_orbits->Flag_orbit_node[f2].fusion_elt
							= NEW_int(A->elt_size_in_int);
						A->element_invert(Elt3,
								Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								0);
						f_processed[f2] = TRUE;
						nb_processed++;
					}
					else {
						cout << "Flag orbit " << f2 << " has already been "
								"identified with flag orbit " << f << endl;
						if (Flag_orbits->Flag_orbit_node[f2].fusion_with != f) {
							cout << "Flag_orbits->Flag_orbit_node[f2]."
									"fusion_with != f" << endl;
							exit(1);
						}
					}
				}
			} // next j
		} // next i


		coset_reps->reallocate(nb_coset_reps, verbose_level - 2);

		strong_generators *Aut_gens;

		{
			longinteger_object ago;

			if (f_v) {
				cout << "classify_double_sixes::upstep "
						"Extending the group by a factor of "
						<< nb_coset_reps << endl;
			}
			Aut_gens = NEW_OBJECT(strong_generators);
			Aut_gens->init_group_extension(S,
					coset_reps, nb_coset_reps,
					verbose_level - 2);
			if (f_v) {
				cout << "classify_double_sixes::upstep "
						"Aut_gens tl = ";
				int_vec_print(cout,
						Aut_gens->tl, Aut_gens->A->base_len());
				cout << endl;
			}

			Aut_gens->group_order(ago);


			if (f_v) {
				cout << "the double six has a stabilizer of order "
						<< ago << endl;
				cout << "The double six stabilizer is:" << endl;
				Aut_gens->print_generators_tex(cout);
			}
		}



		Double_sixes->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
			Double_sixes,
			Flag_orbits->nb_primary_orbits_upper, 
			Aut_gens, dataset + 11, NULL /* extra_data */, verbose_level);

		FREE_OBJECT(coset_reps);
		FREE_OBJECT(S);
		
		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;
	} // next f


	if (nb_processed != Flag_orbits->nb_flag_orbits) {
		cout << "nb_processed != Flag_orbits->nb_flag_orbits" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "Flag_orbits->nb_flag_orbits = "
				<< Flag_orbits->nb_flag_orbits << endl;
		exit(1);
	}

	Double_sixes->nb_orbits = Flag_orbits->nb_primary_orbits_upper;
	
	if (f_v) {
		cout << "We found " << Flag_orbits->nb_primary_orbits_upper
				<< " orbits of double sixes" << endl;
	}
	
	FREE_int(f_processed);


	if (f_v) {
		cout << "classify_double_sixes::upstep done" << endl;
	}
}


void classify_double_sixes::print_five_plus_ones(ostream &ost)
{
	int f, i, l;

	l = Five_plus_one->nb_orbits_at_level(5);

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Classification of $5+1$ Configurations "
			"in $\\PG(3," << q << ")$}" << endl;



	{
		longinteger_object go;
		A->Strong_gens->group_order(go);

		ost << "The order of the group is ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;

		ost << "\\bigskip" << endl;
	}



	longinteger_domain D;
	longinteger_object ol, Ol;
	Ol.create(0, __FILE__, __LINE__);

	ost << "The group has " 
		<< l 
		<< " orbits on five plus one configurations in $\\PG(3,"
		<< q << ").$" << endl << endl;

	ost << "Of these, " << nb << " impose 19 conditions."
			<< endl << endl;


	ost << "Of these, " << Flag_orbits->nb_flag_orbits
			<< " are associated with double sixes. "
				"They are:" << endl << endl;


	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		i = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;


		set_and_stabilizer *R;

		R = Five_plus_one->get_set_and_stabilizer(
				5 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);
		Five_plus_one->orbit_length(
				i /* node */,
				5 /* level */, ol);
		D.add_in_place(Ol, ol);
		
		ost << "$" << f << " / " << Flag_orbits->nb_flag_orbits
				<< "$ is orbit $" << i << " / " << l << "$ $" << endl;
		R->print_set_tex(ost);
		ost << "$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		FREE_OBJECT(R);
	}

	ost << "The overall number of five plus one configurations "
			"associated with double sixes in $\\PG(3," << q
			<< ")$ is: " << Ol << "\\\\" << endl;


	//Double_sixes->print_latex(ost, "Classification of Double Sixes");
}

void classify_double_sixes::identify_double_six(long int *double_six,
	int *transporter, int &orbit_index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 4);
	int f, f2;
	int *Elt1;
	int *Elt2;
	long int transversal_line;
	long int five_lines[5];
	long int five_lines_out_as_neighbors[5];
	int po;
	sorting Sorting;

	if (f_v) {
		cout << "classify_double_sixes::identify_double_six" << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six "
				"identifying the five lines a_1,...,a_5 "
				"with transversal b_6" << endl;
	}
	transversal_line = double_six[11];
	lint_vec_copy(double_six, five_lines, 5);
	
	identify_five_plus_one(five_lines, transversal_line, 
		five_lines_out_as_neighbors, po, 
		Elt1 /* transporter */, 0 /* verbose_level */);

	if (f_vv) {
		cout << "po=" << po << endl;
		cout << "Elt1=" << endl;
		A->element_print_quick(Elt1, cout);
	}

	
	if (!Sorting.int_vec_search(Po, Flag_orbits->nb_flag_orbits, po, f)) {
		cout << "classify_double_sixes::identify_double_six "
				"did not find po in Po" << endl;
		exit(1);
	}
	
	if (f_vv) {
		cout << "po=" << po << " f=" << f << endl;
	}

	if (Flag_orbits->Flag_orbit_node[f].f_fusion_node) {
		A->element_mult(Elt1,
				Flag_orbits->Flag_orbit_node[f].fusion_elt, Elt2, 0);
		f2 = Flag_orbits->Flag_orbit_node[f].fusion_with;
		orbit_index =
				Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit;
	}
	else {
		f2 = -1;
		A->element_move(Elt1, Elt2, 0);
		orbit_index = Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit;
	}
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six "
				"f=" << f << " f2=" << f2 << " orbit_index="
				<< orbit_index << endl;
	}
	A->element_move(Elt2, transporter, 0);
	if (f_vv) {
		cout << "transporter=" << endl;
		A->element_print_quick(transporter, cout);
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six done" << endl;
	}
}

void classify_double_sixes::write_file(ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "classify_double_sixes::write_file" << endl;
	}
	fp.write((char *) &q, sizeof(int));
	fp.write((char *) &nb_neighbors, sizeof(int));
	fp.write((char *) &len, sizeof(int));
	fp.write((char *) &nb, sizeof(int));
	fp.write((char *) &Flag_orbits->nb_flag_orbits, sizeof(int));

	for (i = 0; i < nb; i++) {
		fp.write((char *) &Idx[i], sizeof(int));
	}
	for (i = 0; i < Flag_orbits->nb_flag_orbits; i++) {
		fp.write((char *) &Po[i], sizeof(int));
	}


	if (f_v) {
		cout << "classify_double_sixes::write_file before Five_plus_one->write_file" << endl;
	}
	Five_plus_one->write_file(fp,
			5 /* depth_completed */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::write_file after Five_plus_one->write_file" << endl;
	}


	if (f_v) {
		cout << "classify_double_sixes::write_file before Flag_orbits->write_file" << endl;
	}
	Flag_orbits->write_file(fp, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::write_file after Flag_orbits->write_file" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::write_file before Double_sixes->write_file" << endl;
	}
	Double_sixes->write_file(fp, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::write_file after Double_sixes->write_file" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::write_file finished" << endl;
	}
}

void classify_double_sixes::read_file(ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_flag_orbits;
	
	if (f_v) {
		cout << "classify_double_sixes::read_file" << endl;
	}
	fp.read((char *) &q, sizeof(int));
	fp.read((char *) &nb_neighbors, sizeof(int));
	fp.read((char *) &len, sizeof(int));
	fp.read((char *) &nb, sizeof(int));
	fp.read((char *) &nb_flag_orbits, sizeof(int));

	if (f_v) {
		cout << "classify_double_sixes::read_file q=" << q << endl;
		cout << "classify_double_sixes::read_file nb_neighbors=" << nb_neighbors << endl;
		cout << "classify_double_sixes::read_file len=" << len << endl;
		cout << "classify_double_sixes::read_file nb=" << nb << endl;
		cout << "classify_double_sixes::read_file nb_flag_orbits=" << nb_flag_orbits << endl;
	}

	Idx = NEW_int(nb);
	for (i = 0; i < nb; i++) {
		fp.read((char *) &Idx[i], sizeof(int));
	}

	Po = NEW_int(nb_flag_orbits);
	for (i = 0; i < nb_flag_orbits; i++) {
		fp.read((char *) &Po[i], sizeof(int));
	}


	int depth_completed;

	if (f_v) {
		cout << "classify_double_sixes::read_file before Five_plus_one->read_file" << endl;
	}
	Five_plus_one->read_file(fp, depth_completed, verbose_level);
	if (f_v) {
		cout << "classify_double_sixes::read_file after Five_plus_one->read_file" << endl;
	}
	if (depth_completed != 5) {
		cout << "classify_double_sixes::read_file "
				"depth_completed != 5" << endl;
		exit(1);
	}


	Flag_orbits = NEW_OBJECT(flag_orbits);
	//Flag_orbits->A = A;
	//Flag_orbits->A2 = A;
	if (f_v) {
		cout << "classify_double_sixes::read_file before Flag_orbits->read_file" << endl;
	}
	Flag_orbits->read_file(fp, A, A2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::read_file after Flag_orbits->read_file" << endl;
	}

	Double_sixes = NEW_OBJECT(classification_step);
	//Double_sixes->A = A;
	//Double_sixes->A2 = A2;

	longinteger_object go;
	A->group_order(go);
	//A->group_order(Double_sixes->go);

	if (f_v) {
		cout << "classify_double_sixes::read_file before Double_sixes->read_file" << endl;
	}
	Double_sixes->read_file(fp, A, A2, go, 0/*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::read_file after Double_sixes->read_file" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::read_file finished" << endl;
	}
}


int classify_double_sixes::line_to_neighbor(long int line_rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;
	long int point_rk;
	sorting Sorting;

	if (f_v) {
		cout << "classify_double_sixes::line_to_neighbor" << endl;
	}
	point_rk = Surf->Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level*/);
	if (!Sorting.lint_vec_search(Neighbors, nb_neighbors, point_rk,
			idx, 0 /* verbose_level */)) {
		cout << "classify_double_sixes::line_to_neighbor line " << line_rk
				<< " = point " << point_rk << " not found in Neighbors[]" << endl;
		exit(1);
	}
	return idx;
}



#if 0
int callback_partial_ovoid_test(int len, int *S,
		void *data, int verbose_level)
{
	classify_double_sixes *Classify_double_sixes =
			(classify_double_sixes *) data;
	//surface_classify_wedge *SCW = (surface_classify_wedge *) data;

	return Classify_double_sixes->partial_ovoid_test(
			S, len, verbose_level);
}
#endif


void callback_partial_ovoid_test_early(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	classify_double_sixes *Classify_double_sixes = (classify_double_sixes *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "callback_partial_ovoid_test_early for set ";
		lint_vec_print(cout, S, len);
		cout << endl;
	}
	Classify_double_sixes->partial_ovoid_test_early(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "callback_partial_ovoid_test_early done" << endl;
	}
}

}}



