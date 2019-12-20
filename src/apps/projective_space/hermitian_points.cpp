// hermitian_points.cpp
//
// Anton Betten
//
// started:  March 19, 2010




#include "orbiter.h"

using namespace std;


using namespace orbiter;



int main(int argc, const char **argv)
{
	os_interface Os;

	int t0 = Os.os_ticks();
	int verbose_level = 0;
	int i;
	int f_list_N = FALSE;
	int f_list_N1 = FALSE;
	int f_list_S = FALSE;
	int f_list_Sbar = FALSE;
	int f_n = FALSE;
	int n = 0;
	int f_Q = FALSE;
	int Q = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_list_N = TRUE;
			cout << "-N" << endl;
			}
		else if (strcmp(argv[i], "-N1") == 0) {
			f_list_N1 = TRUE;
			cout << "-N1" << endl;
			}
		else if (strcmp(argv[i], "-S") == 0) {
			f_list_S = TRUE;
			cout << "-S" << endl;
			}
		else if (strcmp(argv[i], "-Sbar") == 0) {
			f_list_Sbar = TRUE;
			cout << "-Sbar" << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n" << n << endl;
			}
		else if (strcmp(argv[i], "-Q") == 0) {
			f_Q = TRUE;
			Q = atoi(argv[++i]);
			cout << "-Q" << Q << endl;
			}

		}
	if (!f_n) {
		cout << "Please specify the projective dimension using -n <n>" << endl;
		exit(1);
		}
	if (!f_Q) {
		cout << "Please specify the order of the field using -Q <Q>" << endl;
		exit(1);
		}

	int len;

	len = n + 1;
	cout << "len=" << len << endl;
	cout << "Q=" << Q << endl;

	finite_field F;
	hermitian H;

	F.init(Q, verbose_level);
	H.init(&F, len, verbose_level);
	if (f_list_N) {
		H.list_all_N(verbose_level);
		}
	if (f_list_N1) {
		H.list_all_N1(verbose_level);
		}
	if (f_list_S) {
		H.list_all_S(verbose_level);
		}
	if (f_list_Sbar) {
		H.list_all_Sbar(verbose_level);
		}


#if 0
	int *Pts;
	int nb_pts;
	int *v;
	int *line_type;
	projective_space *P;
	int f_semilinear = TRUE;
	strong_generators *sg;
	action *A2;
	classify C;
	int f, l, j, a, b, idx;
	int **Intersection_sets;
	int sz, intersection_set_size;
	


	v = NEW_int(len);
	H.list_of_points_embedded_in_PG(Pts, nb_pts, verbose_level);
	cout << "We found " << nb_pts << " points, they are:" << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << i << " : " << Pts[i] << " : ";
		PG_element_unrank_modified(F, v, 1, len, Pts[i]);
		int_vec_print(cout, v, len);
		cout << endl;
		}


	P = new projective_space;

	cout << "Creating projective_space" << endl;
	P->init(n, &F, 
		TRUE /* f_init_group */, 
		TRUE /* f_with_line_action,  */, 
		TRUE /* f_init_incidence_structure */,
		f_semilinear, 
		TRUE /* f_basis */, 
		0 /* verbose_level */);
	cout << "Creating projective_space done" << endl;


	line_type = NEW_int(P->N_lines);

	P->line_intersection_type(Pts, nb_pts, line_type, verbose_level);

	sims *GU;

	C.init(line_type, P->N_lines, FALSE, 0);
	cout << "The line type is:" << endl;
	C.print(TRUE /* f_backwards*/);

	cout << "The secants are:" << endl;
	f = C.type_first[1];
	l = C.type_len[1];
	Intersection_sets = NEW_pint(l);
	sz = C.data_sorted[f];

	int *secants;
	int nb_secants;

	secants = NEW_int(l);
	nb_secants = l;

	for (j = 0; j < l; j++) {
		a = C.sorting_perm_inv[f + j];
		secants[j] = a;
		}

	for (j = 0; j < l; j++) {
		a = C.sorting_perm_inv[f + j];
		cout << j << " : " << a << " : ";

		P->intersection_of_subspace_with_point_set(
			P->Grass_lines, a, Pts, nb_pts, 
			Intersection_sets[j], intersection_set_size, 0 /* verbose_level */);
		if (intersection_set_size != sz) {
			cout << "intersection_set_size != sz" << endl;
			exit(1);
			}
		for (i = 0; i < sz; i++) {
			b = Intersection_sets[j][i];
			if (!int_vec_search_linear(Pts, nb_pts, b, idx)) {
				cout << "cannot find the point" << endl;
				exit(1);
				}
			Intersection_sets[j][i] = idx;
			}

		int_vec_print(cout, Intersection_sets[j], sz);
		cout << endl;
		}


	

	cout << "Computing the unitary group:" << endl;
	GU = projective_space_set_stabilizer(P, 
		Pts, nb_pts, verbose_level);
	longinteger_object go;

	GU->group_order(go);
	cout << "Group has been computed, group order = " << go << endl;



	sg = new strong_generators;

	sg->init_from_sims(GU, 0 /* verbose_level */);

	cout << "strong generators are:" << endl;
	sg->print_generators();

	
	A2 = P->A2;


	action *A2r;

	A2r = new action;

	cout << "Creating restricted action on secants:" << endl;
	A2r->induced_action_by_restriction(*A2, 
		FALSE /* f_induce_action */, NULL, 
		nb_secants, secants, 0 /* verbose_level */);
	cout << "Creating restricted action on secants done." << endl;
	


	delete A2r;
	delete sg;
	delete GU;
	for (j = 0; j < l; j++) {
		FREE_int(Intersection_sets[j]);
		}
	FREE_pint(Intersection_sets);
	FREE_int(line_type);
	delete P;
	FREE_int(Pts);
	FREE_int(v);
#endif

	Os.time_check(cout, t0);
	cout << endl;
}


