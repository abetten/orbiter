// orthogonal_group.C
// 
// Anton Betten
// 10/17/2007
//
// 
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

void usage(int argc, char **argv);
int main(int argc, char **argv);
void do_it(int epsilon, int n, int q, int verbose_level);

void usage(int argc, char **argv)
{
	cout << "usage: " << argv[0] << " [options]" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <n>                   : verbose level n" << endl;
	cout << "-epsilon <epsilon>       : set form type epsilon" << endl;
	cout << "-d <d>                   : set dimension d" << endl;
	cout << "-q <q>                   : set field size q" << endl;
}



int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_epsilon = FALSE;
	int epsilon = 0;
	int f_d = FALSE;
	int d = 0;
	int f_q = FALSE;
	int q = 0;

	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-h") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-help") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	if (!f_epsilon) {
		cout << "please use -epsilon <epsilon>" << endl;
		usage(argc, argv);
		exit(1);
		}	
	if (!f_d) {
		cout << "please use -d <d>" << endl;
		usage(argc, argv);
		exit(1);
		}	
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		usage(argc, argv);
		exit(1);
		}	
	do_it(epsilon, d, q, verbose_level);

	the_end_quietly(t0);
}

void do_it(int epsilon, int n, int q, int verbose_level)
{
	finite_field *F;
	action *A;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int p, h, i, j, a;
	int *v;
	
	A = NEW_OBJECT(action);
	is_prime_power(q, p, h);
	if (h > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;
	
	v = NEW_int(n);

	
	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	A->init_orthogonal_group(epsilon, n, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, verbose_level);
	

	if (!A->f_has_strong_generators) {
		cout << "action does not have strong generators" << endl;
		exit(1);
		}
	strong_generators *SG;
	longinteger_object go;
	action_on_orthogonal *AO = A->G.AO;
	orthogonal *O = AO->O;

	SG = A->Strong_gens;
	SG->group_order(go);

	cout << "The group " << A->label << " has order "
			<< go << " and permutation degree "
			<< A->degree << endl;
	cout << "The points on which the group acts are:" << endl;
	for (i = 0; i < A->degree; i++) {
		O->unrank_point(v, 1 /* stride */, i, 0 /* verbose_level */);
		cout << i << " / " << A->degree << " : ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		cout << "as permutation: " << endl;
		A->element_print_as_permutation(
				SG->gens->ith(i), cout);
		cout << endl;
		}
	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		A->element_print_as_permutation(SG->gens->ith(i), cout);
		cout << endl;
		}
	cout << "Generators in compact permutation form are:" << endl;
	cout << SG->gens->len << " " << A->degree << endl;
	for (i = 0; i < SG->gens->len; i++) {
		for (j = 0; j < A->degree; j++) {
			a = A->element_image_of(j,
					SG->gens->ith(i), 0 /* verbose_level */);
			cout << a << " ";
			}
		cout << endl;
		}
	cout << "-1" << endl;

	schreier *Sch;
	char fname_tree[1000];
	char fname_report[1000];
	int xmax = 2000000;
	int ymax = 1000000;
	int f_circletext = TRUE;
	int rad = 18000;
	int f_embedded = FALSE;
	int f_sideways = TRUE;
	double scale = 0.35;
	double line_width = 1.0;

	sprintf(fname_tree, "O_%d_%d_%d_tree", epsilon, n, q);
	sprintf(fname_report, "O_%d_%d_%d_report.tex", epsilon, n, q);

	Sch = NEW_OBJECT(schreier);

	cout << "computing orbits on points:" << endl;
	A->all_point_orbits(*Sch, verbose_level);
	Sch->draw_tree(fname_tree, 0 /* orbit_no*/,
			xmax, ymax, f_circletext, rad,
			f_embedded, f_sideways,
			scale, line_width,
			FALSE /* f_has_point_labels */, NULL /*  *point_labels */,
			verbose_level);


	{
	ofstream fp(fname_report);


	latex_head_easy(fp);

	SG->print_generators_tex(fp);

	//fp << "Schreier tree:" << endl;
	fp << "\\input " << fname_tree << ".tex" << endl;


	if (q == 3 && n == 5) {
		int u[] = { // singular vectors
				0,0,1,1,0,
				1,2,0,2,1,
				0,0,0,1,0,
				1,0,0,2,1,
				1,1,2,0,2
		};
		int v[] = { // v is orthogonal to u
				0,1,2,0,2,
				2,0,1,2,2,
				1,0,0,0,0,
				2,0,2,0,1,
				0,2,2,0,2
		};
		int w[] = {
				1,1,1,1,0
		};
		int *Mtx;

		Mtx = NEW_int(6 * 25);
		for (i = 0; i < 5; i++) {
			cout << "creating Siegel transformation " << i << " / 5:" << endl;
			::Siegel_Transformation(*F, 0 /*epsilon */, n - 1,
					1 /*form_c1*/, 0 /*form_c2*/, 0 /*form_c3*/,
					Mtx + i * 25, v + i * 5, u + i * 5, verbose_level);
			int_matrix_print(Mtx + i * 25, 5, 5);
			cout << endl;
		}
		O->make_orthogonal_reflection(Mtx + 5 * 25, w, verbose_level - 1);
		int_matrix_print(Mtx + 5 * 25, 5, 5);
		cout << endl;
		cout << "generators for O(5,3) are:" << endl;
		int_matrix_print(Mtx, 6, 25);

		vector_ge *gens;
		gens = NEW_OBJECT(vector_ge);
		gens->init_from_data(A, Mtx,
				6 /* nb_elements */, 25 /* elt_size */, verbose_level);
		gens->print(cout);
		schreier *Sch2;
		Sch2 = NEW_OBJECT(schreier);

		cout << "computing orbits on points:" << endl;
		Sch2->init(A);
		Sch2->init_generators(*gens);
		Sch2->compute_all_point_orbits(verbose_level);

		char fname_tree2[1000];

		sprintf(fname_tree2, "O_%d_%d_%d_tree2", epsilon, n, q);
		Sch2->draw_tree(fname_tree2, 0 /* orbit_no*/,
				xmax, ymax, f_circletext, rad,
				f_embedded, f_sideways,
				scale, line_width,
				FALSE /* f_has_point_labels */, NULL /*  *point_labels */,
				verbose_level);

		longinteger_object go;
		A->group_order(go);

		gens->print_generators_tex(go, fp);


		//fp << "Schreier tree:" << endl;
		fp << "\\input " << fname_tree2 << ".tex" << endl;



	}


	latex_foot(fp);

	}
	FREE_int(v);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
}


