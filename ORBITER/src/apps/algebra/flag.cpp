// flag.C
// 
// Anton Betten
// May 19, 2016
//
//
// 
//
//

#include "orbiter.h"

using namespace orbiter;

void test_indexing(int n, int *type, int type_len,
		finite_field *F, int verbose_level);
void test_action(int n, int *type, int type_len,
		finite_field *F, int verbose_level);
void print_flag(ostream &ost, int pt, void *data);


int main(int argc, const char **argv)
{
	int i;
	int verbose_level;
	int f_n = FALSE;
	int n = 0;
	int f_type = FALSE;
	int type_len = 0;
	int type[100];
	int f_q = FALSE;
	int q = 0;
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-type") == 0) {
			f_type = TRUE;
			while (i < argc) {
				type[type_len] = atoi(argv[++i]);
				if (type[type_len] == -1) {
					break;
					}
				if (type[type_len] < 0) {
					cout << "type must be a sequence of "
							"positive numbers" << endl;
					exit(1);
					}
				type_len++;
				}
			cout << "-type of length " << type_len << " : ";
			int_vec_print(cout, type, type_len);
			cout << endl;
			}
		}
	if (!f_n) {
		cout << "please use -n <n> to specify n" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q <q> to specify q" << endl;
		exit(1);
		}
	if (!f_type) {
		cout << "please use -type" << endl;
		exit(1);
		}
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	test_indexing(n, type, type_len, F, verbose_level);
	test_action(n, type, type_len, F, verbose_level);


	FREE_OBJECT(F);

}


void test_indexing(int n, int *type, int type_len,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, h2, N;
	int *subspace;

	if (f_v) {
		cout << "test_indexing" << endl;
		}
	
	subspace = NEW_int(n * n);
	flag *Flag;

	Flag = NEW_OBJECT(flag);
	Flag->init(n, type, type_len, F, verbose_level);
	
	//Flag->unrank(4, subspace, 1 /*verbose_level*/);
	//int_matrix_print(subspace, n, n);

#if 1
	cout << "The flags are:" << endl;
	N = Flag->N;
	for (h = 0; h < N; h++) {
		cout << h << " / " << N << ":" << endl;
		Flag->unrank(h, subspace, 0 /*verbose_level*/);
		//cout << h << " / " << N << ":" << endl;
		int_matrix_print(subspace, n, n);
		h2 = Flag->rank(subspace, 0 /*verbose_level*/);
		if (h2 == h) {
			cout << "check!" << endl;
			}
		else {
			cout << "problem, the subspace is ranked as " << h2 << endl;
			exit(1);
			}

		}
#endif
}

	action *A;
	matrix_group *Mtx;
	action *AF;
	sims *S;
	int flag_n;
	int *subspace;

void test_action(int n, int *type, int type_len,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_projective = TRUE;
	int f_general = FALSE;
	int f_affine = FALSE;
	int f_semilinear = FALSE;
	int f_special = FALSE;

	if (f_v) {
		cout << "test_action" << endl;
		}
	flag_n = n;

	subspace = NEW_int(n * n);

	create_linear_group(S, A, 
		F, n, 
		f_projective, f_general, f_affine, 
		f_semilinear, f_special, 
		verbose_level);
	Mtx = A->G.matrix_grp;
	
	AF = NEW_OBJECT(action);

	AF->induced_action_on_flags(A, type, type_len, verbose_level);

	AF->print_info();


	strong_generators *SG;
	longinteger_object Borel_go;
	
	SG = NEW_OBJECT(strong_generators);
	SG->generators_for_the_borel_subgroup_lower(A, 
		Mtx, verbose_level);
	SG->group_order(Borel_go);
	
	cout << "Generators for the Borel subgroup:" << endl;
	SG->print_generators();

	schreier *Sch;

	//Sch = A->Strong_gens->orbits_on_points_schreier(AF, verbose_level + 10);
	Sch = SG->orbits_on_points_schreier(AF, verbose_level + 10);
	Sch->print_and_list_orbits(cout);
	Sch->print_and_list_orbits_and_stabilizer(cout, 
		A, Borel_go, 
		print_flag, NULL);


	FREE_int(subspace);
}

void print_flag(ostream &ost, int pt, void *data)
{
	flag *Flag;

	Flag = AF->G.OnFlags->Flag;

	Flag->unrank(pt, subspace, 0 /*verbose_level*/);
	int_matrix_print(subspace, flag_n, flag_n);
	
}

