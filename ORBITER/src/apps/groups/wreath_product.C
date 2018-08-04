// wreath_product.C
//
// Anton Betten
//
// August 4, 2018
//
//
//

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

void usage(int argc, char **argv);
int main(int argc, char **argv);
void do_it(INT nb_factors, INT n, INT q, INT verbose_level);

void usage(int argc, char **argv)
{
	cout << "usage: " << argv[0] << " [options]" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <n>                   : verbose level n" << endl;
	cout << "-nb_factors <nb_factors> : set number of factors" << endl;
	cout << "-d <d>                   : set dimension d" << endl;
	cout << "-q <q>                   : set field size q" << endl;
}



int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_nb_factors = FALSE;
	INT nb_factors = 0;
	INT f_d = FALSE;
	INT d = 0;
	INT f_q = FALSE;
	INT q = 0;

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
		else if (strcmp(argv[i], "-nb_factors") == 0) {
			f_nb_factors = TRUE;
			nb_factors = atoi(argv[++i]);
			cout << "-nb_factors " << nb_factors << endl;
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
	if (!f_nb_factors) {
		cout << "please use -nb_factors <nb_factors>" << endl;
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
	do_it(nb_factors, d, q, verbose_level);
}

void do_it(INT nb_factors, INT n, INT q, INT verbose_level)
{
	finite_field *F;
	action *A;
	INT f_basis = TRUE;
	INT p, h, i, j, a;
	INT *v;

	A = new action;
	is_prime_power(q, p, h);

	v = NEW_INT(n);


	F = new finite_field;

	F->init(q, 0);

	A->init_wreath_product_group(nb_factors, n,
			F,
			f_basis,
			verbose_level);


	if (!A->f_has_strong_generators) {
		cout << "action does not have strong generators" << endl;
		exit(1);
		}
	strong_generators *SG;
	longinteger_object go;
	//wreath_product *W = A->G.wreath_product_group;

	SG = A->Strong_gens;
	SG->group_order(go);

	cout << "The group " << A->label << " has order " << go << " and permutation degree " << A->degree << endl;

#if 0
	cout << "The points on which the group acts are:" << endl;
	for (i = 0; i < A->degree; i++) {
		O->unrank_point(v, 1 /* stride */, i, 0 /* verbose_level */);
		cout << i << " / " << A->degree << " : ";
		INT_vec_print(cout, v, n);
		cout << endl;
		}
#endif

	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / " << SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		cout << "as permutation: " << endl;
		A->element_print_as_permutation(SG->gens->ith(i), cout);
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
			a = A->element_image_of(j, SG->gens->ith(i), 0 /* verbose_level */);
			cout << a << " ";
			}
		cout << endl;
		}
	cout << "-1" << endl;
	FREE_INT(v);
	delete A;
	delete F;
}


