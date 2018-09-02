// rank_anything.C
// 
// Anton Betten
// March 13, 2017

#include "orbiter.h"

void rank_subsets(INT n, INT verbose_level);
void rank_binary_trees(INT n, INT verbose_level);


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_subsets = FALSE;
	INT f_binary_trees = FALSE;
	INT f_n = FALSE;
	INT n;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-subsets") == 0) {
			f_subsets = TRUE;
			cout << "-subsets " << endl;
			}
		else if (strcmp(argv[i], "-binary_trees") == 0) {
			f_binary_trees = TRUE;
			cout << "-binary_trees " << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		}

	if (f_subsets) {
		if (!f_n) {
			cout << "Please use option -n <n>" << endl;
			exit(1);
			}
		rank_subsets(n, verbose_level);
		}
	if (f_binary_trees) {
		if (!f_n) {
			cout << "Please use option -n <n>" << endl;
			exit(1);
			}
		rank_binary_trees(n, verbose_level);
		}
	else {
		cout << "I don't know what to rank" << endl;
		exit(1);
		}
}

void rank_subsets(INT n, INT verbose_level)
{
	INT i;
	INT *set;
	INT N, r, sz;

	set = NEW_INT(n);

	N = i_power_j(2, n);
	for (i = 0; i < N; i++) {
		cout << " rank " << i << " : ";
		unrank_subset(set, sz, n, i);
		cout << " : set ";
		INT_vec_print(cout, set, sz);
		cout << " : ";
		r = rank_subset(set, sz, n);
		cout << " has rank " << r << endl;
		}

	char fname[1000];
	INT h;

	sprintf(fname, "subsets_of_%ld.tree", n);
	{
	ofstream fp(fname);

	N = i_power_j(2, n);
	for (h = 0; h < N; h++) {
		unrank_subset(set, sz, n, h);
		fp << sz;
		for (i = 0; i < sz; i++) {
			fp << " " << set[i];
			}
		fp << endl;
		}
	fp << "-1" << endl;
	}
#if 0
	set[0] = 1;
	set[1] = 3;
	sz = 2;
	r = rank_subset(set, sz, n);
	cout << "The set ";
	INT_vec_print(cout, set, sz);
	cout << " has rank " << r << endl;
#endif
	FREE_INT(set);
}

void rank_binary_trees(INT n, INT verbose_level)
{
	INT i;
	INT *v;
	INT N, r;

	v = NEW_INT(n);

	N = i_power_j(2, n);
	for (i = 0; i < N; i++) {
		cout << " rank " << i << " : ";
		AG_element_unrank(2 /* q */, v, 1, n, i);
		cout << " : bitstring ";
		INT_vec_print(cout, v, n);
		cout << " : ";
		AG_element_rank(2 /* q */, v, 1, n, r);
		cout << " has rank " << r << endl;
		if (r != i) {
			cout << "r != i, something is wrong" << endl;
			exit(1);
			}
		}

	char fname[1000];
	INT h;

	sprintf(fname, "binary_tree_of_depth_%ld.tree", n);
	{
	ofstream fp(fname);

	N = i_power_j(2, n);
	for (h = 0; h < N; h++) {
		AG_element_unrank(2 /* q */, v, 1, n, h);
		fp << n;
		for (i = 0; i < n; i++) {
			fp << " " << 2 * i + v[i];
			}
		fp << endl;
		}
	fp << "-1" << endl;
	}
#if 0
	set[0] = 1;
	set[1] = 3;
	sz = 2;
	r = rank_subset(set, sz, n);
	cout << "The set ";
	INT_vec_print(cout, set, sz);
	cout << " has rank " << r << endl;
#endif
	FREE_INT(v);
}


