// eckardt_point.C
//
// Anton Betten
// January 12, 2017

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


eckardt_point::eckardt_point()
{
	null();
}

eckardt_point::~eckardt_point()
{
	freeself();
}

void eckardt_point::null()
{
}

void eckardt_point::freeself()
{
	null();
}

void eckardt_point::print()
{
	int t, i, j;
	
	if (len == 3) {
		cout << "E_{";
		for (t = 0; t < 3; t++) {
			k2ij(index[t], i, j, 6);
			cout << i + 1 << j + 1;
			if (t < 2) {
				cout << ",";
				}
			}
		cout << "}" << endl;
		}
	else if (len == 2) {
		cout << "E_{" << index[0] + 1 << index[1] + 1 << "}" << endl;
		}
	else {
		cout << "eckardt_point::print len is illegal" << endl;
		exit(1);
		}
}

void eckardt_point::latex(ostream &ost)
{
	int t, i, j;
	
	if (len == 3) {
		ost << "E_{";
		for (t = 0; t < 3; t++) {
			k2ij(index[t], i, j, 6);
			ost << i + 1 << j + 1;
			if (t < 2) {
				ost << ",";
				}
			}
		ost << "}" << endl;
		}
	else if (len == 2) {
		ost << "E_{" << index[0] + 1 << index[1] + 1 << "}" << endl;
		}
	else {
		cout << "eckardt_point::latex len is illegal" << endl;
		exit(1);
		}
}

void eckardt_point::latex_index_only(ostream &ost)
{
	int t, i, j;
	
	if (len == 3) {
		for (t = 0; t < 3; t++) {
			k2ij(index[t], i, j, 6);
			ost << i + 1 << j + 1;
			if (t < 2) {
				ost << ",";
				}
			}
		}
	else if (len == 2) {
		ost << index[0] + 1 << index[1] + 1;
		}
	else {
		cout << "eckardt_point::latex_index_only len is illegal" << endl;
		exit(1);
		}
}

void eckardt_point::latex_to_str(char *str)
{
	int t, i, j;
	
	str[0] = 0;
	if (len == 3) {
		sprintf(str + strlen(str), "E_{");
		for (t = 0; t < 3; t++) {
			k2ij(index[t], i, j, 6);
			sprintf(str + strlen(str), "%d%d", i + 1, j + 1);
			if (t < 2) {
				sprintf(str + strlen(str), ",");
				}
			}
		sprintf(str + strlen(str), "}");
		}
	else if (len == 2) {
		sprintf(str + strlen(str), "E_{%d%d}", index[0] + 1, index[1] + 1);
		}
	else {
		cout << "eckardt_point::latex len is illegal" << endl;
		exit(1);
		}
}

void eckardt_point::latex_to_str_without_E(char *str)
{
	int t, i, j;
	
	str[0] = 0;
	if (len == 3) {
		//sprintf(str + strlen(str), "{");
		for (t = 0; t < 3; t++) {
			k2ij(index[t], i, j, 6);
			sprintf(str + strlen(str), "%d%d", i + 1, j + 1);
			if (t < 2) {
				sprintf(str + strlen(str), ",");
				}
			}
		//sprintf(str + strlen(str), "}");
		}
	else if (len == 2) {
		sprintf(str + strlen(str), "%d%d", index[0] + 1, index[1] + 1);
		}
	else {
		cout << "eckardt_point::latex len is illegal" << endl;
		exit(1);
		}
}



void eckardt_point::init2(int i, int j)
{
	len = 2;
	index[0] = i;
	index[1] = j;
}

void eckardt_point::init3(int ij, int kl, int mn)
{
	len = 3;
	index[0] = ij;
	index[1] = kl;
	index[2] = mn;
}

void eckardt_point::init6(int i, int j, int k, int l, int m, int n)
{
	len = 3;
	index[0] = ij2k(i, j, 6);
	index[1] = ij2k(k, l, 6);
	index[2] = ij2k(m, n, 6);
}

void eckardt_point::init_by_rank(int rk)
{
	if (rk < 30) {
		len = 2;
		ordered_pair_unrank(rk, index[0], index[1], 6);
		}
	else {
		int i, j, k, l, m, n;
		
		len = 3;
		rk -= 30;
		unordered_triple_pair_unrank(rk, i, j, k, l, m, n);
		index[0] = ij2k(i, j, 6);
		index[1] = ij2k(k, l, 6);
		index[2] = ij2k(m, n, 6);
		}
}


void eckardt_point::three_lines(surface *S, int *three_lines)
{
	if (len == 2) {
		three_lines[0] = S->line_ai(index[0]);
		three_lines[1] = S->line_bi(index[1]);
		three_lines[2] = S->line_cij(index[0], index[1]);
		}
	else if (len == 3) {
		int i, j;

		k2ij(index[0], i, j, 6);
		three_lines[0] = S->line_cij(i, j);
		k2ij(index[1], i, j, 6);
		three_lines[1] = S->line_cij(i, j);
		k2ij(index[2], i, j, 6);
		three_lines[2] = S->line_cij(i, j);
		}
	else {
		cout << "eckardt_point::three_lines len must be 2 or 3" << endl;
		exit(1);
		}
}

int eckardt_point::rank()
{
	int a;
	
	if (len == 2) {
		a = ordered_pair_rank(index[0], index[1], 6);
		return a;
		}
	else if (len == 3) {
		int i, j, k, l, m, n;

		k2ij(index[0], i, j, 6);
		k2ij(index[1], k, l, 6);
		k2ij(index[2], m, n, 6);
		a = unordered_triple_pair_rank(i, j, k, l, m, n);
		return 30 + a;
		}
	else {
		cout << "eckardt_point::rank len must be 2 or 3" << endl;
		exit(1);
		}
}

void eckardt_point::unrank(int rk,
		int &i, int &j, int &k, int &l, int &m, int &n)
{
	if (rk < 30) {
		len = 2;
		ordered_pair_unrank(rk, index[0], index[1], 6);
		}
	else {
		rk -= 30;
		unordered_triple_pair_unrank(rk, i, j, k, l, m, n);
		index[0] = ij2k(i, j, 6);
		index[1] = ij2k(k, l, 6);
		index[2] = ij2k(m, n, 6);
		}
}


}
}

