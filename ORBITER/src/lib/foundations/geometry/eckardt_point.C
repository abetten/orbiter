// eckardt_point.C
//
// Anton Betten
// January 12, 2017

#include "foundations.h"

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
	INT t, i, j;
	
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
	INT t, i, j;
	
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
	INT t, i, j;
	
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

void eckardt_point::latex_to_str(BYTE *str)
{
	INT t, i, j;
	
	str[0] = 0;
	if (len == 3) {
		sprintf(str + strlen(str), "E_{");
		for (t = 0; t < 3; t++) {
			k2ij(index[t], i, j, 6);
			sprintf(str + strlen(str), "%ld%ld", i + 1, j + 1);
			if (t < 2) {
				sprintf(str + strlen(str), ",");
				}
			}
		sprintf(str + strlen(str), "}");
		}
	else if (len == 2) {
		sprintf(str + strlen(str), "E_{%ld%ld}", index[0] + 1, index[1] + 1);
		}
	else {
		cout << "eckardt_point::latex len is illegal" << endl;
		exit(1);
		}
}

void eckardt_point::latex_to_str_without_E(BYTE *str)
{
	INT t, i, j;
	
	str[0] = 0;
	if (len == 3) {
		//sprintf(str + strlen(str), "{");
		for (t = 0; t < 3; t++) {
			k2ij(index[t], i, j, 6);
			sprintf(str + strlen(str), "%ld%ld", i + 1, j + 1);
			if (t < 2) {
				sprintf(str + strlen(str), ",");
				}
			}
		//sprintf(str + strlen(str), "}");
		}
	else if (len == 2) {
		sprintf(str + strlen(str), "%ld%ld", index[0] + 1, index[1] + 1);
		}
	else {
		cout << "eckardt_point::latex len is illegal" << endl;
		exit(1);
		}
}



void eckardt_point::init2(INT i, INT j)
{
	len = 2;
	index[0] = i;
	index[1] = j;
}

void eckardt_point::init3(INT ij, INT kl, INT mn)
{
	len = 3;
	index[0] = ij;
	index[1] = kl;
	index[2] = mn;
}

void eckardt_point::init6(INT i, INT j, INT k, INT l, INT m, INT n)
{
	len = 3;
	index[0] = ij2k(i, j, 6);
	index[1] = ij2k(k, l, 6);
	index[2] = ij2k(m, n, 6);
}

void eckardt_point::init_by_rank(INT rk)
{
	if (rk < 30) {
		len = 2;
		ordered_pair_unrank(rk, index[0], index[1], 6);
		}
	else {
		INT i, j, k, l, m, n;
		
		len = 3;
		rk -= 30;
		unordered_triple_pair_unrank(rk, i, j, k, l, m, n);
		index[0] = ij2k(i, j, 6);
		index[1] = ij2k(k, l, 6);
		index[2] = ij2k(m, n, 6);
		}
}


void eckardt_point::three_lines(surface *S, INT *three_lines)
{
	if (len == 2) {
		three_lines[0] = S->line_ai(index[0]);
		three_lines[1] = S->line_bi(index[1]);
		three_lines[2] = S->line_cij(index[0], index[1]);
		}
	else if (len == 3) {
		INT i, j;

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

INT eckardt_point::rank()
{
	INT a;
	
	if (len == 2) {
		a = ordered_pair_rank(index[0], index[1], 6);
		return a;
		}
	else if (len == 3) {
		INT i, j, k, l, m, n;

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

void eckardt_point::unrank(INT rk, INT &i, INT &j, INT &k, INT &l, INT &m, INT &n)
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


