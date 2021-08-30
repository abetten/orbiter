// incidence.cpp

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {



incidence::incidence()
{
	gg = NULL;
	Encoding = NULL;

	//int theY[MAX_V * MAX_R];
	//int pairs[MAX_V][MAX_V];
	f_lambda = FALSE;
	lambda = 0;
	f_find_square = FALSE; /* JS 120100 */
	f_simple = FALSE; /* JS 180100 */

	/* initiale vbars / hbars: */
	nb_i_vbar = 0;
	i_vbar = NULL;
	nb_i_hbar = 0;
	i_hbar = NULL;

	gl_nb_GEN = 0;

	int i;

	for (i = 0; i < MAX_V; i++) {
		iso_type_at_line[i] = NULL;
	}
	iso_type_no_vhbars = NULL;

	back_to_line = 0;

}

incidence::~incidence()
{
	if (Encoding) {
		FREE_OBJECT(Encoding);
	}
	if (i_vbar) {
		FREE_int(i_vbar);
	}
	if (i_hbar) {
		FREE_int(i_hbar);
	}
}

void incidence::init(gen_geo *gg, int v, int b, int *R, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence::init" << endl;
	}

	int i;

	incidence::gg = gg;
	Encoding = NEW_OBJECT(inc_encoding);

	if (f_v) {
		cout << "incidence::init before Encoding->init" << endl;
	}
	Encoding->init(v, b, R, verbose_level);
	if (f_v) {
		cout << "incidence::init after Encoding->init" << endl;
	}


	if (f_v) {
		cout << "incidence::init before init_pairs" << endl;
	}
	init_pairs(verbose_level);
	if (f_v) {
		cout << "incidence::init after init_pairs" << endl;
	}

	f_find_square = TRUE;

	f_lambda = FALSE;
	lambda = 0;
	back_to_line = -1;
	for (i = 0; i < MAX_V; i++) {
		iso_type_at_line[i] = NULL;
	}
	iso_type_no_vhbars = NULL;

	gl_nb_GEN = 0;

	if (f_v) {
		cout << "incidence::init done" << endl;
	}
}

void incidence::init_bars(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "incidence::init_bars" << endl;
	}
	if (f_v) {
		cout << "incidence::init_bars before i_hbar" << endl;
	}
	i_vbar = NEW_int(gg->GB->b_len + 1);
	i_hbar = NEW_int(gg->GB->v_len + 1);

	nb_i_hbar = 0;
	i_hbar[nb_i_hbar++] = 0;


	nb_i_vbar = 0;

	for (j = 0; j < gg->GB->b_len; j++) {
		if (f_v) {
			cout << "j=" << j << endl;
		}
		gg->vbar[gg->Conf[0 * gg->GB->b_len + j].j0] = -1;

		i_vbar[nb_i_vbar++] = gg->Conf[0 * gg->GB->b_len + j].j0;

	}
	if (f_v) {
		cout << "incidence::init_bars done" << endl;
	}

}

void incidence::init_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i1, i2;

	if (f_v) {
		cout << "incidence::init_pairs" << endl;
	}
	for (i1 = 0; i1 < MAX_V; i1++) {
		for (i2 = 0; i2 < MAX_V; i2++) {
			pairs[i1][i2] = 0;
		}
	}
	if (f_v) {
		cout << "incidence::init_pairs done" << endl;
	}
}


int incidence::find_square(int m, int n)
{
	return Encoding->find_square(m, n);
}

void incidence::print_param()
{
	int i;

	cout << "V = " << Encoding->v << ", B = " << Encoding->b << endl;

	cout << "vbar: ";
	for (i = 0; i < nb_i_vbar; i++) {
		cout << i_vbar[i];
		if (i < nb_i_vbar - 1) {
			cout << ", ";
		}
	}
	cout << endl;

	cout << "hbar: ";
	for (i = 0; i < nb_i_hbar; i++) {
		cout << i_hbar[i];
		if (i < nb_i_hbar - 1) {
			cout << ", ";
		}
	}
	cout << endl;

}



void incidence::free_isot()
{
	int i;

	for (i = 0; i < Encoding->v; i++) {
		if (iso_type_at_line[i]) {
			delete iso_type_at_line[i];
		}
		iso_type_at_line[i] = NULL;
	}
	if (iso_type_no_vhbars) {
		delete iso_type_no_vhbars;
	}
	iso_type_no_vhbars = NULL;
}

void incidence::print_R(int v, cperm *p, cperm *q)
{
	int i;

	cout << "p = ";
	p->print();
	cout << ", q = ";
	q->print();
	cout << ", R=";
	for (i = 0; i < v; i++) {
		cout << Encoding->R[i];
		if (i < v - 1) {
			cout << ", ";
		}
	}
	cout << endl;
}



void incidence::print(std::ostream &ost, int v, int v_cut)
{
	Encoding->print_partitioned(ost, v, v_cut, this, TRUE /* f_print_isot */);
}


void incidence::print_override_theX(std::ostream &ost, int *theX, int v, int v_cut)
{
	Encoding->print_partitioned_override_theX(ost, v, v_cut, this, theX, TRUE /* f_print_isot */);
}


void incidence::stuetze_nach_zeile(int i, int tdo_flags, int verbose_level)
/* stuetze in letzter Zeile erlaubt */
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence::stuetze_nach_zeile line = " << i << endl;
	}
	if (i > 0 && i <= Encoding->v) {
		iso_type_at_line[i - 1] = new iso_type;
		iso_type_at_line[i - 1]->init(i, this, tdo_flags, verbose_level);
	}
	else {
		cout << "incidence::stuetze_nach_zeile "
				"out of range: i = " << i << ", v = " << Encoding->v << endl;
		exit(1);
	}
}

void incidence::stuetze2_nach_zeile(int i, int tdo_flags, int verbose_level)
/* stuetze 2 in letzter Zeile nicht erlaubt */
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence::stuetze2_nach_zeile line = " << i << endl;
	}
	if (i > 0 && i < Encoding->v) {
		iso_type_at_line[i - 1] = new iso_type;
		iso_type_at_line[i - 1]->init(i, this, tdo_flags, verbose_level);
		iso_type_at_line[i - 1]->second();
	}
	else {
		cout << "incidence::stuetze2_nach_zeile "
				"out of range: i = " << i << ", v = " << Encoding->v << endl;
		exit(1);
	}
}

void incidence::set_range(int i, int first, int len)
{
	if (i > 0 && i < Encoding->v) {
		iso_type_at_line[i - 1]->set_range(first, len);
	}
	else {
		cout << "incidence::set_range "
				"out of range: i = " << i << ", v = " << Encoding->v << endl;
		exit(1);
	}
}

void incidence::set_flush_to_inc_file(int i, std::string &fname)
// opens the geo_file and stores the file pointer is it->fp
{
	if (i > 0 && i <= Encoding->v) {
		//iso_type_at_line[i - 1]->open_inc_file(fname);
	}

}


void incidence::set_flush_line(int i)
{
	iso_type *it;

	if (i > 0 && i < Encoding->v) {
		it = iso_type_at_line[i - 1];
		it->set_flush_line();
	}
	else {
		cout << "incidence::incidence_set_flush_line out of range: i = " << i << ", v = " << Encoding->v << endl;
		exit(1);
	}
}


void incidence::print_geo(std::ostream &ost, int v, int *theGEO)
{
	int i, j, s, a;

	s = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < Encoding->R[i]; j++, s++) {
			a = theGEO[s];
			ost << i * Encoding->b + a << " ";
		}
	}

}

void incidence::print_inc(std::ostream &ost, int v, long int *theInc)
{
	int i, j, s;
	long int a;

	s = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < Encoding->R[i]; j++, s++) {
			a = theInc[s];
			ost << a << " ";
		}
	}

}

void incidence::print_blocks(std::ostream &ost, int v, long int *theInc)
{
	long int *Blocks;
	int i, b;

	compute_blocks(Blocks, v, theInc);

	b = Encoding->b;

	for (i = 0; i < b; i++) {
		ost << Blocks[i] << " ";
	}
	FREE_lint(Blocks);
}

void incidence::compute_blocks(long int *&Blocks, int v, long int *theInc)
{
	int i, j, s, b, k, h;
	long int a;
	int *Incma;
	int *block;
	combinatorics_domain Combi;

	b = Encoding->b;
	Incma = NEW_int(v * b);
	block = NEW_int(v);
	Blocks = NEW_lint(b);
	for (i = 0; i < v * b; i++) {
		Incma[i] = 0;
	}

	s = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < Encoding->R[i]; j++, s++) {
			a = theInc[s];
			Incma[a] = 1;
		}
	}
	for (j = 0; j < b; j++) {
		for (i = 0; i < v; i++) {
			block[i] = 0;
		}
		h = 0;
		for (i = 0; i < v; i++) {
			if (Incma[i * b + j]) {
				block[h++] = i;
			}
		}
		if (j == 0) {
			k = h;
		}
		else {
			if (k != h) {
				cout << "incidence::compute_blocks not column tactical" << endl;
				exit(1);
			}
		}
		Blocks[j] = Combi.rank_k_subset(block, v, k);
	}
	FREE_int(block);
	FREE_int(Incma);
}

int incidence::compute_k(int v, long int *theInc)
{
	int i, j, s, b, k, h;
	long int a;
	int *Incma;
	int *block;
	combinatorics_domain Combi;

	b = Encoding->b;
	Incma = NEW_int(v * b);
	block = NEW_int(v);
	for (i = 0; i < v * b; i++) {
		Incma[i] = 0;
	}

	s = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < Encoding->R[i]; j++, s++) {
			a = theInc[s];
			Incma[a] = 1;
		}
	}
	for (j = 0; j < b; j++) {
		h = 0;
		for (i = 0; i < v; i++) {
			if (Incma[i * b + j]) {
				block[h++] = i;
			}
		}
		if (j == 0) {
			k = h;
		}
		else {
			if (k != h) {
				cout << "incidence::compute_blocks not column tactical" << endl;
				exit(1);
			}
		}
	}
	FREE_int(block);
	FREE_int(Incma);
	return k;
}

int incidence::is_block_tactical(int v, long int *theInc)
{
	int i, j, s, b, k, h, ret;
	long int a;
	int *Incma;
	int *block;
	combinatorics_domain Combi;

	b = Encoding->b;
	Incma = NEW_int(v * b);
	block = NEW_int(v);
	for (i = 0; i < v * b; i++) {
		Incma[i] = 0;
	}

	ret = TRUE;
	s = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < Encoding->R[i]; j++, s++) {
			a = theInc[s];
			Incma[a] = 1;
		}
	}
	for (j = 0; j < b; j++) {
		h = 0;
		for (i = 0; i < v; i++) {
			if (Incma[i * b + j]) {
				block[h++] = i;
			}
		}
		if (j == 0) {
			k = h;
		}
		else {
			if (k != h) {
				ret = FALSE;
				break;
			}
		}
	}
	FREE_int(block);
	FREE_int(Incma);
	return ret;
}


void incidence::geo_to_inc(int v, int *theGEO, long int *theInc, int nb_flags)
{
	int i, j, s, a;

	s = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < Encoding->R[i]; j++, s++) {
			a = theGEO[s];
			theInc[s] = i * Encoding->b + a;
		}
	}
	if (s != nb_flags) {
		cout << "incidence::geo_to_inc s != nb_flags" << endl;
		exit(1);
	}

}


}}


