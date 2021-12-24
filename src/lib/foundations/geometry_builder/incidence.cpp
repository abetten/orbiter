// incidence.cpp

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {



incidence::incidence()
{
	gg = NULL;
	Encoding = NULL;

	K = NULL;


	theY = NULL;

	pairs = NULL;

#if 0
	// initial vbar / hbar
	nb_i_vbar = 0;
	i_vbar = NULL;
	nb_i_hbar = 0;
	i_hbar = NULL;
#endif

	row_partition = NULL;
	col_partition = NULL;
	Partition = NULL;

	gl_nb_GEN = 0;

	iso_type_at_line = NULL;
	iso_type_no_vhbars = NULL;

	back_to_line = 0;

}

incidence::~incidence()
{
	if (K) {
		FREE_int(K);
	}
	if (theY) {
		int i;

		for (i = 0; i < gg->GB->B; i++) {
			FREE_int(theY[i]);
		}
		FREE_pint(theY);
	}
	if (pairs) {
		int i;

		for (i = 1; i < gg->GB->V; i++) {
			FREE_int(pairs[i]);
		}
		FREE_pint(pairs);
	}
	if (Encoding) {
		FREE_OBJECT(Encoding);
	}
#if 0
	if (i_vbar) {
		FREE_int(i_vbar);
	}
	if (i_hbar) {
		FREE_int(i_hbar);
	}
#endif

	if (row_partition) {
		FREE_int(row_partition);
	}
	if (col_partition) {
		FREE_int(col_partition);
	}
	if (Partition) {
		int i;

		for (i = 0; i <= gg->GB->V; i++) {
			FREE_int(Partition[i]);
		}
		FREE_pint(Partition);
	}
	if (iso_type_at_line) {
		int i;

		for (i = 0; i < gg->GB->V; i++) {
			if (iso_type_at_line[i]) {
				FREE_OBJECT(iso_type_at_line[i]);
			}
		}
		FREE_pvoid((void **) iso_type_at_line);
	}

	if (iso_type_no_vhbars) {
		FREE_OBJECT(iso_type_no_vhbars);
	}

}

void incidence::init(gen_geo *gg, int v, int b, int *R, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence::init" << endl;
	}

	int i, j;

	incidence::gg = gg;
	Encoding = NEW_OBJECT(inc_encoding);

	if (f_v) {
		cout << "incidence::init before Encoding->init" << endl;
	}
	Encoding->init(v, b, R, verbose_level);
	if (f_v) {
		cout << "incidence::init after Encoding->init" << endl;
	}


	K = NEW_int(gg->GB->B);
	for (j = 0; j < gg->GB->B; j++) {
		K[j] = 0;
	}

	theY = NEW_pint(gg->GB->B);
	for (j = 0; j < gg->GB->B; j++) {
		theY[j] = NEW_int(gg->GB->V);
	}


	iso_type_at_line = (iso_type **) NEW_pvoid(gg->GB->V);
	for (i = 0; i < gg->GB->V; i++) {
		iso_type_at_line[i] = NULL;
	}
	iso_type_no_vhbars = NULL;

	if (f_v) {
		cout << "incidence::init before init_pairs" << endl;
	}
	init_pairs(verbose_level);
	if (f_v) {
		cout << "incidence::init after init_pairs" << endl;
	}


	back_to_line = -1;

	gl_nb_GEN = 0;

	if (f_v) {
		cout << "incidence::init done" << endl;
	}
}

#if 0
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



	nb_i_vbar = 0;

	for (j = 0; j < gg->GB->b_len; j++) {
		if (f_v) {
			cout << "j=" << j << endl;
		}
		gg->vbar[gg->Conf[0 * gg->GB->b_len + j].j0] = -1;

		i_vbar[nb_i_vbar++] = gg->Conf[0 * gg->GB->b_len + j].j0;

	}

	if (gg->GB->Descr->f_orderly) {
		int i;

		// set all hbars because we are doing orderly generation:
		for (i = 0; i <= gg->GB->V; i++) {
			gg->hbar[i] = -1;
		}
#if 0
		nb_i_hbar = 0;
		for (i = 0; i <= gg->GB->V; i++) {
			i_hbar[nb_i_hbar++] = i;
		}
#endif
	}
	else {
		nb_i_hbar = 0;
		i_hbar[nb_i_hbar++] = 0;
	}

	if (f_v) {
		cout << "incidence::init_bars done" << endl;
	}

}
#endif

void incidence::init_partition(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, I, J;

	if (f_v) {
		cout << "incidence::init_partition" << endl;
	}
	row_partition = NEW_int(gg->GB->V);
	col_partition = NEW_int(gg->GB->B);

	for (i = 0; i < gg->GB->V; i++) {
		row_partition[i] = 1;
	}
	for (j = 0; j < gg->GB->B; j++) {
		col_partition[j] = 1;
	}


	for (I = 0; I < gg->GB->v_len; I++) {

		i = gg->Conf[I * gg->GB->b_len + 0].i0 + gg->Conf[I * gg->GB->b_len + 0].v - 1;

		if (f_v) {
			cout << "I=" << I << " i=" << i << endl;
		}

		row_partition[i] = 0;

	}


	for (J = 0; J < gg->GB->b_len; J++) {

		j = gg->Conf[0 * gg->GB->b_len + J].j0 + gg->Conf[0 * gg->GB->b_len + J].b - 1;

		if (f_v) {
			cout << "J=" << J << " j=" << j << endl;
		}

		col_partition[j] = 0;

	}

	if (f_v) {
		cout << "row_partition: ";
		Orbiter->Int_vec.print(cout, row_partition, gg->GB->V);
		cout << endl;

		cout << "col_partition: ";
		Orbiter->Int_vec.print(cout, col_partition, gg->GB->B);
		cout << endl;
	}

	Partition = NEW_pint(gg->GB->V + 1);

	for (i = 0; i <= gg->GB->V; i++) {
		Partition[i] = NEW_int(i + gg->GB->B);
		Orbiter->Int_vec.copy(row_partition, Partition[i], i);
		if (i) {
			Partition[i][i - 1] = 0;
		}
		Orbiter->Int_vec.copy(col_partition, Partition[i] + i, gg->GB->B);

		if (f_v) {
			cout << "Partition[" << i << "]: ";
			Orbiter->Int_vec.print(cout, Partition[i], i + gg->GB->B);
			cout << endl;
		}
	}


	if (f_v) {
		cout << "incidence::init_partition done" << endl;
	}

}


void incidence::init_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i1, i2;

	if (f_v) {
		cout << "incidence::init_pairs" << endl;
	}

	pairs = NEW_pint(gg->GB->V);

	for (i1 = 1; i1 <= gg->GB->V; i1++) {
		pairs[i1] = NEW_int(i1 - 1);
		for (i2 = 0; i2 < i1 - 1; i2++) {
			pairs[i1][i2] = 0;
		}
	}
	if (f_v) {
		cout << "incidence::init_pairs done" << endl;
	}
}

void incidence::print_pairs(int v)
{
	int i1, i2, a;
	int *M;

	M = NEW_int(v * v);
	for (i1 = 0; i1 < v; i1++) {
		//cout << i1 << " : ";
		for (i2 = 0; i2 < v; i2++) {
			if (i2 == i1) {
				a = 0;
			}
			else if (i2 < i1) {
				a = pairs[i1][i2];
			}
			else {
				a = pairs[i2][i1];
			}
			M[i1 * v + i2] = a;
		}
	}
	Orbiter->Int_vec.matrix_print(M, v, v);
	FREE_int(M);
}


int incidence::find_square(int m, int n)
{
	return Encoding->find_square(m, n);
}

void incidence::print_param()
{
	int i;

	cout << "V = " << Encoding->v << ", B = " << Encoding->b << endl;

#if 0
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
#endif

}



void incidence::free_isot()
{
	int i;

	for (i = 0; i < Encoding->v; i++) {
		if (iso_type_at_line[i]) {
			FREE_OBJECT(iso_type_at_line[i]);
		}
		iso_type_at_line[i] = NULL;
	}
	if (iso_type_no_vhbars) {
		FREE_OBJECT(iso_type_no_vhbars);
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
	Encoding->print_partitioned(ost,
			v, v_cut, this, TRUE /* f_print_isot */);
}


void incidence::print_override_theX(std::ostream &ost, int *theX, int v, int v_cut)
{
	Encoding->print_partitioned_override_theX(ost,
			v, v_cut, this, theX, TRUE /* f_print_isot */);
}


void incidence::install_isomorphism_test_after_a_given_row(
		int row,
		int tdo_flags, int f_orderly,
		int verbose_level)
// last row is ok
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence::install_isomorphism_test_after_a_given_row line = " << row << endl;
	}
	if (row > 0 && row <= Encoding->v) {
		iso_type_at_line[row - 1] = NEW_OBJECT(iso_type);
		iso_type_at_line[row - 1]->init(row, this, tdo_flags, f_orderly, verbose_level);
	}
	else {
		cout << "incidence::install_isomorphism_test_after_a_given_row "
				"out of range: i = " << row << ", v = " << Encoding->v << endl;
		exit(1);
	}
}

void incidence::install_isomorphism_test_of_second_kind_after_a_given_row(
		int row,
		int tdo_flags, int f_orderly,
		int verbose_level)
// last row is not allowed
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence::install_isomorphism_test_of_second_kind_after_a_given_row "
				"line = " << row << endl;
	}
	if (row > 0 && row < Encoding->v) {
		iso_type_at_line[row - 1] = NEW_OBJECT(iso_type);
		iso_type_at_line[row - 1]->init(row, this, tdo_flags, f_orderly, verbose_level);
		iso_type_at_line[row - 1]->second();
	}
	else {
		cout << "incidence::install_isomorphism_test_of_second_kind_after_a_given_row "
				"out of range: row = " << row << ", v = " << Encoding->v << endl;
		exit(1);
	}
}

void incidence::set_split(int row, int remainder, int modulo)
{
	if (row > 0 && row < Encoding->v) {
		iso_type_at_line[row - 1]->set_split(remainder, modulo);
	}
	else {
		cout << "incidence::set_range "
				"out of range: row = " << row << ", v = " << Encoding->v << endl;
		exit(1);
	}
}

void incidence::set_flush_to_inc_file(int row, std::string &fname)
// opens the geo_file and stores the file pointer is it->fp
{
	if (row > 0 && row <= Encoding->v) {
		//iso_type_at_line[i - 1]->open_inc_file(fname);
	}

}


void incidence::set_flush_line(int row)
{
	iso_type *it;

	if (row > 0 && row < Encoding->v) {
		it = iso_type_at_line[row - 1];
		it->set_flush_line();
	}
	else {
		cout << "incidence::incidence_set_flush_line out of range: "
				"row = " << row << ", v = " << Encoding->v << endl;
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

	compute_blocks_ranked(Blocks, v, theInc);

	b = Encoding->b;

	for (i = 0; i < b; i++) {
		ost << Blocks[i] << " ";
	}
	FREE_lint(Blocks);
}

void incidence::compute_blocks(long int *&Blocks, int *&K, int v, long int *theInc)
{
	int i, j, s, b, h;
	long int a;
	int *Incma;
	combinatorics_domain Combi;

	b = Encoding->b;

	Incma = NEW_int(v * b);

	K = NEW_int(b);
	for (i = 0; i < b; i++) {
		K[i] = 0;
	}

	Blocks = NEW_lint(b * v);
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
				Blocks[j * v + h++] = i;
			}
		}
		K[j] = h;
	}
	FREE_int(Incma);
}


void incidence::compute_blocks_ranked(long int *&Blocks, int v, long int *theInc)
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
				cout << "incidence::compute_blocks_ranked not column tactical" << endl;
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
			a = theGEO[i * Encoding->dim_n + j];
			theInc[s] = i * Encoding->b + a;
		}
	}
	if (s != nb_flags) {
		cout << "incidence::geo_to_inc s != nb_flags" << endl;
		exit(1);
	}

}

void incidence::inc_to_geo(int v, long int *theInc, int *theGEO, int nb_flags)
{
	int i, j, s, a;

	s = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < Encoding->R[i]; j++, s++) {
			a = theInc[s] - i * Encoding->b;
			theGEO[i * Encoding->dim_n + j] = a;
		}
	}
	if (s != nb_flags) {
		cout << "incidence::inc_to_geo s != nb_flags" << endl;
		exit(1);
	}

}


}}


