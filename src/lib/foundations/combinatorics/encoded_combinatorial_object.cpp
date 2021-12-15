/*
 * encoded_combinatorial_object.cpp
 *
 *  Created on: Aug 22, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

encoded_combinatorial_object::encoded_combinatorial_object()
{
	Incma = NULL;
	nb_rows = 0;
	nb_cols = 0;
	partition = NULL;
	canonical_labeling_len = 0;
}

encoded_combinatorial_object::~encoded_combinatorial_object()
{
	if (Incma) {
		FREE_int(Incma);
	}
	if (partition) {
		FREE_int(partition);
	}
}


void encoded_combinatorial_object::init(int nb_rows, int nb_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::init" << endl;
		cout << "encoded_combinatorial_object::init "
				"nb_rows=" << nb_rows << " nb_cols=" << nb_cols << endl;
	}
	int i, L;

	encoded_combinatorial_object::nb_rows = nb_rows;
	encoded_combinatorial_object::nb_cols = nb_cols;
	L = nb_rows * nb_cols;
	canonical_labeling_len = nb_rows + nb_cols;

	Incma = NEW_int(L);
	partition = NEW_int(canonical_labeling_len);

	Orbiter->Int_vec.zero(Incma, L);
	for (i = 0; i < canonical_labeling_len; i++) {
		partition[i] = 1;
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::init done" << endl;
	}
}

void encoded_combinatorial_object::print_incma()
{
	Orbiter->Int_vec.matrix_print_tight(Incma, nb_rows, nb_cols);

}

void encoded_combinatorial_object::print_partition()
{
	int i;

	for (i = 0; i < canonical_labeling_len; i++) {
		//cout << i << " : " << partition[i] << endl;
		cout << partition[i];
	}
	cout << endl;

}

void encoded_combinatorial_object::compute_canonical_incma(int *canonical_labeling,
		int *&Incma_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::compute_canonical_incma" << endl;
	}

	int i, j, ii, jj;

	Incma_out = NEW_int(nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		ii = canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = canonical_labeling[nb_rows + j] - nb_rows;
			//cout << "i=" << i << " j=" << j << " ii=" << ii
			//<< " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = Incma[ii * nb_cols + jj];
		}
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::compute_canonical_incma done" << endl;
	}
}

void encoded_combinatorial_object::compute_canonical_form(bitvector *&Canonical_form,
		int *canonical_labeling, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::compute_canonical_form" << endl;
	}
	int *Incma_out;
	int L;
	int i, j, a;

	L = nb_rows * nb_cols;

	compute_canonical_incma(canonical_labeling,
			Incma_out, verbose_level);


	Canonical_form = NEW_OBJECT(bitvector);
	Canonical_form->allocate(L);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (Incma_out[i * nb_cols + j]) {
				a = i * nb_cols + j;
				Canonical_form->m_i(a, 1);
			}
		}
	}
	FREE_int(Incma_out);

	if (f_v) {
		cout << "encoded_combinatorial_object::compute_canonical_form done" << endl;
	}
}

void encoded_combinatorial_object::incidence_matrix_projective_space_top_left(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::incidence_matrix_projective_space_top_left" << endl;
	}
	int i, j;

	for (i = 0; i < P->N_points; i++) {
		for (j = 0; j < P->N_lines; j++) {
			Incma[i * nb_cols + j] = P->is_incident(i, j);
		}
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::incidence_matrix_projective_space_top_left done" << endl;
	}
}

void encoded_combinatorial_object::canonical_form_given_canonical_labeling(int *canonical_labeling,
		bitvector *&B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::canonical_form_given_canonical_labeling" << endl;
	}


	int *Incma_out;
	int i, j, a;

	compute_canonical_incma(canonical_labeling, Incma_out, verbose_level);


	B->allocate(nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (Incma_out[i * nb_cols + j]) {
				a = i * nb_cols + j;
				B->m_i(a, 1);
			}
		}
	}

	FREE_int(Incma_out);


	if (f_v) {
		cout << "encoded_combinatorial_object::canonical_form_given_canonical_labeling done" << endl;
	}
}

void encoded_combinatorial_object::latex_incma(std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma" << endl;
	}

	latex_interface L;

	int *Vi;
	int *Bj;
	int V, B;
	int i, j, i0, j0;

	V = 0;
	B = 0;
	Vi = NEW_int(nb_rows);
	Bj = NEW_int(nb_cols);

	i0 = 0;
	for (i = 0; i < nb_rows; i++) {
		if (partition[i] == 0) {
			Vi[V++] = i - i0 + 1;
			i0 = i + 1;
		}
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma Vi=";
		Orbiter->Int_vec.print(cout, Vi, V);
		cout << endl;
	}
	j0 = 0;
	for (j = 0; j < nb_cols; j++) {
		if (partition[nb_rows + j] == 0) {
			Bj[B++] = j - j0 + 1;
			j0 = j + 1;
		}
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma Bj=";
		Orbiter->Int_vec.print(cout, Bj, B);
		cout << endl;
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma before L.incma_latex" << endl;
	}
	L.incma_latex(ost,
		nb_rows /*v */,
		nb_cols /*b */,
		V, B, Vi, Bj,
		Incma,
		verbose_level - 1);
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma after L.incma_latex" << endl;
	}

	ost << "\\\\" << endl;

	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma done" << endl;
	}
}




#if 0
void projective_space_with_action::save_Levi_graph(std::string &prefix,
		const char *mask,
		int *Incma, int nb_rows, int nb_cols,
		long int *canonical_labeling, int canonical_labeling_len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::save_Levi_graph" << endl;
	}
	file_io Fio;
	string fname_csv;
	string fname_bin;
	string fname_labeling;
	char str[1000];

	sprintf(str, mask, nb_rows, nb_cols);

	fname_csv.assign(prefix);
	fname_csv.append(str);
	fname_csv.append(".csv");

	fname_bin.assign(prefix);
	fname_bin.append(str);
	fname_bin.append(".graph");


	fname_labeling.assign(prefix);
	fname_labeling.append("_labeling");
	fname_labeling.append(".csv");

	latex_interface L;

#if 0
	cout << "labeling:" << endl;
	L.lint_vec_print_as_matrix(cout,
			canonical_labeling, N, 10 /* width */, TRUE /* f_tex */);
#endif

	Fio.lint_vec_write_csv(canonical_labeling, canonical_labeling_len,
			fname_labeling, "can_lab");
	Fio.int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);


	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	CG->create_Levi_graph_from_incidence_matrix(
			Incma, nb_rows, nb_cols,
			TRUE, canonical_labeling, verbose_level);
	CG->save(fname_bin, verbose_level);
	FREE_OBJECT(CG);
	if (f_v) {
		cout << "projective_space_with_action::save_Levi_graph done" << endl;
	}
}
#endif


void encoded_combinatorial_object::latex_TDA(std::ostream &ost,
		int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA" << endl;
	}

	latex_interface L;

	int *Vi;
	int *Bj;
	int V, B;
	int i, j;
	int fst, len;

	V = 0;
	B = 0;
	Vi = NEW_int(nb_rows);
	Bj = NEW_int(nb_cols);

	for (i = 0; i < nb_orbits; i++) {
		fst = orbit_first[i];
		if (fst == nb_rows) {
			break;
		}
		len = orbit_len[i];
		Vi[V++] = len;
	}
	for (; i < nb_orbits; i++) {
		fst = orbit_first[i];
		len = orbit_len[i];
		Bj[B++] = len;
	}


	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA Vi=";
		Orbiter->Int_vec.print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA Bj=";
		Orbiter->Int_vec.print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Orbiter->Int_vec.zero(Inc2, nb_rows * nb_cols);

	for (i = 0; i < nb_rows; i++) {
		i0 = orbit[i];
		for (j = 0; j < nb_cols; j++) {
			j0 = orbit[nb_rows + j] - nb_rows;
			if (Incma[i0 * nb_cols + j0]) {
				Inc2[i * nb_cols + j] = 1;
			}
		}
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA "
				"before L.incma_latex" << endl;
	}
	L.incma_latex(ost,
			nb_rows /*v */,
			nb_cols /*b */,
			V, B, Vi, Bj,
			Inc2,
			verbose_level - 1);
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA "
				"after L.incma_latex" << endl;
	}

	ost << "\\\\" << endl;

	FREE_int(Inc2);
	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA done" << endl;
	}
}


void encoded_combinatorial_object::latex_TDA_with_labels(std::ostream &ost,
		int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels" << endl;
	}

	latex_interface L;

	int *Vi;
	int *Bj;
	int V, B;
	int i, j;
	int fst, len;

	V = 0;
	B = 0;
	Vi = NEW_int(nb_rows);
	Bj = NEW_int(nb_cols);

	for (i = 0; i < nb_orbits; i++) {
		fst = orbit_first[i];
		if (fst == nb_rows) {
			break;
		}
		len = orbit_len[i];
		Vi[V++] = len;
	}
	for (; i < nb_orbits; i++) {
		fst = orbit_first[i];
		len = orbit_len[i];
		Bj[B++] = len;
	}


	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels Vi=";
		Orbiter->Int_vec.print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels Bj=";
		Orbiter->Int_vec.print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Orbiter->Int_vec.zero(Inc2, nb_rows * nb_cols);

	for (i = 0; i < nb_rows; i++) {
		i0 = orbit[i];
		for (j = 0; j < nb_cols; j++) {
			j0 = orbit[nb_rows + j] - nb_rows;
			if (Incma[i0 * nb_cols + j0]) {
				Inc2[i * nb_cols + j] = 1;
			}
		}
	}

	int v = nb_rows;
	int b = nb_cols;

	std::string *point_labels;
	std::string *block_labels;


	point_labels = new string [v];
	block_labels = new string [b];

	for (i = 0; i < v; i++) {
		char str[1000];

		sprintf(str, "%d", orbit[i]);
		point_labels[i].assign(str);
	}


	for (j = 0; j < b; j++) {
		char str[1000];

		sprintf(str, "%d", orbit[nb_rows + j]);
		block_labels[j].assign(str);
	}

	draw_incidence_structure_description *Descr;

	Descr = Orbiter->Draw_incidence_structure_description;


	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels "
				"before L.incma_latex_with_text_labels" << endl;
	}
	L.incma_latex_with_text_labels(ost,
			Descr,
			nb_rows /*v */,
			nb_cols /*b */,
			V, B, Vi, Bj,
			Inc2,
			TRUE, point_labels,
			TRUE, block_labels,
			verbose_level - 1);
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels "
				"after L.incma_latex_with_text_labels" << endl;
	}

	ost << "\\\\" << endl;

	delete [] point_labels;
	delete [] block_labels;

	FREE_int(Inc2);
	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels done" << endl;
	}
}


void encoded_combinatorial_object::latex_canonical_form(std::ostream &ost,
		nauty_output *NO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form" << endl;
	}

	latex_interface L;

	int *Vi;
	int *Bj;
	int V, B;
	int i, j;
	//int fst, len;

	V = 0;
	B = 0;
	Vi = NEW_int(nb_rows);
	Bj = NEW_int(nb_cols);

	Vi[V++] = nb_rows;
	Bj[B++] = nb_cols;
#if 0
	for (i = 0; i < nb_orbits; i++) {
		fst = orbit_first[i];
		if (fst == nb_rows) {
			break;
		}
		len = orbit_len[i];
		Vi[V++] = len;
	}
	for (; i < nb_orbits; i++) {
		fst = orbit_first[i];
		len = orbit_len[i];
		Bj[B++] = len;
	}
#endif

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form Vi=";
		Orbiter->Int_vec.print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form Bj=";
		Orbiter->Int_vec.print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Orbiter->Int_vec.zero(Inc2, nb_rows * nb_cols);

	for (i = 0; i < nb_rows; i++) {
		i0 = NO->canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			j0 =  NO->canonical_labeling[nb_rows + j] - nb_rows;
			if (Incma[i0 * nb_cols + j0]) {
				Inc2[i * nb_cols + j] = 1;
			}
		}
	}

	ost << "Flags : ";

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (Inc2[i * nb_cols + j]) {
				ost << i * nb_cols + j << ",";
			}
		}
	}
	ost << "\\\\" << endl;

	int *row_labels_int;
	int *col_labels_int;


	row_labels_int = NEW_int(nb_rows);
	col_labels_int = NEW_int(nb_cols);

	for (i = 0; i < nb_rows; i++) {
		row_labels_int[i] = NO->canonical_labeling[i];
	}
	for (j = 0; j < nb_cols; j++) {
		//col_labels_int[j] = NO->canonical_labeling[nb_rows + j] - nb_rows;
		col_labels_int[j] = NO->canonical_labeling[nb_rows + j];
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form "
				"before L.incma_latex" << endl;
	}
	L.incma_latex_with_labels(ost,
			nb_rows /*v */,
			nb_cols /*b */,
			V, B, Vi, Bj,
			row_labels_int,
			col_labels_int,
			Inc2,
			verbose_level - 1);
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form "
				"after L.incma_latex" << endl;
	}

	FREE_int(row_labels_int);
	FREE_int(col_labels_int);


	ost << "\\\\" << endl;

	FREE_int(Inc2);
	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form done" << endl;
	}
}


void encoded_combinatorial_object::latex_canonical_form_with_labels(std::ostream &ost,
		nauty_output *NO,
		std::string *row_labels,
		std::string *col_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form" << endl;
	}

	latex_interface L;

	int *Vi;
	int *Bj;
	int V, B;
	int i, j;
	//int fst, len;

	V = 0;
	B = 0;
	Vi = NEW_int(nb_rows);
	Bj = NEW_int(nb_cols);

	Vi[V++] = nb_rows;
	Bj[B++] = nb_cols;
#if 0
	for (i = 0; i < nb_orbits; i++) {
		fst = orbit_first[i];
		if (fst == nb_rows) {
			break;
		}
		len = orbit_len[i];
		Vi[V++] = len;
	}
	for (; i < nb_orbits; i++) {
		fst = orbit_first[i];
		len = orbit_len[i];
		Bj[B++] = len;
	}
#endif

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form Vi=";
		Orbiter->Int_vec.print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form Bj=";
		Orbiter->Int_vec.print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Orbiter->Int_vec.zero(Inc2, nb_rows * nb_cols);

	for (i = 0; i < nb_rows; i++) {
		i0 = NO->canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			j0 =  NO->canonical_labeling[nb_rows + j] - nb_rows;
			if (Incma[i0 * nb_cols + j0]) {
				Inc2[i * nb_cols + j] = 1;
			}
		}
	}

	ost << "Flags : ";

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (Inc2[i * nb_cols + j]) {
				ost << i * nb_cols + j << ", ";
			}
		}
	}
	ost << "\\\\" << endl;


	draw_incidence_structure_description *Descr;

	Descr = Orbiter->Draw_incidence_structure_description;

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form "
				"before L.incma_latex_with_text_labels" << endl;
	}
	L.incma_latex_with_text_labels(ost,
			Descr,
			nb_rows /*v */,
			nb_cols /*b */,
			V, B, Vi, Bj,
			Inc2,
			TRUE, row_labels,
			TRUE, col_labels,
			verbose_level - 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form "
				"after L.incma_latex_with_text_labels" << endl;
	}


	ost << "\\\\" << endl;

	FREE_int(Inc2);
	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form done" << endl;
	}
}




}}
