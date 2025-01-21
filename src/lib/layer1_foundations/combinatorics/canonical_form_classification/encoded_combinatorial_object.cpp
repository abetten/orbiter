/*
 * encoded_combinatorial_object.cpp
 *
 *  Created on: Aug 22, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace canonical_form_classification {


encoded_combinatorial_object::encoded_combinatorial_object()
{
	Record_birth();
	Incma = NULL;
	nb_rows0 = 0;
	nb_cols0 = 0;
	nb_flags = 0;
	nb_rows = 0;
	nb_cols = 0;
	partition = NULL;
	canonical_labeling_len = 0;

	invariant_set_start = 0;
	invariant_set_size = 0;
}

encoded_combinatorial_object::~encoded_combinatorial_object()
{
	Record_death();
	if (Incma) {
		FREE_int(Incma);
	}
	if (partition) {
		FREE_int(partition);
	}
}

void encoded_combinatorial_object::init_everything(
		int nb_rows, int nb_cols,
		int *Incma, int *partition,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::init_everything" << endl;
	}
	encoded_combinatorial_object::nb_rows = nb_rows;
	encoded_combinatorial_object::nb_cols = nb_cols;
	encoded_combinatorial_object::Incma = Incma;
	encoded_combinatorial_object::partition = partition;
	if (f_v) {
		cout << "encoded_combinatorial_object::init_everything done" << endl;
	}

}

void encoded_combinatorial_object::init_with_matrix(
		int nb_rows, int nb_cols, int *incma,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::init_with_matrix" << endl;
		cout << "encoded_combinatorial_object::init_with_matrix "
				"nb_rows=" << nb_rows << " nb_cols=" << nb_cols << endl;
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::init_with_matrix "
				"before init" << endl;
	}
	init(
			nb_rows, nb_cols,
			verbose_level);
	if (f_v) {
		cout << "encoded_combinatorial_object::init_with_matrix "
				"after init" << endl;
	}

	Int_vec_copy(incma, Incma, nb_rows * nb_cols);

	if (f_v) {
		cout << "encoded_combinatorial_object::init_with_matrix done" << endl;
	}
}

void encoded_combinatorial_object::init_row_and_col_partition(
		int *V, int nb_V, int *B, int nb_B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::init_row_and_col_partition" << endl;

	}
	int i0, j0, h, a;

	i0 = 0;
	for (h = 0; h < nb_V; h++) {
		a = V[h];
		i0 += a;
		partition[i0 - 1] = 0;
	}

	j0 = nb_rows;
	for (h = 0; h < nb_B; h++) {
		a = B[h];
		j0 += a;
		partition[j0 - 1] = 0;
	}


	if (f_v) {
		cout << "encoded_combinatorial_object::init_row_and_col_partition done" << endl;

	}
}


void encoded_combinatorial_object::init(
		int nb_rows, int nb_cols,
		int verbose_level)
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


	// ToDo:

	// action on the whole thing:
	invariant_set_start = 0;
	invariant_set_size = nb_rows + nb_cols;

	// action on rows only:
	//invariant_set_start = 0;
	//invariant_set_size = nb_rows;

	Incma = NEW_int(L);
	Int_vec_zero(Incma, L);

	partition = NEW_int(canonical_labeling_len);

	nb_flags = 0;

	Int_vec_zero(Incma, L);
	for (i = 0; i < canonical_labeling_len; i++) {
		partition[i] = 1;
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::init done" << endl;
	}
}

std::string encoded_combinatorial_object::stringify_incma()
{
	int *Flags;
	int nb_f;
	int i, j;
	string s;

	nb_f = 0;
	Flags = NEW_int(nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (get_incidence_ij(i, j)) {
				Flags[nb_f++] = i * nb_cols + j;
			}
		}
	}
	s = Int_vec_stringify(Flags, nb_f);
	FREE_int(Flags);
	return s;
}

int encoded_combinatorial_object::get_nb_flags()
{
	int nb_f;
	int i, j;

	nb_f = 0;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (get_incidence_ij(i, j)) {
				nb_f++;
			}
		}
	}
	return nb_f;
}

int *encoded_combinatorial_object::get_Incma()
{
	return Incma;
}

void encoded_combinatorial_object::set_incidence_ij(
		int i, int j)
{
	if (Incma[i * nb_cols + j] == 0) {
		Incma[i * nb_cols + j] = 1;
		nb_flags++;
	}
}

int encoded_combinatorial_object::get_incidence_ij(
		int i, int j)
{
	return Incma[i * nb_cols + j];
}

void encoded_combinatorial_object::set_incidence(
		int a)
{
	if (Incma[a] == 0) {
		Incma[a] = 1;
		nb_flags++;
	}
}

void encoded_combinatorial_object::create_Levi_graph(
		combinatorics::graph_theory::colored_graph *&CG,
		std::string &label,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::create_Levi_graph" << endl;
	}

	int i, j, p1, p2;
	int nb_points;

	nb_points = nb_rows + nb_cols;

	std::string label_tex = label;


	CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);

	CG->init_basic(
			nb_points + 2,
			label, label_tex,
			verbose_level);

	// create two special vertices:
	// the first vertex is adjacent to all points.
	// The second vertex is adjacent to all columns.

	for (i = 0; i < nb_rows; i++) {
		CG->set_adjacency(
				0, 2 + i, 1);
	}
	for (i = 0; i < nb_cols; i++) {
		CG->set_adjacency(
				1, 2 + nb_rows + i, 1);
	}


	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (get_incidence_ij(i, j) == 0) {
				continue;
			}
			p1 = i;
			p2 = nb_rows + j;
			CG->set_adjacency(
					2 + p1, 2 + p2, 1);
		}
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::create_Levi_graph done" << endl;
	}

}

void encoded_combinatorial_object::init_canonical_form(
		encoded_combinatorial_object *Enc,
		other::l1_interfaces::nauty_output *NO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::init_canonical_form" << endl;
	}

	encoded_combinatorial_object::nb_rows = Enc->nb_rows;
	encoded_combinatorial_object::nb_cols = Enc->nb_cols;

	Enc->apply_canonical_labeling(Incma, NO);

	canonical_labeling_len = nb_rows + nb_cols;
	partition = NEW_int(canonical_labeling_len);
	Int_vec_copy(Enc->partition, partition, canonical_labeling_len);

	if (f_v) {
		cout << "encoded_combinatorial_object::init_canonical_form done" << endl;
	}
}

void encoded_combinatorial_object::print_incma()
{
	cout << "encoded_combinatorial_object::print_incma" << endl;
	other::orbiter_kernel_system::Orbiter->Int_vec->matrix_print_tight(Incma, nb_rows, nb_cols);

}

void encoded_combinatorial_object::save_incma(
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::save_incma" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;
	string fname_incma;
	string fname_partition;

	fname_incma = fname_base + "_incma.csv";
	fname_partition = fname_base + "_partition.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_incma, Incma, nb_rows, nb_cols);
	if (f_v) {
		cout << "encoded_combinatorial_object::save_incma "
				"written file " << fname_incma << " of size "
				<< Fio.file_size(fname_incma) << endl;
	}

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_partition, partition, 1, canonical_labeling_len);
	if (f_v) {
		cout << "encoded_combinatorial_object::save_incma "
				"written file " << fname_partition << " of size "
				<< Fio.file_size(fname_partition) << endl;
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::save_incma done" << endl;
	}
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

void encoded_combinatorial_object::compute_canonical_incma(
		int *canonical_labeling,
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

void encoded_combinatorial_object::compute_canonical_form(
		other::data_structures::bitvector *&Canonical_form,
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


	Canonical_form = NEW_OBJECT(other::data_structures::bitvector);
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

void encoded_combinatorial_object::incidence_matrix_projective_space_top_left(
		geometry::projective_geometry::projective_space *P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::incidence_matrix_projective_space_top_left" << endl;
	}
	int i, j;

	nb_rows0 = P->Subspaces->N_points;
	nb_cols0 = P->Subspaces->N_lines;


	int f_large = false;
	long int sz;

	sz = P->Subspaces->N_points * P->Subspaces->N_lines;

	if (sz > 100 * ONE_MILLION) {
		if (f_v) {
			cout << "encoded_combinatorial_object::incidence_matrix_projective_space_top_left "
					"the matrix is large" << endl;
		}
		f_large = true;

	}

	if (f_large) {
		long int sz_100;
		long int cnt = 0;

		sz_100 = sz / 100;
		for (i = 0; i < P->Subspaces->N_points; i++) {
			for (j = 0; j < P->Subspaces->N_lines; j++, cnt++) {

				if ((cnt % sz_100) == 0) {
					cout << "encoded_combinatorial_object::incidence_matrix_projective_space_top_left "
							<< cnt / sz_100 << " percent" << endl;
				}

				if (P->Subspaces->is_incident(i, j)) {
					set_incidence_ij(i, j);
				}
			}
		}
	}
	else {
		for (i = 0; i < P->Subspaces->N_points; i++) {
			for (j = 0; j < P->Subspaces->N_lines; j++) {
				if (P->Subspaces->is_incident(i, j)) {
					set_incidence_ij(i, j);
				}
			}
		}
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::incidence_matrix_projective_space_top_left done" << endl;
	}
}


void encoded_combinatorial_object::extended_incidence_matrix_projective_space_top_left(
		geometry::projective_geometry::projective_space *P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::extended_incidence_matrix_projective_space_top_left" << endl;
	}

	int i, j, h, k, l;

	nb_rows0 = P->Subspaces->N_points + P->Subspaces->N_lines;
	nb_cols0 = P->Subspaces->N_lines + P->Subspaces->Nb_subspaces[2]; // number of lines and planes

	// point / line incidence matrix:

	for (i = 0; i < P->Subspaces->N_points; i++) {
		for (j = 0; j < P->Subspaces->N_lines; j++) {
			if (P->Subspaces->is_incident(i, j)) {
				set_incidence_ij(i, j);
			}
		}
	}

	// identity matrix in the lower part:
	for (j = 0; j < P->Subspaces->N_lines; j++) {
		set_incidence_ij(P->Subspaces->N_points + j, j);
	}


	if (f_v) {
		cout << "encoded_combinatorial_object::extended_incidence_matrix_projective_space_top_left "
				"computing points on lines" << endl;
	}

	long int *Pts_on_line;

	Pts_on_line = NEW_lint(P->Subspaces->N_lines * P->Subspaces->k);

	for (j = 0; j < P->Subspaces->N_lines; j++) {
		P->Subspaces->create_points_on_line(
				j /* line_rk */,
				Pts_on_line + j * P->Subspaces->k,
				0 /* verbose_level*/);
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::extended_incidence_matrix_projective_space_top_left "
				"computing planes through lines" << endl;
	}

	std::vector<std::vector<long int>> Plane_ranks;

	for (j = 0; j < P->Subspaces->N_lines; j++) {

		std::vector<long int> plane_ranks;

		P->Subspaces->planes_through_a_line(
				j /* line_rk */, plane_ranks,
				0 /*verbose_level*/);

		Plane_ranks.push_back(plane_ranks);

		for (h = 0; h < plane_ranks.size(); h++) {

			k = plane_ranks[h];

			set_incidence_ij(
					P->Subspaces->N_points + j,
					P->Subspaces->N_lines + k);

			for (l = 0; l < P->Subspaces->k; l++) {

				i = Pts_on_line[j * P->Subspaces->k + l];

				set_incidence_ij(i, P->Subspaces->N_lines + k);

			}
		}
	}

	FREE_lint(Pts_on_line);


	if (f_v) {
		cout << "encoded_combinatorial_object::extended_incidence_matrix_projective_space_top_left done" << endl;
	}
}



void encoded_combinatorial_object::canonical_form_given_canonical_labeling(
		int *canonical_labeling,
		other::data_structures::bitvector *&B,
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

void encoded_combinatorial_object::latex_set_system_by_rows_and_columns(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_set_system_by_rows_and_columns" << endl;
	}

	latex_set_system_by_rows(ost, verbose_level);

	latex_set_system_by_columns(ost, verbose_level);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_set_system_by_rows_and_columns done" << endl;
	}
}

void encoded_combinatorial_object::latex_set_system_by_columns(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_set_system_by_columns" << endl;
	}

	//ost << "\\subsubsection*{encoded\\_combinatorial\\_object::latex\\_set\\_system\\_by\\_columns}" << endl;

	if (nb_rows >= 30) {
		ost << "Too big, number of rows is bigger than 30\\\\" << endl;
		return;
	}

	other::l1_interfaces::latex_interface L;
	int i, j;
	int *Block;
	int sz;
	long int rk;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	Block = NEW_int(nb_rows);

	ost << "Column sets of the encoded object:\\\\" << endl;
	for (j = 0; j < nb_cols; j++) {
		sz = 0;
		for (i = 0; i < nb_rows; i++) {
			if (Incma[i * nb_cols + j]) {
				Block[sz++] = i;
			}
		}
		L.int_set_print_tex(ost, Block, sz);
		rk = Combi.rank_k_subset(Block, nb_rows, sz);
		ost << " = " << rk;
		if (j < nb_cols - 1) {
			ost << ", " << endl;
		}
		ost << "\\\\" << endl;
	}

	FREE_int(Block);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_set_system_by_columns done" << endl;
	}
}

void encoded_combinatorial_object::latex_set_system_by_rows(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_set_system_by_rows" << endl;
	}

	//ost << "\\subsubsection*{encoded\\_combinatorial\\_object::latex\\_set\\_system\\_by\\_rows}" << endl;

	if (nb_cols >= 30) {
		ost << "Too big, number of cols is bigger than 30\\\\" << endl;
		return;
	}

	if (nb_cols >= 30) {
		return;
	}

	other::l1_interfaces::latex_interface L;
	int i, j;
	int *Block;
	int sz;
	long int rk;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	Block = NEW_int(nb_cols);

	ost << "Row sets of the encoded object:\\\\" << endl;
	for (i = 0; i < nb_rows; i++) {
		sz = 0;
		for (j = 0; j < nb_cols; j++) {
			if (Incma[i * nb_cols + j]) {
				Block[sz++] = j;
			}
		}
		rk = Combi.rank_k_subset(Block, nb_cols, sz);
		L.int_set_print_tex(ost, Block, sz);
		ost << " = " << rk;
		if (i < nb_cols - 1) {
			ost << ", ";
		}
		ost << "\\\\" << endl;
	}

	FREE_int(Block);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_set_system_by_rows done" << endl;
	}
}

void encoded_combinatorial_object::latex_incma_as_01_matrix(
		std::ostream &ost,
		int verbose_level)
{
	other::l1_interfaces::latex_interface L;

	ost << "$$" << endl;
	L.print_integer_matrix_tex(
			ost,
		Incma, nb_rows, nb_cols);
	ost << "$$" << endl;

}

void encoded_combinatorial_object::latex_incma(
		std::ostream &ost,
		other::graphics::draw_incidence_structure_description
			*Draw_incidence_structure_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma" << endl;
	}

	ost << "\\subsubsection*{encoded\\_combinatorial\\_object::latex\\_incma}" << endl;

	other::l1_interfaces::latex_interface L;

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
		Int_vec_print(cout, Vi, V);
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
		Int_vec_print(cout, Bj, B);
		cout << endl;
	}

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma "
				"before L.incma_latex" << endl;
	}
	L.incma_latex(
			ost,
			Draw_incidence_structure_description,
		nb_rows /*v */,
		nb_cols /*b */,
		V, B, Vi, Bj,
		Incma,
		verbose_level - 1);
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma "
				"after L.incma_latex" << endl;
	}


	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_incma done" << endl;
	}
}





void encoded_combinatorial_object::latex_TDA_incidence_matrix(
		std::ostream &ost,
		other::graphics::draw_incidence_structure_description
			*Draw_incidence_structure_description,
		int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_incidence_matrix" << endl;
	}

	ost << "\\subsubsection*{encoded\\_combinatorial\\_object::latex\\_TDA\\_incidence\\_matrix}" << endl;


	other::l1_interfaces::latex_interface L;

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
		cout << "encoded_combinatorial_object::latex_TDA_incidence_matrix Vi=";
		Int_vec_print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_incidence_matrix Bj=";
		Int_vec_print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Inc2, nb_rows * nb_cols);

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
		cout << "encoded_combinatorial_object::latex_TDA_incidence_matrix "
				"before L.incma_latex" << endl;
	}
	L.incma_latex(
			ost,
			Draw_incidence_structure_description,
			nb_rows /*v */,
			nb_cols /*b */,
			V, B, Vi, Bj,
			Inc2,
			verbose_level - 1);
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_incidence_matrix "
				"after L.incma_latex" << endl;
	}


	FREE_int(Inc2);
	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_incidence_matrix done" << endl;
	}
}

void encoded_combinatorial_object::compute_labels(
		int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
		int *&point_labels, int *&block_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::compute_labels" << endl;
	}


	point_labels = NEW_int(nb_rows);
	block_labels = NEW_int(nb_cols);

	int i, j;

	for (i = 0; i < nb_rows; i++) {
		point_labels[i] = orbit[i];
	}
	for (j = 0; j < nb_cols; j++) {
		block_labels[j] = orbit[nb_rows + j] - nb_rows;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::compute_labels done" << endl;
	}

}


void encoded_combinatorial_object::latex_TDA_with_labels(
		std::ostream &ost,
		other::graphics::draw_incidence_structure_description *Draw_options,
		int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels" << endl;
	}

	ost << "\\subsubsection*{encoded\\_combinatorial\\_object::latex\\_TDA\\_with\\_labels}" << endl;


	other::l1_interfaces::latex_interface L;

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
		Int_vec_print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels Bj=";
		Int_vec_print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Inc2, nb_rows * nb_cols);

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
		point_labels[i] = std::to_string(orbit[i]);
	}


	for (j = 0; j < b; j++) {
		block_labels[j] = std::to_string(orbit[nb_rows + j]);
	}


	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels "
				"before L.incma_latex_with_text_labels" << endl;
	}
	L.incma_latex_with_text_labels(
			ost,
			Draw_options,
			nb_rows /*v */,
			nb_cols /*b */,
			V, B, Vi, Bj,
			Inc2,
			true, point_labels,
			true, block_labels,
			verbose_level - 1);
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels "
				"after L.incma_latex_with_text_labels" << endl;
	}


	delete [] point_labels;
	delete [] block_labels;

	FREE_int(Inc2);
	FREE_int(Vi);
	FREE_int(Bj);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_TDA_with_labels done" << endl;
	}
}


void encoded_combinatorial_object::latex_canonical_form(
		std::ostream &ost,
		other::graphics::draw_incidence_structure_description *Draw_options,
		other::l1_interfaces::nauty_output *NO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form" << endl;
	}

	ost << "\\subsubsection*{encoded\\_combinatorial\\_object::latex\\_canonical\\_form}" << endl;


	other::l1_interfaces::latex_interface L;

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
		Int_vec_print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form Bj=";
		Int_vec_print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Inc2, nb_rows * nb_cols);

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
	L.incma_latex_with_labels(
			ost,
			Draw_options,
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

void encoded_combinatorial_object::apply_canonical_labeling(
		int *&Inc2,
		other::l1_interfaces::nauty_output *NO)
{
	int i, j, i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Inc2, nb_rows * nb_cols);

	for (i = 0; i < nb_rows; i++) {
		i0 = NO->canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			j0 =  NO->canonical_labeling[nb_rows + j] - nb_rows;
			if (Incma[i0 * nb_cols + j0]) {
				Inc2[i * nb_cols + j] = 1;
			}
		}
	}

}

void encoded_combinatorial_object::apply_canonical_labeling_and_get_flags(
		int *&Inc2,
		int *&Flags, int &nb_flags_counted,
		other::l1_interfaces::nauty_output *NO)
{
	int i, j;

	apply_canonical_labeling(Inc2, NO);

	Flags = NEW_int(nb_rows * nb_cols);
	nb_flags_counted = 0;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (Inc2[i * nb_cols + j]) {
				Flags[nb_flags_counted++] = i * nb_cols + j;
			}
		}
	}
	if (nb_flags_counted != nb_flags) {
		cout << "encoded_combinatorial_object::apply_canonical_labeling_and_get_flags "
				"nb_flags_counted != nb_flags" << endl;
		exit(1);
	}
}

void encoded_combinatorial_object::latex_canonical_form_with_labels(
		std::ostream &ost,
		other::graphics::draw_incidence_structure_description *Draw_options,
		other::l1_interfaces::nauty_output *NO,
		std::string *row_labels,
		std::string *col_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form" << endl;
	}

	ost << "\\subsubsection*{encoded\\_combinatorial\\_object::latex\\_canonical\\_form\\_with\\_labels}" << endl;


	other::l1_interfaces::latex_interface L;

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
		Int_vec_print(cout, Vi, V);
		cout << endl;
	}
	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form Bj=";
		Int_vec_print(cout, Bj, B);
		cout << endl;
	}

	int *Inc2;
	int i0, j0;

	Inc2 = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Inc2, nb_rows * nb_cols);

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

	int *Flags;
	int nb_flags = 0;

	Flags = NEW_int(nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (Inc2[i * nb_cols + j]) {
				Flags[nb_flags++] = i * nb_cols + j;
			}
		}
	}
	if (nb_flags < 100) {
		Int_vec_print(ost, Flags, nb_flags);
	}
	else {
		ost << "too many to print." << endl;
	}
	FREE_int(Flags);
	ost << "\\\\" << endl;


	if (f_v) {
		cout << "encoded_combinatorial_object::latex_canonical_form "
				"before L.incma_latex_with_text_labels" << endl;
	}
	L.incma_latex_with_text_labels(
			ost,
			Draw_options,
			nb_rows /*v */,
			nb_cols /*b */,
			V, B, Vi, Bj,
			Inc2,
			true, row_labels,
			true, col_labels,
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




}}}}


