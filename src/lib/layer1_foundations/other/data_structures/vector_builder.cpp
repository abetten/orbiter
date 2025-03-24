/*
 * vector_builder.cpp
 *
 *  Created on: Nov 1, 2021
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


vector_builder::vector_builder()
{
	Record_birth();
	Descr = NULL;
	F = NULL;
	v = NULL;
	f_has_k = false;
	k = 0;
	len = 0;
}

vector_builder::~vector_builder()
{
	Record_death();
	if (v) {
		FREE_lint(v);
		v = NULL;
	}
}

void vector_builder::init(
		vector_builder_description *Descr,
		algebra::field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_builder::init" << endl;
	}

	vector_builder::Descr = Descr;
	vector_builder::F = F;

	if (Descr->f_dense) {
		if (f_v) {
			cout << "vector_builder::init -dense" << endl;
		}
		Lint_vec_scan(Descr->dense_text, v, len);

		if (Descr->f_format) {
			f_has_k = true;
			k = Descr->format_k;
		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}
	else if (Descr->f_compact) {
		if (f_v) {
			cout << "vector_builder::init -compact" << endl;
		}
		int i, j;
		char c;

		len = 0;
		for (i = 0; i < Descr->compact_text.length(); i++) {
			c = Descr->compact_text[i];
			if (c == ' ' || c == ',' || c == '\n') {
				continue;
			}
			len++;
		}
		v = NEW_lint(len);
		j = 0;
		for (i = 0; i < Descr->compact_text.length(); i++) {
			c = Descr->compact_text[i];
			if (c == ' ' || c == ',' || c == '\n') {
				continue;
			}
			v[j++] = c - '0';
		}

		if (Descr->f_format) {
			f_has_k = true;
			k = Descr->format_k;
		}

		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}
	else if (Descr->f_repeat) {
		if (f_v) {
			cout << "vector_builder::init -repeat" << endl;
		}
		int *w;
		int sz;
		int i;

		Int_vec_scan(Descr->repeat_text, w, sz);
		if (f_v) {
			cout << "vector_builder::init repeat pattern: ";
			Int_vec_print(cout, w, sz);
			cout << endl;
		}

		v = NEW_lint(Descr->repeat_length);
		len = Descr->repeat_length;

		for (i = 0; i < Descr->repeat_length; i++) {
			v[i] = w[i % sz];
		}

		if (Descr->f_format) {
			f_has_k = true;
			k = Descr->format_k;
		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}
	else if (Descr->f_file) {
		if (f_v) {
			cout << "vector_builder::init "
					"-file " << Descr->file_name << endl;
		}
		orbiter_kernel_system::file_io Fio;
		int m, n;

		Fio.Csv_file_support->lint_matrix_read_csv(
				Descr->file_name, v, m, n, verbose_level);
		len = m * n;
		f_has_k = true;
		k = m;
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}

	else if (Descr->f_file_column) {
		if (f_v) {
			cout << "vector_builder::init "
					"-f_file_column " << Descr->file_column_name
					<< " " << Descr->file_column_label << endl;
		}
		orbiter_kernel_system::file_io Fio;
		int m, n;
		data_structures::set_of_sets *SoS;


		//Fio.Csv_file_support->lint_matrix_read_csv(
		//		Descr->file_name, v, m, n, verbose_level);

		if (f_v) {
			cout << "vector_builder::init "
					"reading file " << Descr->file_column_name
					<< ", column " << Descr->file_column_label << endl;
		}

		Fio.Csv_file_support->read_column_as_set_of_sets(
				Descr->file_column_name, Descr->file_column_label,
					SoS,
					verbose_level);

		if (SoS->nb_sets == 0) {
			cout << "vector_builder::init the file seems to be empty" << endl;
			exit(1);
		}

		m = SoS->nb_sets;
		n = SoS->Set_size[0];
		v = NEW_lint(m * n);

		int i;

		for (i = 0; i < m; i++) {
			Lint_vec_copy(SoS->Sets[i], v + i * n, n);
		}

		FREE_OBJECT(SoS);

		len = m * n;
		f_has_k = true;
		k = m;
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}

	else if (Descr->f_load_csv_no_border) {
		if (f_v) {
			cout << "vector_builder::init "
					"-load_csv_no_border "
					<< Descr->load_csv_no_border_fname << endl;
		}
		orbiter_kernel_system::file_io Fio;
		int m, n;

		Fio.Csv_file_support->lint_matrix_read_csv_no_border(
				Descr->load_csv_no_border_fname,
				v, m, n, verbose_level);
		len = m * n;
		f_has_k = true;
		k = m;

		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}
	}
	else if (Descr->f_load_csv_data_column) {
		if (f_v) {
			cout << "vector_builder::init "
					"-load_csv_data_column "
					<< Descr->load_csv_data_column_fname << endl;
		}
		orbiter_kernel_system::file_io Fio;
		int m, n;

		Fio.Csv_file_support->lint_matrix_read_csv_data_column(
				Descr->load_csv_data_column_fname,
				v, m, n,
				Descr->load_csv_data_column_idx,
				verbose_level);
		len = m * n;
		f_has_k = true;
		k = m;

		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}
		if (f_v) {
			cout << "vector_builder::init matrix of size " << m << " x " << n << endl;
			Lint_matrix_print(v, m, n);
		}
	}
	else if (Descr->f_sparse) {
		if (f_v) {
			cout << "vector_builder::init -sparse" << endl;
		}
		int *pairs;
		int sz;
		int nb_pairs;
		int i, idx;
		int c;

		Int_vec_scan(Descr->sparse_pairs, pairs, sz);

		len = Descr->sparse_len;
		v = NEW_lint(len);
		Lint_vec_zero(v, len);
		nb_pairs = sz >> 1;
		for (i = 0; i < nb_pairs; i++) {
			c = pairs[2 * i + 0];
			idx = pairs[2 * i + 1];
			if (idx < 0 || idx >= len) {
				cout << "vector_builder::init idx is out of range" << endl;
				exit(1);
			}
			v[idx] = c;
		}
		FREE_int(pairs);

		if (Descr->f_format) {
			f_has_k = true;
			k = Descr->format_k;
		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}
	else if (Descr->f_concatenate) {

		int i, j;
		string_tools ST;


		std::vector<std::string> values;

		ST.parse_comma_separated_list(
				Descr->concatenate_list, values,
				0 /*verbose_level*/);

		len = 0;

		for (i = 0; i < values.size(); i++) {
			vector_builder *VB;

			VB = Get_vector(values[i]);
			len += VB->len;
		}
		v = NEW_lint(len);
		j = 0;
		for (i = 0; i < values.size(); i++) {
			vector_builder *VB;

			VB = Get_vector(values[i]);
			Lint_vec_copy(VB->v, v + j, VB->len);
			j += VB->len;
		}
		if (Descr->f_format) {
			f_has_k = true;
			k = Descr->format_k;
		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}
	else if (Descr->f_loop) {
		if (f_v) {
			cout << "vector_builder::init "
					"using index set through loop, start="
					<< Descr->loop_start
					<< " upper_bound=" <<  Descr->loop_upper_bound
					<< " increment=" << Descr->loop_increment << endl;
		}
		int i, cnt;

		len = 0;
		for (i = Descr->loop_start; i < Descr->loop_upper_bound;
				i += Descr->loop_increment) {
			len++;
		}
		v = NEW_lint(len);
		cnt = 0;
		for (i = Descr->loop_start; i < Descr->loop_upper_bound;
				i += Descr->loop_increment) {

			v[cnt++] = i;

		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}
	}
	else if (Descr->f_index_of_support) {
		if (f_v) {
			cout << "vector_builder::init "
					"-index_of_support" << endl;
		}

		vector_builder *VB;

		VB = Get_vector(Descr->index_of_support_input);

		int i, j, s, c;

		len = 0;
		for (i = 0; i < VB->len; i++) {
			if (VB->v[i]) {
				len++;
			}
		}

		v = NEW_lint(len);

		if (VB->f_has_k) {
			f_has_k = true;
			k = VB->k;
			s = VB->len / k;
			if (f_v) {
				cout << "vector_builder::init "
						"reading matrix with k=" << k << " rows and " << s << " columns" << endl;
			}
			if (k * s != VB->len) {
				cout << "vector_builder::init k * s != VB->len" << endl;
				exit(1);
			}
			c = 0;
			for (i = 0; i < k; i++) {
				for (j = 0; j < s; j++) {
					if (VB->v[i * s + j]) {
						v[c++] = j;
					}
				}
			}
		}
		else {
			j = 0;
			for (i = 0; i < VB->len; i++) {
				if (VB->v[i]) {
					v[j++] = i;
				}
			}

		}

	}

	else if (Descr->f_permutation_matrix) {
		if (f_v) {
			cout << "vector_builder::init "
					"-permutation_matrix" << endl;
		}

		int *perm;
		int sz;

		Int_vec_scan(Descr->permutation_matrix_data, perm, sz);

		f_has_k = true;
		k = sz;
		len = sz * sz;

		v = NEW_lint(len);
		Lint_vec_zero(v, len);

		int i;

		for (i = 0; i < sz; i++) {
			v[i * sz + perm[i]] = 1;
		}


		FREE_int(perm);

	}

	else if (Descr->f_permutation_matrix_inverse) {
		if (f_v) {
			cout << "vector_builder::init "
					"-permutation_matrix_inverse" << endl;
		}

		int *perm;
		int *perm_inv;
		int sz;

		Int_vec_scan(Descr->permutation_matrix_inverse_data, perm, sz);

		perm_inv = NEW_int(sz);

		combinatorics::other_combinatorics::combinatorics_domain Combi;

		Combi.Permutations->perm_inverse(perm, perm_inv, sz);

		f_has_k = true;
		k = sz;
		len = sz * sz;

		v = NEW_lint(len);
		Lint_vec_zero(v, len);

		int i;

		for (i = 0; i < sz; i++) {
			v[i * sz + perm_inv[i]] = 1;
		}


		FREE_int(perm);
		FREE_int(perm_inv);

	}
	else if (Descr->f_binary_data_lint) {
		if (f_v) {
			cout << "vector_builder::init f_binary_data_lint" << endl;
		}
		len = Descr->binary_data_lint_sz;
		v = NEW_lint(len);
		Lint_vec_copy(Descr->binary_data_lint, v, len);

		if (Descr->f_format) {
			f_has_k = true;
			k = Descr->format_k;
		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}
	else if (Descr->f_binary_data_int) {
		if (f_v) {
			cout << "vector_builder::init f_binary_data_int" << endl;
		}
		len = Descr->binary_data_int_sz;
		v = NEW_lint(len);
		Int_vec_copy_to_lint(Descr->binary_data_int, v, len);

		if (Descr->f_format) {
			f_has_k = true;
			k = Descr->format_k;
		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}

	}

	else {
		cout << "vector_builder::init please specify "
				"how the vector should be created" << endl;
		exit(1);
	}

	if (Descr->f_field) {
		int i, a;

		for (i = 0; i < len; i++) {
			a = v[i];
			if (a < 0) {
				if (Descr->f_allow_negatives) {
					algebra::number_theory::number_theory_domain NT;
					int a0;

					a0 = a;
					v[i] = NT.mod(a, F->p);
					cout << "vector_builder::init "
							"entry mapped from = " << a0 << " to " << v[i] << endl;
				}
				else {
					cout << "vector_builder::init "
							"entry is out of range: value = " << a << endl;
					exit(1);
				}
			}
			if (a >= F->q) {
				cout << "vector_builder::init "
						"entry is out of range: value = " << a << endl;
				exit(1);
			}
		}
		if (f_v) {
			cout << "vector_builder::init found a vector of length " << len << endl;
		}
	}

	if (f_v) {
		cout << "vector_builder::init "
				"created vector of length " << len << endl;
		Lint_vec_print(cout, v, len);
		cout << endl;
		//Lint_vec_print_fully(cout, v, len);
		//cout << endl;
		if (f_has_k) {
			cout << "also seen as matrix of size "
					<< k << " x " << len / k << endl;
			if (k > 20 || (len / k) > 20) {
				cout << "too large to print" << endl;
			}
			else {
				Lint_matrix_print(v, k, len / k);
				cout << endl;
			}
		}
	}


	if (f_v) {
		cout << "vector_builder::init done" << endl;
	}
}


void vector_builder::print(
		std::ostream &ost)
{

	Lint_vec_print(ost, v, len);
	ost << endl;
}


}}}}




