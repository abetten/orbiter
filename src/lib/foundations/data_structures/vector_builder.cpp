/*
 * vector_builder.cpp
 *
 *  Created on: Nov 1, 2021
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {
namespace data_structures {


vector_builder::vector_builder()
{
	Descr = NULL;
	F = NULL;
	v = NULL;
	f_has_k = FALSE;
	k = 0;
	len = 0;
}

vector_builder::~vector_builder()
{
	if (v) {
		FREE_int(v);
		v = NULL;
	}
}

void vector_builder::init(vector_builder_description *Descr,
		finite_field *F,
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
		Orbiter->Int_vec->scan(Descr->dense_text, v, len);

		if (Descr->f_format) {
			f_has_k = TRUE;
			k = Descr->format_k;
		}

	}
	else if (Descr->f_compact) {
		if (f_v) {
			cout << "vector_builder::init -compact" << endl;
		}
		int i, j;
		char c;

		//len = Descr->compact_text.length();
		len = 0;
		for (i = 0; i < Descr->compact_text.length(); i++) {
			c = Descr->compact_text[i];
			if (c == ' ' || c == ',' || c == '\n') {
				continue;
			}
			len++;
		}
		v = NEW_int(len);
		j = 0;
		for (i = 0; i < Descr->compact_text.length(); i++) {
			c = Descr->compact_text[i];
			if (c == ' ' || c == ',' || c == '\n') {
				continue;
			}
			v[j++] = c - '0';
		}

		if (Descr->f_format) {
			f_has_k = TRUE;
			k = Descr->format_k;
		}

	}
	else if (Descr->f_repeat) {
		if (f_v) {
			cout << "vector_builder::init -repeat" << endl;
		}
		int *w;
		int sz;
		int i;

		Orbiter->Int_vec->scan(Descr->repeat_text, w, sz);
		if (f_v) {
			cout << "vector_builder::init repeat pattern: ";
			Orbiter->Int_vec->print(cout, w, sz);
			cout << endl;
		}

		v = NEW_int(Descr->repeat_length);
		len = Descr->repeat_length;

		for (i = 0; i < Descr->repeat_length; i++) {
			v[i] = w[i % sz];
		}

		if (Descr->f_format) {
			f_has_k = TRUE;
			k = Descr->format_k;
		}

	}
	else if (Descr->f_file) {
		if (f_v) {
			cout << "vector_builder::init -file " << Descr->file_name << endl;
		}
		file_io Fio;
		int m, n;

		Fio.int_matrix_read_csv(Descr->file_name, v, m, n, verbose_level);
		len = m * n;
		f_has_k = TRUE;
		k = m;

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

		Orbiter->Int_vec->scan(Descr->sparse_pairs, pairs, sz);

		len = Descr->sparse_len;
		v = NEW_int(len);
		Orbiter->Int_vec->zero(v, len);
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
			f_has_k = TRUE;
			k = Descr->format_k;
		}

	}
	else if (Descr->concatenate_list.size()) {

		int i, j;

		len = 0;

		for (i = 0; i < Descr->concatenate_list.size(); i++) {
			vector_builder *VB;

			VB = Orbiter->get_object_of_type_vector(Descr->concatenate_list[i]);
			len += VB->len;
		}
		v = NEW_int(len);
		j = 0;
		for (i = 0; i < Descr->concatenate_list.size(); i++) {
			vector_builder *VB;

			VB = Orbiter->get_object_of_type_vector(Descr->concatenate_list[i]);
			Orbiter->Int_vec->copy(VB->v, v + j, VB->len);
			j += VB->len;
		}

	}
	else if (Descr->f_loop) {
		if (f_v) {
			cout << "vector_builder::init using index set through loop, start="
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
		v = NEW_int(len);
		cnt = 0;
		for (i = Descr->loop_start; i < Descr->loop_upper_bound;
				i += Descr->loop_increment) {


			v[cnt++] = i;

		}
	}
	else {
		cout << "vector_builder::init please specify how the vector should be created" << endl;
		exit(1);
	}

	if (Descr->f_field) {
		int i, a;

		for (i = 0; i < len; i++) {
			a = v[i];
			if (a < 0) {
				cout << "vector_builder::init entry is out of range: value = " << a << endl;
				exit(1);
			}
			if (a >= F->q) {
				cout << "vector_builder::init entry is out of range: value = " << a << endl;
				exit(1);
			}
		}
	}

	if (f_v) {
		cout << "vector_builder::init created vector of length " << len << endl;
		Orbiter->Int_vec->print(cout, v, len);
		cout << endl;
		if (f_has_k) {
			cout << "also seen as matrix of size  " << k << " x " << len / k << endl;
			Orbiter->Int_vec->matrix_print(v, k, len / k);
			cout << endl;

		}
	}


	if (f_v) {
		cout << "vector_builder::init done" << endl;
	}
}




}}}



