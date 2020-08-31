/*
 * sims_io.cpp
 *
 *  Created on: Aug 24, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


void sims::create_group_tree(const char *fname,
		int f_full, int verbose_level)
{
	long int i, j, h, go, l;
	longinteger_object Go;
	int *Elt;
	int *Elt2;
	int *Fst;
	file_io Fio;

	Elt = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	group_order(Go);
	go = Go.as_lint();
	//cout << "Group of order " << go << endl;

	{
	ofstream fp(fname);

	l = last_moved_base_point() + 1;
	Fst = NEW_int(l);
	for (i = 0; i < l; i++) {
		if (i == 0) {
			Fst[i] = 0;
			}
		else {
			Fst[i] = Fst[i - 1] + orbit_len[i - 1];
			}
		}
	if (f_full) {
		for (h = 0; h < go; h++) {
			element_unrank_lint(h, Elt);
			//cout << "Element " << h << " / " << go << " : path=";
			//int_vec_print(cout, S->path, S->A->base_len);
			//cout << endl;
			fp << l;
			for (i = 0; i < l; i++) {
				j = path[i];
				fp << " " << Fst[i] + j;
				coset_rep(Elt2, i, j, 0 /* verbose_level*/);
				if (i) {
					//cout << "*" << endl;
					}
				//A->element_print_quick(Elt2, cout);
				}
			fp << endl;
			//cout << "=" << endl;
			//A->element_print_quick(Elt, cout);
			//A->element_print_as_permutation(Elt, cout);
			//cout << endl;
			}
		}
	else {
		for (h = l - 1; h >= 0; h--) {
			for (j = 0; j < orbit_len[h]; j++) {
				if (h < l - 1 && j == 0) {
					continue;
					}
				int_vec_zero(path, l);
				path[h] = j;
				fp << h + 1;
				for (i = 0; i <= h; i++) {
					j = path[i];
					fp << " " << orbit[i][j];
					}
				fp << endl;
				}
			}
		}
	fp << "-1" << endl;
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	FREE_int(Elt);
	FREE_int(Elt2);
	FREE_int(Fst);
}

void sims::print_transversals()
{
	int i, j, l;
	cout << "sims data structure, a group of order ";
	print_group_order(cout);
	cout << endl;
	l = last_moved_base_point();
	for (i = 0 ; i <= l; i++) {
		for (j = 0; j < orbit_len[i]; j++) {
			cout << orbit[i][j] << " ";
			}
		cout << "(length " << orbit_len[i] << ")" << endl;
#if 0
		cout << "printing orbit_inv:" << endl;
		for (j = 0; j < A->degree; j++) {
			cout << orbit_inv[i][j] << " ";
			}
#endif
		cout << endl;
		}
}

void sims::print_transversals_short()
{
	int i, j, l;

	if (my_base_len != A->base_len()) {
		cout << "sims::print_transversals_short: "
				"my_base_len != A->base_len" << endl;
		exit(1);
		}
	l = last_moved_base_point();
	for (i = 0 ; i <= l; i++) {
		cout << "(";
		for (j = 0; j < orbit_len[i]; j++) {
			cout << orbit[i][j];
			if (j < orbit_len[i] - 1)
				cout << ",";
			}
		cout << ")";
		if (i < l)
			cout << ", ";
		}
	cout << endl;
}

void sims::print_transversal_lengths()
{
	int_vec_print(cout, orbit_len, A->base_len());
	cout << endl;
#if 0
	int i, l;

	l = last_moved_base_point();
	for (i = 0 ; i <= l; i++) {
		cout << i << " base point " << orbit[i][0]
			<< " length " << orbit_len[i] << endl;
		}
#endif
}

void sims::print_orbit_len()
{
	int i;

	for (i = 0; i < A->base_len(); i++) {
		cout << orbit_len[i] << " ";
		}
	cout << endl;
}



void sims::print(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, j;

	j = last_moved_base_point();
	cout << "sims data structure, a group of order ";
	print_group_order(cout);
	cout << endl;
	cout << "number of generators = " << nb_gen[0] << endl;
	cout << "last moved base point = " << j << endl;
	if (f_v) {
		cout << "depth : base pt : transversal "
				"length : # generators" << endl;
		for (i = 0; i <= j; i++) {
			cout << i << " : " << A->base_i(i) << " : "
					<< orbit_len[i] << " : "
					<< nb_gen[i] - nb_gen[i + 1] << endl;
			}
		cout << endl;
		}
	if (f_vv) {
		print_generator_depth_and_perm();
		}
	if (f_vvv) {
		print_generators();
		print_basic_orbits();
		}
}

void sims::print_generators()
{
	int i, j, nbg, nbg1, gen_idx;

	cout << "generators are:" << endl;
	//gens.print(cout);

	for (i = A->base_len() - 1; i >= 0; i--) {
		nbg = nb_gen[i];
		nbg1 = nb_gen[i + 1];
		cout << "level " << i << ":" << endl;
		//cout << "i=" << i << " nbg1=" << nbg1
		//<< " nbg=" << nbg << endl;
		for (j = nbg1; j < nbg; j++) {
			gen_idx = gen_perm[j];

			cout << "generator " << gen_idx << ":" << endl;
			A->element_print(gens.ith(gen_idx), cout);
			cout << endl;
			}
		cout << "orbit_len[" << i << "]=" << orbit_len[i] << endl;
		}
	cout << endl;
}

void sims::print_generators_tex(ostream &ost)
{
	int i, j, nbg, nbg1, gen_idx, cnt, f_first;

	ost << "\\begin{align*}" << endl;
	cnt = 0;
	f_first = TRUE;
	for (i = A->base_len() - 1; i >= 0; i--) {
		nbg = nb_gen[i];
		nbg1 = nb_gen[i + 1];
		//cout << "i=" << i << " nbg1=" << nbg1
		//<< " nbg=" << nbg << endl;
		for (j = nbg1; j < nbg; j++) {
			gen_idx = gen_perm[j];

			if ((cnt % 3) == 0) {
				if (!f_first) {
					ost << "\\\\" << endl;
					}
				ost << "&" << endl;
				}
			A->element_print_latex(gens.ith(gen_idx), ost);
			cnt++;
			f_first = FALSE;
			if (j < nbg - 1) {
				ost << ", \\; " << endl;
				}
			}
		ost << ";_{" << orbit_len[i] << "}" << endl;
		}
	ost << "\\end{align*}" << endl;
}


void sims::print_generators_as_permutations()
{
	int i, l;

	cout << "generators as permutations:" << endl;
	l = gens.len;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		A->element_print_as_permutation(gens.ith(i), cout);
		cout << endl;
		}
	cout << endl;
}

void sims::print_generators_as_permutations_override_action(
		action *A)
{
	int i, l;

	cout << "generators as permutations (override action):" << endl;
	l = gens.len;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		A->element_print_as_permutation(gens.ith(i), cout);
		cout << endl;
		}
	cout << endl;
}

void sims::print_basic_orbits()
{
	int i;

	if (my_base_len != A->base_len()) {
		cout << "sims::print_basic_orbits: "
				"my_base_len != A->base_len" << endl;
		exit(1);
		}
	for (i = 0 ; i < A->base_len() /* <= j */; i++) {
		print_basic_orbit(i);
		}
}

void sims::print_basic_orbit(int i)
{
	int j;

	cout << "basic orbit " << i << " of length " << orbit_len[i] << endl;
	cout << "j : orbit[i][j] : prev[i][j] : label[i][j]" << endl;
	for (j = 0; j < orbit_len[i] /* A->degree */; j++) {
		//coset_rep[i][j];
		//coset_rep_inv(i);
		if (j == orbit_len[i])
			cout << "======================================" << endl;
		cout << setw(5) << j << " : "
			<< setw(5) << orbit[i][j] << " : "
			<< setw(5) << prev[i][j] << " : "
			<< setw(5) << label[i][j];
		cout << endl;
		}
	cout << endl;
}

void sims::print_generator_depth_and_perm()
{
	int i;

	cout << "i : gen_depth[i] : gen_perm[i]" << endl;
	for (i = 0; i < gens.len; i++) {
		cout << i << " : " << gen_depth[i] << " : " << gen_perm[i] << endl;
		}
	cout << "i : base[i] : nb_gen[i]" << endl;
	for (i = 0; i <= A->base_len(); i++) {
		int pt;

		if (i < A->base_len()) {
			pt = get_orbit(i, 0);
		}
		else {
			pt = -1;
		}
		cout << i << " : " << pt << " : " << nb_gen[i] << endl;
		}
}

void sims::print_group_order(ostream &ost)
{
	longinteger_object go;
	group_order(go);
	cout << go;
}

void sims::print_group_order_factored(ostream &ost)
{
	int i, j, f_first = TRUE;

	j = last_moved_base_point();
	for (i = 0; i <= j; i++) {
		if (f_first) {
			f_first = FALSE;
			}
		else {
			cout << " * ";
			}
		cout << orbit_len[i];
		}
}

void sims::print_generators_at_level_or_below(int lvl)
{
	int i, gen_idx;

	cout << "sims::print_generators_at_level_or_below lvl=" << lvl << ":" << endl;
	for (i = 0; i < nb_gen[lvl]; i++) {
		gen_idx = gen_perm[i];
		cout << "sims::print_generators_at_level_or_below "
				"generator " << i << ":" << endl;
		A->element_print_quick(gens.ith(gen_idx), cout);
		cout << "as permutation:" << endl;
		A->element_print_as_permutation(gens.ith(gen_idx), cout);
		}
}

void sims::write_all_group_elements(char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	//char *elt;
	longinteger_object go;
	long int i;
	file_io Fio;

	Elt = NEW_int(A->elt_size_in_int);
	//elt = NEW_char(A->coded_elt_size_in_char);
	group_order(go);

	{
		ofstream fp(fname, ios::binary);

		for (i = 0; i < go.as_lint(); i++) {
			element_unrank_lint(i, Elt);
			A->element_write_file_fp(Elt, fp, 0/* verbose_level*/);
			}
	}
	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	FREE_int(Elt);
	//FREE_char(elt);
}

void sims::print_all_group_elements_to_file(char *fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	longinteger_object go;
	long int i;
	file_io Fio;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);

	{
	ofstream fp(fname);
	for (i = 0; i < go.as_lint(); i++) {
		element_unrank_lint(i, Elt);
		fp << "Element " << setw(5) << i << " / "
				<< go.as_int() << endl;
		A->element_print(Elt, fp);
		fp << endl;
		}
	}
	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	FREE_int(Elt);
}

void sims::print_all_group_elements()
{
	int *Elt;
	longinteger_object go;
	long int i;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);

	for (i = 0; i < go.as_lint(); i++) {
		element_unrank_lint(i, Elt);
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	FREE_int(Elt);
}

void sims::print_all_group_elements_tex(ostream &ost)
{
	int *Elt;
	longinteger_object go;
	long int i, ord;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);

	for (i = 0; i < go.as_lint(); i++) {
		element_unrank_lint(i, Elt);
		ord = A->element_order(Elt);
		ost << "Element " << setw(5) << i << " / "
				<< go.as_int() << " of order " << ord << ":" << endl;
		ost << "$$" << endl;
		A->element_print_latex(Elt, ost);
		ost << "$$" << endl;
		//A->element_print_as_permutation(Elt, cout);
		//cout << endl;
		}
	FREE_int(Elt);
}

void sims::print_all_group_elements_with_permutations_tex(ostream &ost)
{
	int *Elt;
	longinteger_object go;
	long int i, ord;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);

	for (i = 0; i < go.as_lint(); i++) {
		element_unrank_lint(i, Elt);
		ord = A->element_order(Elt);
		ost << "Element " << setw(5) << i << " / "
				<< go.as_int() << " of order " << ord << ":" << endl;
		ost << "$$" << endl;
		ost << "\\begin{array}{c}" << endl;
		A->element_print_latex(Elt, ost);
		ost << "\\\\" << endl;
		A->element_print_as_permutation(Elt, ost);
		ost << "\\\\" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
		//cout << endl;
		}
	FREE_int(Elt);
}



void sims::print_all_group_elements_as_permutations()
{
	int *Elt;
	longinteger_object go;
	long int i;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);

	for (i = 0; i < go.as_lint(); i++) {
		element_unrank_lint(i, Elt);
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		//A->element_print(Elt, cout);
		//cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	FREE_int(Elt);
}

void sims::print_all_group_elements_as_permutations_in_special_action(
		action *A_special)
{
	int *Elt;
	longinteger_object go;
	long int i;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);

	for (i = 0; i < go.as_lint(); i++) {
		element_unrank_lint(i, Elt);
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A_special->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	FREE_int(Elt);
}

void sims::print_all_transversal_elements()
{
	int *Elt;
	longinteger_object go;
	int i, j, ii;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);

	for (i = A->base_len() - 1; i >= 0; i--) {
		for (j = 0; j < A->transversal_length_i(i); j++) {
			if (j == 0 && i < A->base_len() - 1) {
				// skip the identity in the upper transversals
				continue;
				}
			for (ii = 0; ii < A->base_len(); ii++) {
				path[ii] = 0;
				}
			path[i] = j;
			element_from_path(Elt, 0 /* verbose_level */);
			for (ii = 0; ii < A->base_len(); ii++) {
				cout << setw(5) << path[ii] << " ";
				}
			cout << endl;
			A->element_print(Elt, cout);
			cout << endl;
			A->element_print_as_permutation(Elt, cout);
			cout << endl;
			}
		}
	FREE_int(Elt);
}

void sims::save_list_of_elements(char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	long int goi, i;
	longinteger_object go;
	file_io Fio;

	group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "sims::save_list_of_elements(): saving "
				<< goi << " elements to file " << fname << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);

	{
		ofstream fp(fname, ios::binary);

		for (i = 0; i < goi; i++) {
			element_unrank_lint(i, Elt1);
			//cout << "element " << i << ":" << endl;
			//A->element_print(Elt1, cout);
			A->element_write_file_fp(Elt1, fp, 0/* verbose_level*/);
			//A->element_print_as_permutation(Elt1, cout);
			//AA.print_as_permutation(cout, Elt1);
			//cout << endl;
			}
	}
	FREE_int(Elt1);
	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
}

void sims::read_list_of_elements(action *A, char *fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1, *Elt2;
	int goi, i;
	file_io Fio;

	goi = Fio.file_size(fname) / A->coded_elt_size_in_char;
	if (f_v) {
		cout << "sims::read_list_of_elements(): reading "
				<< goi << " elements from file " << fname << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	init(A, verbose_level - 2);
	init_trivial_group(verbose_level - 1);

#if 0
	FILE *f2;
	f2 = fopen(fname, "rb");

	for (i = 0; i < goi; i++) {
		A->element_read_file_fp(Elt1, f2, 0/* verbose_level*/);
		//cout << "element " << i << ":" << endl;
		//A->element_print(Elt1, cout);
		strip_and_add(Elt1, Elt2, verbose_level - 1);
		}
	fclose(f2);
#else
	{
		ifstream fp(fname, ios::binary);

		for (i = 0; i < goi; i++) {
			A->element_read_file_fp(Elt1, fp, 0/* verbose_level*/);
			//cout << "element " << i << ":" << endl;
			//A->element_print(Elt1, cout);
			strip_and_add(Elt1, Elt2, verbose_level - 1);
			}
	}
#endif
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "read file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
}

#if 0
void sims::write_sgs(const char *fname, int verbose_level)
// used by compute_stabilizer::update_stabilizer
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	char *elt;
	longinteger_object go;
	int i;
	vector_ge SG;
	int *tl;
	file_io Fio;

	if (f_v) {
		cout << "sims::write_sgs fname=" << fname << endl;
		}
	Elt = NEW_int(A->elt_size_in_int);
	elt = NEW_char(A->coded_elt_size_in_char);
	group_order(go);

	tl = NEW_int(A->base_len());
	extract_strong_generators_in_order(SG, tl, 0 /*verbose_level*/);


	{
	ofstream fp(fname);
	fp << "# group of order " << go << endl;
	fp << "# action: " << A->label << endl;
	fp << "# action: " << A->label_tex << endl;
	fp << "# base length: " << endl;
	fp << A->base_len() << endl;
	fp << "# base: " << endl;
	for (i = 0; i < A->base_len(); i++) {
		fp << setw(5) << A->base_i(i) << ", ";
		}
	fp << endl;
	fp << "# transversal lengths: " << endl;
	for (i = 0; i < A->base_len(); i++) {
		fp << setw(5) << tl[i] << ", ";
		}
	fp << endl;
	fp << "# number of generators: " << endl;
	fp << SG.len << endl;
	fp << "# make_element_size: " << endl;
	fp << A->make_element_size << endl;
	fp << "# generators are:" << endl;
	for (i = 0; i < SG.len; i++) {
		A->element_print_for_make_element(SG.ith(i), fp);
		fp << endl;
		}
	}
	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	FREE_int(Elt);
	FREE_char(elt);
}



#define READ_SGS_BUF_SIZE 50000


void sims::read_sgs(const char *fname,
		vector_ge *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt;
	char *elt;
	longinteger_object go;
	int i, j;
	char label[1000];
	char *buf;
	int base_length;
	int *base1; // [base_length]
	int *tl; //[base_length]
	int nb_gens;
	int make_element_size1;
	int *data; // [nb_gens * make_element_size1]
	//char str[1000];
	file_io Fio;

	if (f_v) {
		cout << "sims::read_sgs fname=" << fname << endl;
		}

	buf = NEW_char(READ_SGS_BUF_SIZE);
	Elt = NEW_int(A->elt_size_in_int);
	elt = NEW_char(A->coded_elt_size_in_char);
	//group_order(go);

	if (f_v) {
		cout << "reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	{
		ifstream fp(fname);
		//string str;
		char *p_buf;

		// reading a comment line: "group of order ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading action
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		strcpy(label, buf + 10);
		cout << "action: " << label << endl;
		if (strcmp(label, A->label)) {
			cout << "actions do not match" << endl;
			cout << "action on file: " << label << endl;
			cout << "current action: " << A->label << endl;
			exit(1);
			}
		// reading action
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading comment "base_len ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading base_len
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		base_length = atoi(buf);
		if (f_vv) {
			cout << "base_length=" << base_length << endl;
			}

		base1 = NEW_int(base_length);
		// reading comment
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading base
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		p_buf = buf;
		for (i = 0; i < base_length; i++) {
			if (!s_scan_int(&p_buf, &base1[i])) {
				cout << "error reading base" << endl;
				exit(1);
				}
			}
		if (f_vv) {
			cout << "read base: ";
			int_vec_print(cout, base1, base_length);
			cout << endl;
			}

		tl = NEW_int(base_length);

		// reading comment "transversal lengths ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading transversal lengths
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		p_buf = buf;
		for (i = 0; i < base_length; i++) {
			if (!s_scan_int(&p_buf, &tl[i])) {
				cout << "error reading tl[" << i << "]" << endl;
				exit(1);
				}
			if (f_vv) {
				cout << "read tl[" << i << "]=" << tl[i] << endl;
				//cout << "remaining string: '" << p_buf << "'" << endl;
				}
			}
		if (f_vv) {
			cout << "read tl: ";
			int_vec_print(cout, tl, base_length);
			cout << endl;
			}

		// reading comment line "number of generators ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading number of generators
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		nb_gens = atoi(buf);
		if (f_vv) {
			cout << "nb_gens=" << nb_gens << endl;
			}
		// reading comment "make element size ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading make_element_size1
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		make_element_size1 = atoi(buf);
		if (f_vv) {
			cout << "make_element_size=" << make_element_size1 << endl;
			}
		if (make_element_size1 != A->make_element_size) {
			cout << "make element size does not match" << endl;
			exit(1);
			}

		// reading comment "generators are ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');

		data = NEW_int(nb_gens * make_element_size1);
		for (i = 0; i < nb_gens; i++) {
			// reading generator i:
			if (f_vv) {
				cout << "reading generator " << i << ":" << endl;
				}
			fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
			p_buf = buf;
			for (j = 0; j < make_element_size1; j++) {
				if (!s_scan_int(&p_buf, &data[i * make_element_size1 + j])) {
					cout << "error reading generator "
							<< i << " integer " << j << endl;
					exit(1);
					}
				}
			}
	}

	if (f_v) {
		cout << "parsing generators:" << endl;
		}
	SG->init(A, verbose_level - 2);
	SG->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->make_element(Elt, data + i * make_element_size1, 0);
		A->element_move(Elt, SG->ith(i), 0);
		if (f_vv) {
			cout << "generator " << i << ":" << endl;
			A->element_print_quick(SG->ith(i), cout);
			cout << endl;
			}
		}

	if (A->f_has_base()) {
		if (base_length != A->base_len()) {
			cout << "base_len does not match" << endl;
			cout << "A->base_len=" << A->base_len() << endl;
			cout << "read " << base_length << endl;
			exit(1);
			}
		for (i = 0; i < base_length; i++) {
			if (base1[i] != A->base_i(i)) {
				cout << "base does not match" << endl;
				cout << "A->base[" << i << "]=" << A->base_i(i) << endl;
				cout << "base1[" << i << "]=" << base1[i] << endl;
				exit(1);
				}
			}
		}
	else {
		if (f_v) {
			cout << "action does not have a base, "
					"we will initialize with the base from file" << endl;
			}
		A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
		A->Stabilizer_chain->allocate_base_data(A, base_length, verbose_level);
		//A->allocate_base_data(base_length);
		//A->Stabilizer_chain->base_len = base_length;
		for (i = 0; i < base_length; i++) {
			A->Stabilizer_chain->base_i(i) = base1[i];
			}
		if (f_vv) {
			cout << "the base is: ";
			int_vec_print(cout, A->get_base(), A->base_len());
			cout << endl;

			cout << "rallocating base:" << endl;
			}
		reallocate_base(0, verbose_level - 1);
		}

	if (f_v) {
		cout << "before init_trivial_group" << endl;
		}
	init_trivial_group(verbose_level - 2);
	if (f_v) {
		cout << "before init_generators" << endl;
		}
	init_generators(*SG, verbose_level - 2);
	if (f_v) {
		cout << "before compute_base_orbits_known_length" << endl;
		}
	compute_base_orbits_known_length(tl, verbose_level - 2);
	group_order(go);
	if (f_v) {
		cout << "sims::read_sgs created a group of order "
				<< go << endl;
		}

	FREE_int(data);
	FREE_int(tl);
	FREE_int(base1);
	FREE_int(Elt);
	FREE_char(elt);
	FREE_char(buf);
}
#endif

void sims::write_as_magma_permutation_group(const char *fname_base,
		vector_ge *gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, k, n, l, h;
	longinteger_object go;
	int *Elt1;
	int *Elt2;
	int *Table;
	combinatorics_domain Combi;
	file_io Fio;

	if (f_v) {
		cout << "sims::write_as_magma_permutation_group" << endl;
		}
	group_order(go);
	n = go.as_int();
	l = gens->len;
	if (f_v) {
		cout << "sims::write_as_magma_permutation_group "
				"Computing the Table, go=" << go << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Table = NEW_int(l * n);
	int_vec_zero(Table, l * n);
	for (h = 0; h < l; h++) {
		if (f_v) {
			cout << "sims::write_as_magma_permutation_group "
					"h = " << h <<  " / " << l << endl;
			}
		for (i = 0; i < n; i++) {
			if (f_v) {
				cout << "sims::write_as_magma_permutation_group "
						"i = " << i <<  " / " << n << endl;
				}
			element_unrank_lint(i, Elt1);

			A->element_mult(Elt1, gens->ith(h), Elt2, 0);

			if (f_v) {
				cout << "Elt2=" << endl;
				A->element_print(Elt2, cout);
				}
			k = element_rank_lint(Elt2);
			if (f_v) {
				cout << "has rank k=" << k << endl;
				}
			Table[h * n + i] = k;
			}
		}
#if 0
> G := PermutationGroup< 12 | (1,6,7)(2,5,8,3,4,9)(11,12),
>                             (1,3)(4,9,12)(5,8,10,6,7,11) >;
#endif
	char fname[1000];

	sprintf(fname, "%s.magma", fname_base);
	{
	ofstream fp(fname);

	fp << "G := PermutationGroup< " << n << " | " << endl;
	for (i = 0; i < l; i++) {
		Combi.perm_print_counting_from_one(fp, Table + i * n, n);
		if (i < l - 1) {
			fp << ", " << endl;
			}
		}
	fp << " >;" << endl;
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Table);

	if (f_v) {
		cout << "sims::write_as_magma_permutation_group done" << endl;
		}
}

void sims::report(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::report" << endl;
	}
	int i;
	sorting Sorting;

	ost << endl << "\\subsection*{Stabilizer chain}" << endl << endl;
	ost << endl;


	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Level} & \\mbox{Base pt} & \\mbox{Orbit length} & \\mbox{Subgroup order}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < my_base_len; i++) {



		longinteger_object go;

		subgroup_order_verbose(go, i, FALSE /*verbose_level*/);
		ost << i << " & " << get_orbit(i, 0) << " & " << get_orbit_length(i) << " & ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	string fname_base;
	char str[1000];


	for (i = 0; i < my_base_len; i++) {

		if (f_v) {
			cout << "sims::report tree " << i << " / " << my_base_len << endl;
		}
		ost << endl << "\\subsection*{Basic Orbit " << i << "}" << endl << endl;

		if (orbit_len[i] < 1000) {

			sprintf(str, "sims_%d", i);
			fname_base.assign(str);

			layered_graph *LG;
			Sorting.schreier_vector_tree(
				orbit_len[i], orbit[i], prev[i], TRUE /* f_use_pts_inv */, orbit_inv[i],
				fname_base,
				LG,
				FALSE/* f_embedded */, FALSE /* f_sideways */,
				verbose_level - 3);

			FREE_OBJECT(LG);

			ost << "\\input " << fname_base << ".tex" << endl;
			ost << endl;
		}
		ost << "\\bigskip" << endl;
		if (f_v) {
			cout << "sims::report tree " << i << " / " << my_base_len << " done" << endl;
		}

	}

	if (f_v) {
		cout << "sims::report done" << endl;
	}
}




}}
