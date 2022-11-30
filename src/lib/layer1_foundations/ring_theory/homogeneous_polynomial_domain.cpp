// homogeneous_polynomial_domain.cpp
//
// Anton Betten
//
// September 9, 2016



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {


static int homogeneous_polynomial_domain_compare_monomial_with(void *data,
	int i, void *data2, void *extra_data);
static int homogeneous_polynomial_domain_compare_monomial(void *data,
	int i, int j, void *extra_data);
static void homogeneous_polynomial_domain_swap_monomial(void *data,
	int i, int j, void *extra_data);



homogeneous_polynomial_domain::homogeneous_polynomial_domain()
{
	Monomial_ordering_type = t_LEX;
	F = NULL;
	nb_monomials = 0;
	Monomials = NULL;
	//symbols;
	//symbols_latex;
	//monomial_symbols;
	//monomial_symbols_latex;
	//monomial_symbols_easy;
	Variables = NULL;
	nb_affine = 0;
	Affine = NULL;
	v = NULL;
	Affine_to_monomial = NULL;
	coeff2 = NULL;
	coeff3 = NULL;
	coeff4 = NULL;
	factors = NULL;
	my_affine = NULL;
	base_cols = NULL;
	type1 = NULL;
	type2 = NULL;


	q = 0;
	nb_variables = 0;
	degree = 0;


}

homogeneous_polynomial_domain::~homogeneous_polynomial_domain()
{
	if (v) {
		FREE_int(v);
	}
	if (Monomials) {
		FREE_int(Monomials);
	}
	if (Variables) {
		FREE_int(Variables);
	}
	if (Affine) {
		FREE_int(Affine);
	}
	if (Affine_to_monomial) {
		FREE_int(Affine_to_monomial);
	}
	if (coeff2) {
		FREE_int(coeff2);
	}
	if (coeff3) {
		FREE_int(coeff3);
	}
	if (coeff4) {
		FREE_int(coeff4);
	}
	if (factors) {
		FREE_int(factors);
	}
	if (my_affine) {
		FREE_int(my_affine);
	}
	if (base_cols) {
		FREE_int(base_cols);
	}
	if (type1) {
		FREE_int(type1);
	}
	if (type2) {
		FREE_int(type2);
	}
}

void homogeneous_polynomial_domain::init(polynomial_ring_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init" << endl;
	}

	field_theory::finite_field *F;

	if (!Descr->f_field) {
		cout << "Please specify whether the polynomial ring is over a field" << endl;
		exit(1);
	}

	F = orbiter_kernel_system::Orbiter->get_object_of_type_finite_field(Descr->finite_field_label);

	if (!Descr->f_number_of_variables) {
		cout << "Please specify the number of variables of the polynomial ring using -number_of_variables <n>" << endl;
		exit(1);
	}

	if (!Descr->f_homogeneous) {
		cout << "Please specify the degree of the homogeneous polynomial ring using -homogeneous <d>" << endl;
		exit(1);
	}



	if (Descr->f_variables) {
		data_structures::string_tools ST;
		std::vector<std::string> managed_variables_txt;
		std::vector<std::string> managed_variables_tex;

		ST.parse_comma_separated_strings(Descr->variables_txt, managed_variables_txt);
		ST.parse_comma_separated_strings(Descr->variables_tex, managed_variables_tex);

		if (managed_variables_txt.size() != managed_variables_tex.size()) {
			cout << "number of variables in txt and in tex differ" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init before init with variables" << endl;
		}
		init_with_or_without_variables(F,
				Descr->number_of_variables,
				Descr->homogeneous_of_degree,
				Descr->Monomial_ordering_type,
				TRUE,
				&managed_variables_txt,
				&managed_variables_tex,
				verbose_level);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init after init with variables" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init before init w/o variables" << endl;
		}
		init_with_or_without_variables(F,
				Descr->number_of_variables,
				Descr->homogeneous_of_degree,
				Descr->Monomial_ordering_type,
				FALSE,
				NULL,
				NULL,
				verbose_level);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init after init w/o variables" << endl;
		}

	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init done" << endl;
	}
}

void homogeneous_polynomial_domain::init(field_theory::finite_field *F,
		int nb_vars, int degree,
		monomial_ordering_type Monomial_ordering_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init before init_with_or_without_variables" << endl;
	}
	init_with_or_without_variables(F, nb_vars, degree,
			Monomial_ordering_type,
			FALSE, NULL, NULL,
			verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init after init_with_or_without_variables" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init done" << endl;
	}
}

void homogeneous_polynomial_domain::init_with_or_without_variables(field_theory::finite_field *F,
		int nb_vars, int degree,
		monomial_ordering_type Monomial_ordering_type,
		int f_has_variables,
		std::vector<std::string> *variables_txt,
		std::vector<std::string> *variables_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init_with_or_without_variables" << endl;
	}
	homogeneous_polynomial_domain::F = F;
	q = F->q;
	homogeneous_polynomial_domain::nb_variables = nb_vars;
	homogeneous_polynomial_domain::degree = degree;
	homogeneous_polynomial_domain::Monomial_ordering_type = Monomial_ordering_type;
	
	v = NEW_int(nb_variables);
	type1 = NEW_int(degree + 1);
	type2 = NEW_int(degree + 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init_with_or_without_variables before make_monomials" << endl;
	}
	make_monomials(Monomial_ordering_type,
			f_has_variables, variables_txt, variables_tex,
			verbose_level - 2);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init_with_or_without_variables after make_monomials" << endl;
	}
	
	m = MAXIMUM(nb_monomials, degree + 1);
		// substitute_semilinear needs [nb_monomials]
		// substitute_line needs [degree + 1]

	coeff2 = NEW_int(m);
	coeff3 = NEW_int(m);
	coeff4 = NEW_int(m);
	factors = NEW_int(degree);

	my_affine = NEW_int(degree);
	base_cols = NEW_int(nb_monomials);
	
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init_with_or_without_variables done" << endl;
	}
	
}

void homogeneous_polynomial_domain::print()
{
	cout << "Polynomial ring over a field of order " << F->q
			<< " in " << nb_variables << " variables "
			"and of degree " << degree << endl;
}


void homogeneous_polynomial_domain::print_latex(std::ostream &ost)
{
	ost << "Polynomial ring over a field of order " << F->q
			<< " in " << nb_variables << " variables "
			"and of degree " << degree << "\\\\" << endl;
}


int homogeneous_polynomial_domain::get_nb_monomials()
{
	return nb_monomials;
}

int homogeneous_polynomial_domain::get_nb_variables()
{
	return nb_variables;
}

field_theory::finite_field *homogeneous_polynomial_domain::get_F()
{
	return F;
}

int homogeneous_polynomial_domain::get_monomial(int i, int j)
{
	if (j > nb_variables) {
		cout << "homogeneous_polynomial_domain::get_monomial j > nb_variables" << endl;
		exit(1);
	}
	return Monomials[i * nb_variables + j];
}

std::string &homogeneous_polynomial_domain::get_monomial_symbol_easy(int i)
{
	return monomial_symbols_easy[i];
}

int *homogeneous_polynomial_domain::get_monomial_pointer(int i)
{
	return Monomials + i * nb_variables;
}

int homogeneous_polynomial_domain::evaluate_monomial(int idx_of_monomial, int *coords)
{
	int r;

	r = F->Linear_algebra->evaluate_monomial(
			Monomials + idx_of_monomial * nb_variables,
			coords, nb_variables);
	return r;
}

void homogeneous_polynomial_domain::remake_symbols(int symbol_offset,
		std::string &symbol_mask, std::string &symbol_mask_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols" << endl;
	}

	int i; //, l;
	char label[1000];

	symbols.clear();
	symbols_latex.clear();
	for (i = 0; i < nb_variables; i++) {
		string s;
		snprintf(label, 1000, symbol_mask.c_str(), i + symbol_offset);
		s.assign(label);
		symbols.push_back(s);
	}
	for (i = 0; i < nb_variables; i++) {
		string s;
		snprintf(label, 1000, symbol_mask_latex.c_str(), i + symbol_offset);
		s.assign(label);
		symbols_latex.push_back(s);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols done" << endl;
	}
}

void homogeneous_polynomial_domain::remake_symbols_interval(int symbol_offset,
		int from, int len,
		std::string &symbol_mask, std::string &symbol_mask_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols_interval" << endl;
	}

	int i, j; //, l;
	char label[1000];

	for (j = 0; j < len; j++) {
		i = from + j;
		snprintf(label, 1000, symbol_mask.c_str(), i + symbol_offset);
		symbols[i].assign(label);
	}
	for (j = 0; j < len; j++) {
		i = from + j;
		snprintf(label, 1000, symbol_mask_latex.c_str(), i + symbol_offset);
		symbols_latex[i].assign(label);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols_interval done" << endl;
	}
}

void homogeneous_polynomial_domain::make_monomials(
		monomial_ordering_type Monomial_ordering_type,
		int f_has_variables,
		std::vector<std::string> *variables_txt,
		std::vector<std::string> *variables_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, h, idx, t;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials" << endl;
	}
	
	nb_monomials = Combi.int_n_choose_k(nb_variables + degree - 1, nb_variables - 1);

	solvers::diophant *D;

	D = NEW_OBJECT(solvers::diophant);

	D->open(1, nb_variables);
	D->fill_coefficient_matrix_with(1);
	D->RHSi(0) = degree;
	D->type[0] = t_EQ;
	D->set_x_min_constant(0);
	D->set_x_max_constant(degree);
	D->f_has_sum = TRUE;
	D->sum = degree;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials before D->solve_all_betten" << endl;
	}
	D->solve_all_betten(0 /* verbose_level */);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials after D->solve_all_betten" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"We found " << D->_resultanz << " monomials" << endl;
	}

	int nb_sol;

	D->get_solutions_full_length(Monomials, nb_sol, 0 /* verbose_level */);
	if (f_v) {
		cout << "There are " << nb_sol << " monomials." << endl;

#if 0
		if (nb_sol < 100) {
			Int_matrix_print(Monomials, nb_sol, nb_variables);
		}
		else {
			cout << "too many to print" << endl;
		}
#endif
	}

	if (nb_sol != nb_monomials) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"nb_sol != nb_monomials" << endl;
		cout << "nb_sol=" << nb_sol << endl;
		cout << "nb_monomials=" << nb_monomials << endl;
		exit(1);
	}

	FREE_OBJECT(D);
	
	if (Monomial_ordering_type == t_PART) {

		if (f_v) {
			cout << "homogeneous_polynomial_domain::make_monomials rearranging by partition type:" << endl;
		}
		rearrange_monomials_by_partition_type(verbose_level - 2);

	}

	if (FALSE) {
		cout << "After rearranging by type:" << endl;
		if (nb_monomials < 100) {
			Int_matrix_print(Monomials, nb_monomials, nb_variables);
		}
		else {
			cout << "too many to print" << endl;
		}
	}

	char str[1000];

	symbols.clear();
	for (i = 0; i < nb_variables; i++) {

		string s;

		
		if (f_has_variables) {
			s.assign((*variables_txt)[i]);
		}
		else {
			if (TRUE) {
				str[0] = 'X';
				str[1] = '0' + i;
				str[2] = 0;
			}
			else {
				str[0] = 'A' + i;
				str[1] = 0;
			}

			s.assign(str);
		}
		symbols.push_back(s);
	}


	symbols_latex.clear();
	for (i = 0; i < nb_variables; i++) {

		string s;

		
		if (f_has_variables) {
			s.assign((*variables_tex)[i]);
		}
		else {

			if (TRUE) {
				str[0] = 'X';
				str[1] = '_';
				str[2] = '0' + i;
				str[3] = 0;
			}
			else {
				str[0] = 'A' + i;
				str[1] = 0;
			}

			s.assign(str);
		}
		symbols_latex.push_back(s);
	}

	int f_first = FALSE;

	string label;


	label.assign("");
	monomial_symbols.clear();
	for (i = 0; i < nb_monomials; i++) {
		f_first = TRUE;
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			if (a) {
				if (!f_first) {
					label.append("*");
				}
				else {
					f_first = FALSE;
				}
				label.append(symbols[j]);
				if (a > 1) {
					snprintf(str, sizeof(str), "^%d", a);
					label.append(str);
				}
			}
		}
		monomial_symbols.push_back(label);

	}

	label.assign("");

	monomial_symbols_latex.clear();
	for (i = 0; i < nb_monomials; i++) {
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			if (a) {
				label.append(symbols_latex[j]);
				if (a > 1) {
					if (a >= 10) {
						snprintf(str, sizeof(str), "^{%d}", a);
					}
					else {
						snprintf(str, sizeof(str), "^%d", a);
					}
					label.append(str);
				}
			}
		}
		monomial_symbols_latex.push_back(label);

	}

	Variables = NEW_int(nb_monomials * degree);
	for (i = 0; i < nb_monomials; i++) {
		h = 0;
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			for (t = 0; t < a; t++) {
				Variables[i * degree + h] = j;
				h++;
			}
		}
		if (h != degree) {
			cout << "homogeneous_polynomial_domain::make_monomials "
					"h != degree" << endl;
			exit(1);
		}
	}

	label.assign("");
	monomial_symbols_easy.clear();
	for (i = 0; i < nb_monomials; i++) {
		str[0] = 'X';
		str[1] = 0;
		label.append(str);
		for (j = 0; j < degree; j++) {
			a = Variables[i * degree + j];
			snprintf(str, sizeof(str), "%d", a);
			label.append(str);
		}
		monomial_symbols_easy.push_back(label);

	}


	if (FALSE) {
		cout << "homogeneous_polynomial_domain::make_monomials the "
				"variable lists are:" << endl;
		if (nb_monomials < 100) {
			for (i = 0; i < nb_monomials; i++) {
				cout << i << " : " << monomial_symbols[i] << " : ";
				Int_vec_print(cout, Variables + i * degree, degree);
				cout << endl;
			}
		}
		else {
			cout << "too many to print" << endl;
		}
	}




	nb_affine = NT.i_power_j(nb_variables, degree);

	if (nb_affine < ONE_MILLION) {
		Affine = NEW_int(nb_affine * degree);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::make_monomials  "
					"Affine, nb_affine=" << nb_affine << endl;
		}
		for (h = 0; h < nb_affine; h++) {
			Gg.AG_element_unrank(nb_variables /* q */, Affine + h * degree, 1, degree, h);
		}
		if (FALSE) {
			cout << "homogeneous_polynomial_domain::make_monomials  "
					"Affine" << endl;
			Int_matrix_print(Affine, nb_affine, degree);
		}
		Affine_to_monomial = NEW_int(nb_affine);
		for (i = 0; i < nb_affine; i++) {
			if (i > 0 && (i & ((1 << 20) - 1)) == 0) {
				cout << "homogeneous_polynomial_domain::make_monomials "
						"i = " << i << " / " << nb_affine << endl;
			}
			Int_vec_zero(v, nb_variables);
			for (j = 0; j < degree; j++) {
				a = Affine[i * degree + j];
				v[a]++;
			}
			idx = index_of_monomial(v);
			Affine_to_monomial[i] = idx;
		}
	}
	else {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"nb_affine is too big, skipping Affine_to_monomial" << endl;
		Affine = NULL;
		Affine_to_monomial = NULL;
	}

	if (FALSE) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"Affine : idx:" << endl;
		for (i = 0; i < nb_affine; i++) {
			cout << i << " : ";
			Int_vec_print(cout, Affine + i * degree, degree);
			cout << " : " << Affine_to_monomial[i] << endl;
		}
	}
	

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials done" << endl;
	}
}

void homogeneous_polynomial_domain::rearrange_monomials_by_partition_type(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type before Sorting.Heapsort_general" << endl;
	}
	Sorting.Heapsort_general(Monomials, nb_monomials,
		homogeneous_polynomial_domain_compare_monomial, 
		homogeneous_polynomial_domain_swap_monomial, 
		this);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type after Sorting.Heapsort_general" << endl;
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type done" << endl;
	}
}

int homogeneous_polynomial_domain::index_of_monomial(int *v)
{
	data_structures::sorting Sorting;

#if 0
	int i, j;
	
	for (i = 0; i < nb_monomials; i++) {
		for (j = 0; j < n; j++) {
			if (v[j] != Monomials[i * n + j]) {
				break;
				}
			}
		if (j == n) {
			return i;
			}
		}
#endif
	int idx;
	
	if (!Sorting.search_general(Monomials, nb_monomials, v, idx,
		homogeneous_polynomial_domain_compare_monomial_with, 
		this /* extra_data */, 0 /* verbose_level */)) {

		cout << "homogeneous_polynomial_domain::index_of_monomial "
				"Did not find the monomial v=";
		Int_vec_print(cout, v, nb_variables);
		cout << endl;
		cout << "Monomials:" << endl;
		//int_matrix_print(Monomials, nb_monomials, n);
		int i;
		for (i = 0; i < nb_monomials; i++) {
			cout << setw(3) << i << " : ";
			Int_vec_print(cout, Monomials + i * nb_variables, nb_variables);
			cout << endl;
		}
		cout << "homogeneous_polynomial_domain::index_of_monomial "
				"Did not find the monomial v=";
		Int_vec_print(cout, v, nb_variables);
		cout << endl;
		Sorting.search_general(Monomials, nb_monomials, v, idx,
				homogeneous_polynomial_domain_compare_monomial_with,
				this /* extra_data */, 3);
		exit(1);
	}
	return idx;
}

void homogeneous_polynomial_domain::affine_evaluation_kernel(
		int *&Kernel, int &dim_kernel, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a, b, c, idx, f_kernel;
	int *mon;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::affine_evaluation_kernel" << endl;
	}
	dim_kernel = 0;
	mon = NEW_int(nb_variables);
	for (i = 0; i < nb_monomials; i++) {
		Int_vec_copy(Monomials + i * nb_variables, mon, nb_variables);
		f_kernel = FALSE;
		for (j = 0; j < nb_variables - 1; j++) {
			a = mon[j];
			if (a >= q) {
				b = a % (q - 1);
				if (b == 0) {
					b += (q - 1);
				}
				c = a - b;
				mon[j] = b;
				mon[nb_variables - 1] += c;
				f_kernel = TRUE;
			}
		}
		if (f_kernel) {
			if (FALSE) {
				cout << "homogeneous_polynomial_domain::affine_evaluation_kernel "
						"monomial ";
				Int_vec_print(cout, Monomials + i * nb_variables, nb_variables);
				cout << " = ";
				Int_vec_print(cout, mon, nb_variables);
				cout << endl;
			}
			dim_kernel++;
		}
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::affine_evaluation_kernel "
				"dim_kernel = " << dim_kernel << endl;
	}
	Kernel = NEW_int(dim_kernel * 2);
	h = 0;
	for (i = 0; i < nb_monomials; i++) {
		Int_vec_copy(Monomials + i * nb_variables, mon, nb_variables);
		f_kernel = FALSE;
		for (j = 0; j < nb_variables - 1; j++) {
			a = mon[j];
			if (a >= q) {
				b = a % (q - 1);
				if (b == 0) {
					b += (q - 1);
				}
				c = a - b;
				mon[j] = b;
				mon[nb_variables - 1] += c;
				f_kernel = TRUE;
			}
		}
		if (f_kernel) {
			if (FALSE) {
				cout << "homogeneous_polynomial_domain::affine_evaluation_kernel monomial ";
				Int_vec_print(cout, Monomials + i * nb_variables, nb_variables);
				cout << " = ";
				Int_vec_print(cout, mon, nb_variables);
				cout << endl;
			}
			idx = index_of_monomial(mon);
			Kernel[h * 2 + 0] = i;
			Kernel[h * 2 + 1] = idx;
			h++;
		}
	}
	FREE_int(mon);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::affine_evaluation_kernel done" << endl;
	}
}

void homogeneous_polynomial_domain::get_quadratic_form_matrix(int *eqn, int *M)
{
	int h, i, j, a;

	if (degree != 2) {
		cout << "homogeneous_polynomial_domain::get_quadratic_form_matrix degree != 2" << endl;
		exit(1);
	}
	Int_vec_zero(M, nb_variables * nb_variables);
	for (h = 0; h < nb_monomials; h++) {
		a = eqn[h];
		i = Variables[h * 2 + 0];
		j = Variables[h * 2 + 1];
		M[i * nb_variables + j] = a;
	}
}


void homogeneous_polynomial_domain::print_monomial(ostream &ost, int i)
{
	int j, a, f_first = TRUE;
	
	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = FALSE;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial(ostream &ost, int *mon)
{
	int j, a, f_first = TRUE;
	
	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = FALSE;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial_latex(std::ostream &ost, int *mon)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		ost << symbols_latex[j];
		if (a >= 10) {
			ost << "^{" << a << "}";
		}
		else if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial_latex(std::ostream &ost, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;
	print_monomial_latex(ost, mon);
}

void homogeneous_polynomial_domain::print_monomial_relaxed(std::ostream &ost, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;

	string s;
	print_monomial_relaxed(s, mon);
	ost << s;
}

void homogeneous_polynomial_domain::print_monomial_latex(std::string &s, int *mon)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		s.append(symbols_latex[j]);

		char str[1000];


		if (a >= 10) {
			snprintf(str, sizeof(str), "^{%d}", a);
		}
		else if (a > 1) {
			snprintf(str, sizeof(str), "^%d", a);
		}
		s.append(str);
	}
}

void homogeneous_polynomial_domain::print_monomial_relaxed(std::string &s, int *mon)
{
	int i, j, a;
	int f_first = TRUE;

	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			s.append("*");
		}

		s.append(symbols[j]);

		for (i = 1; i < a; i++) {
			s.append("*");
			s.append(symbols[j]);
		}
	}
}



void homogeneous_polynomial_domain::print_monomial_latex(std::string &s, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;
	print_monomial_latex(s, mon);
}


void homogeneous_polynomial_domain::print_monomial_str(std::stringstream &ost, int i)
{
	int j, a, f_first = TRUE;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = FALSE;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}


void homogeneous_polynomial_domain::print_monomial_latex_str(std::stringstream &ost, int i)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		ost << symbols_latex[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_equation(std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;

	//cout << "homogeneous_polynomial_domain::print_equation" << endl;
	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			//F->print_element(ost, c);
			ost << c;
		}
		print_monomial(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_simple(std::ostream &ost, int *coeffs)
{

	Int_vec_print_fully(cout, coeffs, nb_monomials);
}


void homogeneous_polynomial_domain::print_equation_tex(std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_relaxed(std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			ost << "*";
		}
		print_monomial_relaxed(ost, i);
	}
}


void homogeneous_polynomial_domain::print_equation_numerical(std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			//F->print_element(ost, c);
			ost << c;
		}
		print_monomial(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_lint(std::ostream &ost, long int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_lint_tex(std::ostream &ost, long int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_str(std::stringstream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << "+";
		}
		if (c > 1) {
			F->print_element_str(ost, c);
			//ost << c;
		}
		print_monomial_str(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_with_line_breaks_tex(
		std::ostream &ost, int *coeffs, int nb_terms_per_line,
	const char *new_line_text)
{
	int i, c, cnt = 0;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}

		if ((cnt % nb_terms_per_line) == 0 && cnt) {
			ost << new_line_text;
		}


		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
		cnt++;
	}
}

void homogeneous_polynomial_domain::print_equation_with_line_breaks_tex_lint(
	ostream &ost, long int *coeffs, int nb_terms_per_line,
	const char *new_line_text)
{
	int i, c, cnt = 0;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {

		c = coeffs[i];

		if (c == 0) {
			continue;
		}

		if ((cnt % nb_terms_per_line) == 0 && cnt) {
			ost << new_line_text;
		}

		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
		cnt++;
	}
}

void homogeneous_polynomial_domain::algebraic_set(int *Eqns, int nb_eqns,
		long int *Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, a, i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::algebraic_set" << endl;
	}

	long int N_points;
	geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::algebraic_set "
				"N_points=" << N_points << endl;
	}

	nb_pts = 0;
	for (rk = 0; rk < N_points; rk++) {
		unrank_point(v, rk);
		for (i = 0; i < nb_eqns; i++) {
			a = evaluate_at_a_point(Eqns + i * nb_monomials, v);
			if (a) {
				break;
			}
		}
		if (i == nb_eqns) {
			Pts[nb_pts++] = rk;
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::algebraic_set "
				"done" << endl;
	}
}

void homogeneous_polynomial_domain::polynomial_function(int *coeff, int *f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, a;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::polynomial_function" << endl;
	}
	long int N_points;
	geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::polynomial_function "
				"N_points=" << N_points << endl;
	}

	for (rk = 0; rk < N_points; rk++) {
		unrank_point(v, rk);
		a = evaluate_at_a_point(coeff, v);
		f[rk] = a;
	}
}

void homogeneous_polynomial_domain::polynomial_function_affine(
		int *coeff, int *f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, a;
	long int N;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::polynomial_function_affine" << endl;
	}
	geometry::geometry_global Geo;
	number_theory::number_theory_domain NT;

	N = NT.i_power_j(F->q, nb_variables - 1);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::polynomial_function_affine "
				"N=" << N << endl;
	}

	for (rk = 0; rk < N; rk++) {

		Geo.AG_element_unrank(F->q, v, 1, nb_variables - 1, rk);
		v[nb_variables - 1] = 1;

		//unrank_point(v, rk);
		a = evaluate_at_a_point(coeff, v);
		f[rk] = a;
	}
}


void homogeneous_polynomial_domain::enumerate_points(int *coeff,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	long int rk;
	int a;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points" << endl;
	}

	long int N_points;
	geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);

	if (f_vv) {
		cout << "homogeneous_polynomial_domain::enumerate_points N_points=" << N_points << endl;
		cout << "homogeneous_polynomial_domain::enumerate_points coeff=" << endl;
		Int_vec_print(cout, coeff, nb_monomials);
		cout << endl;
#if 0
		print_equation_with_line_breaks_tex(cout,
				coeff, 8 /* nb_terms_per_line*/,
				"\\\\\n");
		cout << endl;
#endif
	}
	//nb_pts = 0;
	for (rk = 0; rk < N_points; rk++) {
		unrank_point(v, rk);
		a = evaluate_at_a_point(coeff, v);
		if (f_vv) {
			cout << "homogeneous_polynomial_domain::enumerate_points point " << rk << " / " << N_points << " :";
			Int_vec_print(cout, v, nb_variables);
			cout << " evaluates to " << a << endl;
		}
		if (a == 0) {
			Pts.push_back(rk);
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points "
				"done" << endl;
	}
}

void homogeneous_polynomial_domain::enumerate_points_lint(int *coeff,
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint" << endl;
	}

	vector<long int> Points;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint before "
				"enumerate_points" << endl;
	}
	enumerate_points(coeff, Points, verbose_level - 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint after "
				"enumerate_points" << endl;
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint The object "
				"has " << Points.size() << " points" << endl;
	}
	int i;

	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint done" << endl;
	}
}


void homogeneous_polynomial_domain::enumerate_points_zariski_open_set(int *coeff,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int rk;
	int a;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_zariski_open_set" << endl;
	}
	long int N_points;
	geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);

	if (f_vv) {
		cout << "homogeneous_polynomial_domain::enumerate_points_zariski_open_set "
				"N_points=" << N_points << endl;
		print_equation_with_line_breaks_tex(cout,
				coeff, 8 /* nb_terms_per_line*/,
				"\\\\\n");
		cout << endl;
	}
	for (rk = 0; rk < N_points; rk++) {
		unrank_point(v, rk);
		a = evaluate_at_a_point(coeff, v);
		if (a) {
			Pts.push_back(rk);
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_zariski_open_set "
				"done" << endl;
	}
}

int homogeneous_polynomial_domain::evaluate_at_a_point_by_rank(
		int *coeff, int pt)
{
	int a;
	
	unrank_point(v, pt);
	a = evaluate_at_a_point(coeff, v);
	return a;
}

int homogeneous_polynomial_domain::evaluate_at_a_point(
		int *coeff, int *pt_vec)
{
	int i, a, b, c;
	
	a = 0;
	for (i = 0; i < nb_monomials; i++) {
		if (coeff[i] == 0) {
			continue;
		}
		b = F->Linear_algebra->evaluate_monomial(Monomials + i * nb_variables, pt_vec, nb_variables);
		c = F->mult(coeff[i], b);
		a = F->add(a, c);
	}
	return a;
}

void homogeneous_polynomial_domain::substitute_linear(
	int *coeff_in, int *coeff_out,
	int *Mtx_inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_linear" << endl;
		}

	substitute_semilinear(coeff_in, coeff_out, 
		FALSE /* f_semilinear */, 0 /* frob_power */,
		Mtx_inv, verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_linear "
				"done" << endl;
		}
}

void homogeneous_polynomial_domain::substitute_semilinear(
	int *coeff_in, int *coeff_out,
	int f_semilinear, int frob_power, int *Mtx_inv,
	int verbose_level)
// applies frob_power field automorphisms and then performs substitution
{
	int f_v = (verbose_level >= 1);
	int a, b, c, i, j, idx;
	int *A;
	int *V;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_semilinear" << endl;
	}


	if (f_semilinear) {
		F->frobenius_power_vec_to_vec(coeff_in, coeff4, nb_monomials, frob_power);
	}
	else {
		Int_vec_copy(coeff_in, coeff4, nb_monomials);
	}


	Int_vec_zero(coeff3, nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		c = coeff4[i];
		if (c == 0) {
			continue;
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_semilinear monomial " << c << " * ";
		print_monomial(cout, i);
		cout << endl;
#endif
		
		V = Variables + i * degree;
			// a list of the indices of the variables
			// which appear in the monomial
			// (possibly with repeats)
			// Example: the monomial x_0^3 becomes 0,0,0

#if 0
		cout << "variables: ";
		int_vec_print(cout, V, degree);
		cout << endl;

		cout << "Mtx:" << endl;
		int_matrix_print(Mtx_inv, n, n);
#endif

		Int_vec_zero(coeff2, nb_monomials);
		for (a = 0; a < nb_affine; a++) {
			if (Affine) {
				A = Affine + a * degree;
			}
			else {
				A = my_affine;
				Gg.AG_element_unrank(nb_variables /* q */, my_affine, 1, degree, a);
					// sequence of length degree over the alphabet  0,...,n-1.
			}
			for (j = 0; j < degree; j++) {
				//factors[j] = Mtx_inv[V[j] * n + A[j]];
				factors[j] = Mtx_inv[A[j] * nb_variables + V[j]];
			}

			b = F->product_n(factors, degree);
			if (Affine_to_monomial) {
				idx = Affine_to_monomial[a];
			}
			else {
				Int_vec_zero(v, nb_variables);
				for (j = 0; j < degree; j++) {
					a = Affine[i * degree + j];
					v[a]++;
				}
				idx = index_of_monomial(v);
			}

#if 0
			cout << "affine " << a << " / " << nb_affine << " : ";
			int_vec_print(cout, A, 3);
			cout << " factors ";
			int_vec_print(cout, factors, 3);
			cout << " b=" << b << " idx=" << idx << endl;
#endif
			coeff2[idx] = F->add(coeff2[idx], b);
		}
		for (j = 0; j < nb_monomials; j++) {
			coeff2[j] = F->mult(coeff2[j], c);
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_semilinear "
				"monomial " << c << " * ";
		print_monomial(cout, i);
		cout << " yields:" << endl;
		int_vec_print(cout, coeff2, nb_monomials);
		cout << endl;
#endif
		
		for (j = 0; j < nb_monomials; j++) {
			coeff3[j] = F->add(coeff2[j], coeff3[j]);
		}
	}
#if 0
	cout << "homogeneous_polynomial_domain::substitute_semilinear "
			"input:" << endl;
	int_vec_print(cout, coeff_in, nb_monomials);
	cout << endl;
	cout << "homogeneous_polynomial_domain::substitute_semilinear "
			"output:" << endl;
	int_vec_print(cout, coeff3, nb_monomials);
	cout << endl;
#endif





	Int_vec_copy(coeff3, coeff_out, nb_monomials);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_semilinear "
				"done" << endl;
	}
}

void homogeneous_polynomial_domain::substitute_line(
	int *coeff_in, int *coeff_out,
	int *Pt1_coeff, int *Pt2_coeff,
	int verbose_level)
// coeff_in[nb_monomials], coeff_out[degree + 1]
{
	int f_v = (verbose_level >= 1);
	int rk, b, c, i, j, idx;
	int *A;
	int *V;
	int *Mtx;
	int my_nb_affine, wt;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;


	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_line" << endl;
	}

	my_nb_affine = NT.i_power_j(2, degree);

	Mtx = NEW_int(nb_variables * 2);

	for (i = 0; i < nb_variables; i++) {
		Mtx[i * 2 + 0] = Pt1_coeff[i];
		Mtx[i * 2 + 1] = Pt2_coeff[i];
	}

	Int_vec_copy(coeff_in, coeff4, nb_monomials);


	Int_vec_zero(coeff3, degree + 1);

	for (i = 0; i < nb_monomials; i++) {
		c = coeff4[i];
		if (c == 0) {
			continue;
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_line monomial " << c << " * ";
		print_monomial(cout, i);
		cout << endl;
#endif

		V = Variables + i * degree;
			// a list of the indices of the variables
			// which appear in the monomial
			// (possibly with repeats)
			// Example: the monomial x_0^3 becomes 0,0,0

#if 0
		cout << "variables: ";
		int_vec_print(cout, V, degree);
		cout << endl;

		cout << "Mtx:" << endl;
		int_matrix_print(Mtx, n, 2);
#endif

		Int_vec_zero(coeff2, degree + 1);
		for (rk = 0; rk < my_nb_affine; rk++) {

			A = my_affine;
			Gg.AG_element_unrank(2 /* q */, my_affine, 1, degree, rk);
					// sequence of length degree over the alphabet  0,1.

			wt = 0;
			for (j = 0; j < degree; j++) {
				if (my_affine[j]) {
					wt++;
				}
			}
			for (j = 0; j < degree; j++) {
				//factors[j] = Mtx_inv[V[j] * n + A[j]];
				factors[j] = Mtx[V[j] * 2 + A[j]];
			}

			b = F->product_n(factors, degree);

#if 0
			if (Affine_to_monomial) {
				idx = Affine_to_monomial[a];
			}
			else {
				int_vec_zero(v, n);
				for (j = 0; j < degree; j++) {
					a = Affine[i * degree + j];
					v[a]++;
				}
				idx = index_of_monomial(v);
			}
#else
			idx = wt;
#endif

#if 0
			cout << "affine " << a << " / " << nb_affine << " : ";
			int_vec_print(cout, A, 3);
			cout << " factors ";
			int_vec_print(cout, factors, 3);
			cout << " b=" << b << " idx=" << idx << endl;
#endif
			coeff2[idx] = F->add(coeff2[idx], b);
		}
		for (j = 0; j <= degree; j++) {
			coeff2[j] = F->mult(coeff2[j], c);
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_line "
				"monomial " << c << " * ";
		print_monomial(cout, i);
		cout << " yields:" << endl;
		int_vec_print(cout, coeff2, nb_monomials);
		cout << endl;
#endif

		for (j = 0; j <= degree; j++) {
			coeff3[j] = F->add(coeff2[j], coeff3[j]);
		}
	}
#if 0
	cout << "homogeneous_polynomial_domain::substitute_line "
			"input:" << endl;
	int_vec_print(cout, coeff_in, nb_monomials);
	cout << endl;
	cout << "homogeneous_polynomial_domain::substitute_line "
			"output:" << endl;
	int_vec_print(cout, coeff3, nb_monomials);
	cout << endl;
#endif





	Int_vec_copy(coeff3, coeff_out, degree + 1);

	FREE_int(Mtx);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_line "
				"done" << endl;
	}
}

void homogeneous_polynomial_domain::multiply_by_scalar(
	int *coeff_in, int scalar, int *coeff_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, c;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_by_scalar" << endl;
	}

	Int_vec_zero(coeff_out, nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		a = coeff_in[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "coeff_in[" << i << "] = " << a << endl;
		}
		c = F->mult(a, scalar);
		coeff_out[i] = c;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_by_scalar done" << endl;
	}
}


void homogeneous_polynomial_domain::multiply_mod(
	int *coeff1, int *coeff2, int *coeff3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod" << endl;
	}

	Int_vec_zero(coeff3, nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		a = coeff1[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "coeff1[" << i << "] = " << a << endl;
		}
		for (j = 0; j < nb_monomials; j++) {
			b = coeff2[j];
			if (b == 0) {
				continue;
			}
			if (f_v) {
				cout << "coeff2[" << j << "] = " << b << endl;
			}
			c = F->mult(a, b);
			idx = (i + j) % nb_monomials;
			coeff3[idx] = F->add(coeff3[idx], c);
			if (f_v) {
				cout << "coeff3[" << idx << "] += " << c << endl;
			}
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod done" << endl;
	}
}

void homogeneous_polynomial_domain::multiply_mod_negatively_wrapped(
	int *coeff1, int *coeff2, int *coeff3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod_negatively_wrapped" << endl;
	}

	Int_vec_zero(coeff3, nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		a = coeff1[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "coeff1[" << i << "] = " << a << endl;
		}
		for (j = 0; j < nb_monomials; j++) {
			b = coeff2[j];
			if (b == 0) {
				continue;
			}
			if (f_v) {
				cout << "coeff2[" << j << "] = " << b << endl;
			}
			c = F->mult(a, b);
			idx = i + j;
			if (idx < nb_monomials) {
				coeff3[idx] = F->add(coeff3[idx], c);
			}
			else {
				idx = idx % nb_monomials;
				coeff3[idx] = F->add(coeff3[idx], F->negate(c));
			}
			if (f_v) {
				cout << "coeff3[" << idx << "] += " << c << endl;
			}
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod_negatively_wrapped done" << endl;
	}
}


int homogeneous_polynomial_domain::is_zero(int *coeff)
{
	int i;
	
	for (i = 0; i < nb_monomials; i++) {
		if (coeff[i]) {
			return FALSE;
		}
	}
	return TRUE;
}

void homogeneous_polynomial_domain::unrank_point(int *v, long int rk)
{
	//P->unrank_point(v, rk);
	F->PG_element_unrank_modified_lint(v, 1, nb_variables, rk);
}

long int homogeneous_polynomial_domain::rank_point(int *v)
{
	long int rk;

	//rk = P->rank_point(v);
	F->PG_element_rank_modified_lint(v, 1, nb_variables, rk);
	return rk;
}

void homogeneous_polynomial_domain::unrank_coeff_vector(int *v, long int rk)
{
	F->PG_element_unrank_modified_lint(v, 1, nb_monomials, rk);
}

long int homogeneous_polynomial_domain::rank_coeff_vector(int *v)
{
	long int rk;

	F->PG_element_rank_modified_lint(v, 1, nb_monomials, rk);
	return rk;
}

int homogeneous_polynomial_domain::test_weierstrass_form(int rk, 
	int &a1, int &a2, int &a3, int &a4, int &a6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m_one;

	unrank_coeff_vector(coeff2, rk);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_weierstrass_form"
				<< endl;
	}
	if (nb_variables != 3) {
		cout << "homogeneous_polynomial_domain::test_weierstrass_form "
				"nb_variables != 3" << endl;
		exit(1);
	}
	if (degree != 3) {
		cout << "homogeneous_polynomial_domain::test_weierstrass_form "
				"degree != 3" << endl;
		exit(1);
	}
	if (coeff2[1] || coeff2[3] || coeff2[6]) {
		return FALSE;
	}
	F->PG_element_normalize_from_front(coeff2, 1, nb_monomials);
	if (coeff2[0] != 1) {
		return FALSE;
	}
	m_one = F->negate(1);
	if (coeff2[7] != m_one) {
		return FALSE;
	}
	a1 = F->negate(coeff2[4]);
	a2 = coeff2[2];
	a3 = F->negate(coeff2[8]);
	a4 = coeff2[5];
	a6 = coeff2[9];
	return TRUE;
}

void homogeneous_polynomial_domain::explore_vanishing_ideal(long int *Pts,
		int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal" << endl;
	}

	int *Kernel;
	int r, rk;

	Kernel = NEW_int(get_nb_monomials() * get_nb_monomials());


	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal the input set has size " << nb_pts << endl;
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal the input set is: " << endl;
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		//P->print_set_numerical(cout, GOC->Pts, GOC->nb_pts);
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal before vanishing_ideal" << endl;
	}
	vanishing_ideal(Pts, nb_pts,
			rk, Kernel, verbose_level - 1);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal after vanishing_ideal" << endl;
	}

	int h;
	int nb_pts2;
	long int *Pts2;

	r = get_nb_monomials() - rk;

	for (h = 0; h < r; h++) {
		cout << "generator " << h << " / " << r << " is ";
		print_equation_relaxed(cout, Kernel + h * get_nb_monomials());
		cout << endl;

	}


	cout << "looping over all generators of the ideal:" << endl;
	for (h = 0; h < r; h++) {
		cout << "generator " << h << " / " << r << " is ";
		Int_vec_print(cout, Kernel + h * get_nb_monomials(), get_nb_monomials());
		cout << " : " << endl;

		vector<long int> Points;
		int i;

		enumerate_points(Kernel + h * get_nb_monomials(),
				Points, verbose_level);
		nb_pts2 = Points.size();

		Pts2 = NEW_lint(nb_pts2);
		for (i = 0; i < nb_pts2; i++) {
			Pts2[i] = Points[i];
		}


		cout << "We found " << nb_pts2 << " points on the generator of the ideal" << endl;
		cout << "They are : ";
		Lint_vec_print(cout, Pts2, nb_pts2);
		cout << endl;
		//P->print_set_numerical(cout, Pts, nb_pts);

		FREE_lint(Pts2);

	} // next h

	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal done" << endl;
	}

}


void homogeneous_polynomial_domain::vanishing_ideal(long int *Pts,
		int nb_pts, int &r, int *Kernel, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	int *System;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal" << endl;
	}
	System = NEW_int(MAX(nb_pts, nb_monomials) * nb_monomials);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(v, Pts[i]);
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] =
					F->Linear_algebra->evaluate_monomial(Monomials + j * nb_variables, v, nb_variables);
		}
	}
	if (f_vv) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system:" << endl;
		Int_matrix_print(System, nb_pts, nb_monomials);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"before RREF_and_kernel" << endl;
	}
	r = F->Linear_algebra->RREF_and_kernel(nb_monomials,
			nb_pts, System, 0 /* verbose_level */);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system has rank " << r << endl;
	}
	if (f_vv) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system in RREF:" << endl;
		Int_matrix_print(System, r, nb_monomials);
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The kernel:" << endl;
		Int_matrix_print(System + r * nb_monomials,
				nb_monomials - r, nb_monomials);
	}
	Int_vec_copy(System + r * nb_monomials, Kernel,
			(nb_monomials - r) * nb_monomials);
	FREE_int(System);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal done" << endl;
	}
}

int homogeneous_polynomial_domain::compare_monomials(int *M1, int *M2)
{
	data_structures::sorting Sorting;

	if (Monomial_ordering_type == t_PART) {
		return compare_monomials_PART(M1, M2);
	}
	if (Monomial_ordering_type == t_LEX) {
		return Sorting.int_vec_compare(M1, M2, nb_variables) * -1;
	}
	else {
		cout << "homogeneous_polynomial_domain::compare_monomials "
				"monomial ordering unrecognized" << endl;
		exit(1);
	}
}

int homogeneous_polynomial_domain::compare_monomials_PART(int *M1, int *M2)
{
	int h, a;
	int ret = 0;
	
	Int_vec_zero(type1, degree + 1);
	Int_vec_zero(type2, degree + 1);

	for (h = 0; h < nb_variables; h++) {
		a = M1[h];
		type1[a]++;
	}
	for (h = 0; h < nb_variables; h++) {
		a = M2[h];
		type2[a]++;
	}
	for (h = degree; h >= 0; h--) {
		if (type2[h] > type1[h]) {
			//cout << "type2[h] > type1[h] h=" << h << ", needs swap" << endl;
			ret = 1;
			goto the_end;
		}
		if (type2[h] < type1[h]) {
			ret = -1;
			goto the_end;
		}
	}
	
	for (a = degree; a >= 1; a--) {
		for (h = 0; h < nb_variables; h++) {
			if ((M1[h] != a) && (M2[h] != a)) {
				continue;
			}
			if (M1[h] > M2[h]) {
				ret = -1;
				goto the_end;
			}
			if (M1[h] < M2[h]) {
				//cout << "M1[h] < M2[h] h=" << h << ", needs swap" << endl;
				ret = 1;
				goto the_end;
			}
		}
	}

the_end:
	return ret;

}


void homogeneous_polynomial_domain::print_monomial_ordering(ostream &ost)
{
	int h, i, l;
	
	//ost << "The ordering of monomials is:\\\\" << endl;

	for (i = 0; i < (nb_monomials + 24) / 25; i++) {

		l = MINIMUM((i + 1) * 25, nb_monomials) - i * 25;

		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "h &  \\mbox{monomial} & \\mbox{vector} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;

		for (h = 0; h < l; h++) {
			ost << i * 25 + h << " & ";
			print_monomial_latex(ost, i * 25 + h);
			ost << " & ";
			Int_vec_print(ost, Monomials + (i * 25 + h) * nb_variables, nb_variables);
			ost << "\\\\" << endl;
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;

		ost << "\\clearpage" << endl;
	}
}

int *homogeneous_polynomial_domain::read_from_string_coefficient_pairs(std::string &str, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_pairs" << endl;
	}

	int *coeff;
	number_theory::number_theory_domain NT;

	coeff = NEW_int(get_nb_monomials());

	Int_vec_zero(coeff, get_nb_monomials());

	{
		int *coeff_pairs;
		int len;
		int a, b, i;

		Int_vec_scan(str, coeff_pairs, len);
		for (i = 0; i < len / 2; i++) {
			a = coeff_pairs[2 * i];
			b = coeff_pairs[2 * i + 1];
			if (b >= get_nb_monomials()) {
				cout << "homogeneous_polynomial_domain::read_from_string_coefficient_pairs "
						"b >= get_nb_monomials()" << endl;
				exit(1);
			}
			if (b < 0) {
				cout << "homogeneous_polynomial_domain::read_from_string_coefficient_pairs "
						"b < 0" << endl;
				exit(1);
			}
			if (a < 0 || a >= F->q) {
				if (F->e > 1) {
					cout << "homogeneous_polynomial_domain::read_from_string_coefficient_pairs "
							"In a field extension, what do you mean by " << a << endl;
					exit(1);
				}
				a = NT.mod(a, F->q);
			}
			coeff[b] = a;

		}
		FREE_int(coeff_pairs);
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_pairs done" << endl;
	}
	return coeff;
}

int *homogeneous_polynomial_domain::read_from_string_coefficient_vector(std::string &str, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_vector" << endl;
	}

	int *coeff;
	int len;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_vector "
				"before F->read_from_string_coefficient_vector" << endl;
	}
	F->read_from_string_coefficient_vector(str,
				coeff, len,
				verbose_level - 2);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_vector "
				"after F->read_from_string_coefficient_vector" << endl;
	}
	if (len != get_nb_monomials()) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_vector "
				"len != get_nb_monomials()" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_vector done" << endl;
	}
	return coeff;
}




void homogeneous_polynomial_domain::number_of_conditions_satisfied(
		std::string &variety_label_txt,
		std::string &variety_label_tex,
		std::vector<std::string> &Variety_coeffs,
		std::string &number_of_conditions_satisfied_fname,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::number_of_conditions_satisfied" << endl;
	}


	if (f_v) {
		cout << "Reading file " << number_of_conditions_satisfied_fname << " of size "
				<< Fio.file_size(number_of_conditions_satisfied_fname) << endl;
	}
	Fio.read_set_from_file(number_of_conditions_satisfied_fname, Pts, nb_pts, verbose_level);

	int *Cnt;

	Cnt = NEW_int(nb_pts);
	Int_vec_zero(Cnt, nb_pts);


	number_theory::number_theory_domain NT;
	int h, i, a;
	long int rk;
	int *v;

	v = NEW_int(get_nb_variables());


	label_txt.assign(variety_label_txt);
	label_tex.assign(variety_label_tex);
	//fname.append(".txt");



	for (h = 0; h < Variety_coeffs.size(); h++) {

		if (f_v) {
			cout << "homogeneous_polynomial_domain::number_of_conditions_satisfied "
					"h=" << h << " / " << Variety_coeffs.size() << " : ";
			cout << Variety_coeffs[h] << endl;
		}

		int *coeff;

		coeff = read_from_string_coefficient_pairs(Variety_coeffs[h], verbose_level - 2);

		if (f_v) {
			cout << "homogeneous_polynomial_domain::number_of_conditions_satisfied "
					"h=" << h << " / " << Variety_coeffs.size() << " coeff:";
			Int_vec_print(cout, coeff, get_nb_monomials());
			cout << endl;
		}

		for (i = 0; i < nb_pts; i++) {
			rk = Pts[i];
			unrank_point(v, rk);
			a = evaluate_at_a_point(coeff, v);
			if (a == 0) {
				Cnt[i]++;
			}
		}

		FREE_int(coeff);


	} // next h


	data_structures::tally T;

	T.init(Cnt, nb_pts, FALSE, 0);

	cout << "Number of conditions satisfied:" << endl;
	T.print_naked(TRUE);
	cout << endl;

	//T.save_classes_individually(fname);

	int f, l, t, j, pos;

	// go through classes in reverse order:
	for (i = T.nb_types - 1; i >= 0; i--) {

		f = T.type_first[i];
		l = T.type_len[i];
		t = T.data_sorted[f];


		string fname2;
		char str[10000];

		fname2.assign(number_of_conditions_satisfied_fname);
		snprintf(str, sizeof(str), "%d", t);
		fname2.append(str);
		fname2.append(".csv");



		long int *the_class;

		the_class = NEW_lint(l);
		for (j = 0; j < l; j++) {
			pos = T.sorting_perm_inv[f + j];
			the_class[j] = Pts[pos];
		}
		string label;

		label.assign("case");

		Fio.lint_vec_write_csv(the_class, l, fname2, label);

		cout << "class of type " << t << " contains " << l << " elements:" << endl;
		F->display_table_of_projective_points(
				cout, the_class, l, get_nb_variables());

		FREE_lint(the_class);

	}



	FREE_int(Cnt);

	FREE_int(v);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::number_of_conditions_satisfied done" << endl;
	}
}


void homogeneous_polynomial_domain::create_intersection_of_zariski_open_sets(
		std::string &variety_label_txt,
		std::string &variety_label_tex,
		std::vector<std::string> &Variety_coeffs,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_intersection_of_zariski_open_sets" << endl;
	}
	number_theory::number_theory_domain NT;
	int h;
	long int *Pts1;
	int sz1;
	long int *Pts2;
	int sz2;
	data_structures::sorting Sorting;
	long int N_points;
	geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);


	label_txt.assign(variety_label_txt);
	label_tex.assign(variety_label_tex);

	for (h = 0; h < Variety_coeffs.size(); h++) {

		if (f_v) {
			cout << "homogeneous_polynomial_domain::create_intersection_of_zariski_open_sets "
					"h=" << h << " / " << Variety_coeffs.size() << " : ";
			cout << Variety_coeffs[h] << endl;
		}

		int *coeff;

		coeff = read_from_string_coefficient_pairs(Variety_coeffs[h], verbose_level - 2);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::create_intersection_of_zariski_open_sets "
					"h=" << h << " / " << Variety_coeffs.size() << " coeff:";
			Int_vec_print(cout, coeff, get_nb_monomials());
			cout << endl;
		}

		Pts = NEW_lint(N_points);

		if (f_v) {
			cout << "homogeneous_polynomial_domain::create_intersection_of_zariski_open_sets "
					"before HPD->enumerate_points_zariski_open_set" << endl;
		}

		vector<long int> Points;

		enumerate_points_zariski_open_set(coeff, Points, verbose_level);

		FREE_int(coeff);

		if (h == 0) {
			int i;
			nb_pts = Points.size();
			Pts1 = NEW_lint(nb_pts);
			Pts2 = NEW_lint(nb_pts);
			for (i = 0; i < nb_pts; i++) {
				Pts1[i] = Points[i];
			}
			sz1 = nb_pts;
		}
		else {
			int i, idx;
			long int a;
			nb_pts = Points.size();
			sz2 = 0;
			for (i = 0; i < nb_pts; i++) {
				a = Points[i];
				if (Sorting.lint_vec_search(Pts1, sz1, a, idx, 0)) {
					Pts2[sz2++] = a;
				}
			}
			Lint_vec_copy(Pts2, Pts1, sz2);
			sz1 = sz2;
		}
		if (f_v) {
			cout << "homogeneous_polynomial_domain::create_intersection_of_zariski_open_sets "
					"after HPD->enumerate_points_zariski_open_set, "
					"nb_pts = " << nb_pts << endl;
		}
	} // next h

	nb_pts = sz1;
	Pts = NEW_lint(sz1);
	Lint_vec_copy(Pts1, Pts, sz1);

	F->display_table_of_projective_points(
			cout, Pts, nb_pts, get_nb_variables());

	FREE_lint(Pts1);
	FREE_lint(Pts2);



	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_intersection_of_zariski_open_sets done" << endl;
	}
}


void homogeneous_polynomial_domain::create_projective_variety(
		std::string &variety_label,
		std::string &variety_label_tex,
		std::string &variety_coeffs,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety" << endl;
	}

	number_theory::number_theory_domain NT;
	long int N_points;
	geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);


	label_txt.assign(variety_label);
	label_tex.append(variety_label_tex);

	int *coeff;
	int sz;

	Get_int_vector_from_label(variety_coeffs, coeff, sz, verbose_level);

	if (sz != get_nb_monomials()) {
		cout << "homogeneous_polynomial_domain::create_projective_variety "
				"the number of coefficients should be " << get_nb_monomials()
				<< " but is " << sz << endl;
		exit(1);
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety coeff:";
		Int_vec_print(cout, coeff, get_nb_monomials());
		cout << endl;
	}

	Pts = NEW_lint(N_points);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety "
				"before HPD->enumerate_points" << endl;
	}

	vector<long int> Points;
	int i;

	enumerate_points(coeff, Points, verbose_level);
	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety "
				"after HPD->enumerate_points, nb_pts = " << nb_pts << endl;
	}

	F->display_table_of_projective_points(
			cout, Pts, nb_pts, get_nb_variables());

	FREE_int(coeff);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety done" << endl;
	}
}

void homogeneous_polynomial_domain::create_ideal(
		std::string &ideal_label,
		std::string &ideal_label_tex,
		std::string &ideal_point_set_label,
		int &dim_kernel, int &nb_monomials, int *&Kernel,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_ideal" << endl;
	}

	number_theory::number_theory_domain NT;


	nb_monomials = get_nb_monomials();


	long int *Pts;
	int nb_pts;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_ideal "
				"ideal_point_set_label=" << ideal_point_set_label << endl;
	}

	orbiter_kernel_system::Orbiter->get_lint_vector_from_label(ideal_point_set_label, Pts, nb_pts, verbose_level);

	if (f_v) {
		cout << "polynomial_ring_activity::create_ideal "
				"nb_pts=" << nb_pts << endl;
		cout << "polynomial_ring_activity::create_ideal "
				"points:" << endl;
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_ideal "
				"before HPD->vanishing_ideal" << endl;
	}

	Kernel = NEW_int(nb_monomials * nb_monomials);

	int r;

	vanishing_ideal(Pts,
			nb_pts, r, Kernel, verbose_level - 3);


	dim_kernel = nb_monomials - r;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_ideal done" << endl;
	}
}



void homogeneous_polynomial_domain::create_projective_curve(
		std::string &variety_label_txt,
		std::string &variety_label_tex,
		std::string &curve_coeffs,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_curve" << endl;
	}

	int *coeff;

	if (get_nb_variables() != 3) {
		cout << "homogeneous_polynomial_domain::create_projective_curve number of variables must be 3" << endl;
		exit(1);
	}

	coeff = NEW_int(get_nb_monomials());
	Int_vec_zero(coeff, get_nb_monomials());

	label_txt.assign(variety_label_txt);
	label_tex.assign(variety_label_tex);
	int *coeffs;
	int len, i, j, a, b, c, s, t;
	int *v;
	int v2[2];

	Int_vec_scan(curve_coeffs, coeffs, len);
	if (len != degree + 1) {
		cout << "homogeneous_polynomial_domain::create_projective_curve "
				"len != degree + 1" << endl;
		exit(1);
	}

	nb_pts = F->q + 1;

	v = NEW_int(get_nb_variables());
	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		F->PG_element_unrank_modified(v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		for (j = 0; j < get_nb_variables(); j++) {
			a = get_monomial(j, 0);
			b = get_monomial(j, 1);
			v[j] = F->mult3(coeffs[j], F->power(s, a), F->power(t, b));
		}
		F->PG_element_rank_modified(v, 1, get_nb_variables(), c);
		Pts[i] = c;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, get_nb_variables());
			cout << " : " << setw(5) << c << endl;
		}
	}

	F->display_table_of_projective_points(
			cout, Pts, nb_pts, get_nb_variables());


	FREE_int(v);
	FREE_int(coeffs);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_curve done" << endl;
	}
}


void homogeneous_polynomial_domain::get_coefficient_vector(
		expression_parser::formula *Formula,
		std::string &evaluate_text,
		int *Coefficient_vector,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector" << endl;
		cout << "formula:" << endl;
		Formula->print();
	}

	if (!Formula->f_is_homogeneous) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"Formula is not homogeneous" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"Formula is homogeneous of degree " << Formula->degree << endl;
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"Formula->nb_managed_vars = " << Formula->nb_managed_vars << endl;
	}
	if (Formula->nb_managed_vars != nb_variables) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"Formula->nb_managed_vars != nb_variables" << endl;
		exit(1);
	}
	if (Formula->degree != degree) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"Formula->nb_managed_vars != degree" << endl;
		exit(1);
	}


	expression_parser::syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(this, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"after Formula->get_subtrees" << endl;
	}

	int i;

	for (i = 0; i < nb_monomials; i++) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector Monomial " << i << " : ";
		if (Subtrees[i]) {
			Subtrees[i]->print_expression(cout);
			cout << " * ";
			print_monomial(cout, i);
			cout << endl;
		}
		else {
			cout << "homogeneous_polynomial_domain::get_coefficient_vector no subtree" << endl;
		}
	}


	//int *Coefficient_vector;

	//Coefficient_vector = NEW_int(nb_monomials);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"before Formula->evaluate" << endl;
	}
	Formula->evaluate(this,
			Subtrees, evaluate_text, Coefficient_vector,
			verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"after Formula->evaluate" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"coefficient vector:" << endl;
		Int_vec_print(cout, Coefficient_vector, nb_monomials);
		cout << endl;
	}

#if 0
	del_pezzo_surface_of_degree_two_domain *del_Pezzo;

	del_Pezzo = NEW_OBJECT(del_pezzo_surface_of_degree_two_domain);

	del_Pezzo->init(P, Poly4_3, verbose_level);

	del_pezzo_surface_of_degree_two_object *del_Pezzo_surface;

	del_Pezzo_surface = NEW_OBJECT(del_pezzo_surface_of_degree_two_object);

	del_Pezzo_surface->init(del_Pezzo,
			Formula, Subtrees, Coefficient_vector,
			verbose_level);

	del_Pezzo_surface->enumerate_points_and_lines(verbose_level);

	del_Pezzo_surface->pal->write_points_to_txt_file(Formula->name_of_formula, verbose_level);

	del_Pezzo_surface->create_latex_report(Formula->name_of_formula, Formula->name_of_formula_latex, verbose_level);

	FREE_OBJECT(del_Pezzo_surface);
	FREE_OBJECT(del_Pezzo);
#endif

	//FREE_int(Coefficient_vector);
	//FREE_OBJECT(Poly);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector done" << endl;
	}
}


void homogeneous_polynomial_domain::evaluate_regular_map(
		int *Coefficient_vector,
		int nb_eqns,
		geometry::projective_space *P,
		long int *&Image_pts, int &N_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::evaluate_regular_map" << endl;
	}

	if (nb_eqns != P->n + 1) {
		cout << "homogeneous_polynomial_domain::evaluate_regular_map nb_eqns != P->n + 1" << endl;
		exit(1);
	}

	int *v;
	int *w;
	int h;
	long int i, j;
	int f_vv = FALSE;

	N_points = P->N_points;
	Image_pts = NEW_lint(N_points);
	v = NEW_int(P->n + 1);
	w = NEW_int(P->n + 1);

	for (i = 0; i < N_points; i++) {

#if 0
		if (i == 98 || i == 99) {
			f_vv = TRUE;
		}
		else {
			f_vv = FALSE;
		}
#endif

		P->unrank_point(v, i);

		if (f_vv) {
			cout << "homogeneous_polynomial_domain::evaluate_regular_map point " << i << " is ";
			Int_vec_print(cout, v, P->n + 1);
			cout << endl;
		}

		for (h = 0; h < P->n + 1; h++) {
			w[h] = evaluate_at_a_point(Coefficient_vector + h * nb_monomials, v);
		}


		if (!Int_vec_is_zero(w, P->n + 1)) {
			j = P->rank_point(w);
		}
		else {
			j = -1;
		}

		if (f_vv) {
			cout << "homogeneous_polynomial_domain::evaluate_regular_map maps to ";
			Int_vec_print(cout, w, P->n + 1);
			cout << " = " << j << endl;
		}

		Image_pts[i] = j;
	}
	FREE_int(v);
	FREE_int(w);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::evaluate_regular_map done" << endl;
	}
}



// #############################################################################
// global functions:
// #############################################################################






static int homogeneous_polynomial_domain_compare_monomial_with(
		void *data, int i, void *data2, void *extra_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) extra_data;
	int *Data;
	int ret, nb_variables;

	Data = (int *) data;
	nb_variables = HPD->nb_variables;
	ret = HPD->compare_monomials(Data + i * nb_variables, (int *) data2);
	return ret;
}

static int homogeneous_polynomial_domain_compare_monomial(
		void *data, int i, int j, void *extra_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) extra_data;
	int *Data;
	int ret, nb_variables;

	Data = (int *) data;
	nb_variables = HPD->nb_variables;
	ret = HPD->compare_monomials(Data + i * nb_variables, Data + j * nb_variables);
	return ret;
}

static void homogeneous_polynomial_domain_swap_monomial(
		void *data, int i, int j, void *extra_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) extra_data;
	int *Data;
	int h, a, nb_variables;

	Data = (int *) data;
	nb_variables = HPD->nb_variables;

	for (h = 0; h < nb_variables; h++) {
		a = Data[i * nb_variables + h];
		Data[i * nb_variables + h] = Data[j * nb_variables + h];
		Data[j * nb_variables + h] = a;
	}
	
}

#if 0

static void HPD_callback_print_function(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;

	coeff = NEW_int(HPD->get_nb_monomials());
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
}

static void HPD_callback_print_function2(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;
	//long int *Pts;
	//int nb_pts;
	vector<long int> Points;

	//Pts = NEW_lint(HPD->get_P()->N_points);
	coeff = NEW_int(HPD->get_nb_monomials());
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	HPD->enumerate_points(coeff, Points,  0 /*verbose_level*/);
	ost << Points.size();
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	//HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
	//FREE_lint(Pts);
}
#endif




}}}



