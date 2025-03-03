// homogeneous_polynomial_domain.cpp
//
// Anton Betten
//
// September 9, 2016



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace ring_theory {


static int homogeneous_polynomial_domain_compare_monomial_with(
		void *data,
	int i, void *data2, void *extra_data);
static int homogeneous_polynomial_domain_compare_monomial(
		void *data,
	int i, int j, void *extra_data);
static void homogeneous_polynomial_domain_swap_monomial(
		void *data,
	int i, int j, void *extra_data);



homogeneous_polynomial_domain::homogeneous_polynomial_domain()
{
	Record_birth();
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
	Record_death();
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

void homogeneous_polynomial_domain::init(
		polynomial_ring_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init" << endl;
	}

	algebra::field_theory::finite_field *F;

	if (!Descr->f_field) {
		cout << "Please specify whether the polynomial "
				"ring is over a field" << endl;
		exit(1);
	}

	F = Get_finite_field(Descr->finite_field_label);

	if (!Descr->f_number_of_variables) {
		cout << "Please specify the number of variables "
				"of the polynomial ring using "
				"-number_of_variables <n>" << endl;
		exit(1);
	}

	if (!Descr->f_homogeneous) {
		cout << "Please specify the degree of the homogeneous "
				"polynomial ring using -homogeneous <d>" << endl;
		exit(1);
	}



	if (Descr->f_variables) {
		other::data_structures::string_tools ST;
		std::vector<std::string> managed_variables_txt;
		std::vector<std::string> managed_variables_tex;

		ST.parse_comma_separated_strings(
				Descr->variables_txt, managed_variables_txt);
		ST.parse_comma_separated_strings(
				Descr->variables_tex, managed_variables_tex);

		if (managed_variables_txt.size() != managed_variables_tex.size()) {
			cout << "number of variables in txt and in tex differ" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init "
					"before init with variables" << endl;
		}
		init_with_or_without_variables(F,
				Descr->number_of_variables,
				Descr->homogeneous_of_degree,
				Descr->Monomial_ordering_type,
				true,
				&managed_variables_txt,
				&managed_variables_tex,
				verbose_level);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init "
					"after init with variables" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init "
					"before init w/o variables" << endl;
		}
		init_with_or_without_variables(F,
				Descr->number_of_variables,
				Descr->homogeneous_of_degree,
				Descr->Monomial_ordering_type,
				false,
				NULL,
				NULL,
				verbose_level);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::init "
					"after init w/o variables" << endl;
		}

	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init done" << endl;
	}
}

void homogeneous_polynomial_domain::init(
		algebra::field_theory::finite_field *F,
		int nb_vars, int degree,
		monomial_ordering_type Monomial_ordering_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init" << endl;
		cout << "homogeneous_polynomial_domain::init nb_vars = " << nb_vars << endl;
		cout << "homogeneous_polynomial_domain::init degree = " << degree << endl;
	}


	ring_theory_global RG;
	std::string s;

	RG.Monomial_ordering_type_as_string(Monomial_ordering_type, s);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init "
				"nb_vars = " << nb_vars << endl;
		cout << "homogeneous_polynomial_domain::init "
				"degree = " << degree << endl;
		cout << "homogeneous_polynomial_domain::init "
				"Monomial_ordering_type = " << s << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init "
				"before init_with_or_without_variables" << endl;
	}
	init_with_or_without_variables(
			F, nb_vars, degree,
			Monomial_ordering_type,
			false, NULL, NULL,
			verbose_level - 2);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init "
				"after init_with_or_without_variables" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init done" << endl;
	}
}

void homogeneous_polynomial_domain::init_with_or_without_variables(
		algebra::field_theory::finite_field *F,
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
		cout << "homogeneous_polynomial_domain::init_with_or_without_variables "
				"before make_monomials" << endl;
	}
	make_monomials(Monomial_ordering_type,
			f_has_variables, variables_txt, variables_tex,
			verbose_level - 2);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init_with_or_without_variables "
				"after make_monomials" << endl;
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


void homogeneous_polynomial_domain::print_latex(
		std::ostream &ost)
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

algebra::field_theory::finite_field *homogeneous_polynomial_domain::get_F()
{
	return F;
}

std::string homogeneous_polynomial_domain::get_symbol(
		int i)
{
	return symbols[i];
}

std::string homogeneous_polynomial_domain::list_of_variables()
{
	string s;
	int i;

	for (i = 0; i < nb_variables; i++) {
		s += symbols[i];
		if (i < nb_variables - 1) {
			s += ",";
		}
	}
	return s;
}

int homogeneous_polynomial_domain::variable_index(std::string &s)
{
	int i;

	for (i = 0; i < nb_variables; i++) {
		if (symbols[i] == s) {
			return i;
		}
	}
	return -1;
}

int homogeneous_polynomial_domain::get_monomial(
		int i, int j)
{
	if (j > nb_variables) {
		cout << "homogeneous_polynomial_domain::get_monomial "
				"j > nb_variables" << endl;
		exit(1);
	}
	return Monomials[i * nb_variables + j];
}

std::string homogeneous_polynomial_domain::get_monomial_symbol_easy(
		int i)
{
	return monomial_symbols_easy[i];
}

std::string homogeneous_polynomial_domain::get_monomial_symbols_latex(
		int i)
{
	return monomial_symbols_latex[i];
}

std::string homogeneous_polynomial_domain::get_monomial_symbols(
		int i)
{
	return monomial_symbols[i];
}

int *homogeneous_polynomial_domain::get_monomial_pointer(
		int i)
{
	return Monomials + i * nb_variables;
}

int homogeneous_polynomial_domain::evaluate_monomial(
		int idx_of_monomial, int *coords)
{
	int r;

	r = F->Linear_algebra->evaluate_monomial(
			Monomials + idx_of_monomial * nb_variables,
			coords, nb_variables);
	return r;
}

void homogeneous_polynomial_domain::remake_symbols(
		int symbol_offset,
		std::string &symbol_mask, std::string &symbol_mask_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols symbol_mask = " << symbol_mask << endl;
		cout << "homogeneous_polynomial_domain::remake_symbols symbol_mask_latex = " << symbol_mask_latex << endl;
	}

	other::data_structures::string_tools ST;

	int i;

	symbols.clear();
	symbols_latex.clear();

	for (i = 0; i < nb_variables; i++) {

		string label;

		label = ST.printf_d(symbol_mask, i + symbol_offset);

		symbols.push_back(label);
	}
	for (i = 0; i < nb_variables; i++) {

		string label;

		label = ST.printf_d(symbol_mask_latex, i + symbol_offset);

		symbols_latex.push_back(label);
	}


	if (f_v) {
		print_symbols(cout);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols done" << endl;
	}
}

void homogeneous_polynomial_domain::remake_symbols_interval(
		int symbol_offset,
		int from, int len,
		std::string &symbol_mask, std::string &symbol_mask_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::remake_symbols_interval" << endl;
	}

	other::data_structures::string_tools ST;

	int i, j;

	for (j = 0; j < len; j++) {
		i = from + j;

		string label;

		label = ST.printf_d(symbol_mask, i + symbol_offset);

		symbols[i].assign(label);
	}
	for (j = 0; j < len; j++) {
		i = from + j;

		string label;

		label = ST.printf_d(symbol_mask_latex, i + symbol_offset);

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
	algebra::number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials" << endl;
	}
	
	nb_monomials = Combi.int_n_choose_k(
			nb_variables + degree - 1, nb_variables - 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"nb_monomials = " << nb_monomials << endl;
	}
	combinatorics::solvers::diophant *D;

	D = NEW_OBJECT(combinatorics::solvers::diophant);

	D->open(1, nb_variables, verbose_level - 1);
	D->fill_coefficient_matrix_with(1);
	D->RHSi(0) = degree;
	D->type[0] = t_EQ;
	D->set_x_min_constant(0);
	D->set_x_max_constant(degree);
	D->f_has_sum = true;
	D->sum = degree;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"before D->solve_all_betten" << endl;
	}
	D->solve_all_betten(verbose_level - 1);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"after D->solve_all_betten" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"We found " << D->_resultanz << " monomials" << endl;
	}

	int nb_sol;

	D->get_solutions(Monomials, nb_sol, verbose_level - 1);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"There are " << nb_sol << " monomials." << endl;

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
			cout << "homogeneous_polynomial_domain::make_monomials "
					"rearranging by partition type:" << endl;
		}
		rearrange_monomials_by_partition_type(verbose_level - 2);

	}

	if (false) {
		cout << "After rearranging by type:" << endl;
		if (nb_monomials < 100) {
			Int_matrix_print(
					Monomials, nb_monomials, nb_variables);
		}
		else {
			cout << "too many to print" << endl;
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"making symbols" << endl;
	}

	symbols.clear();
	for (i = 0; i < nb_variables; i++) {

		string s;
		char str[1000];

		
		if (f_has_variables) {
			s.assign((*variables_txt)[i]);
		}
		else {
			if (true) {
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

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"after making symbols" << endl;
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"making symbols_latex" << endl;
	}


	symbols_latex.clear();
	for (i = 0; i < nb_variables; i++) {

		string s;
		char str[1000];

		
		if (f_has_variables) {
			s.assign((*variables_tex)[i]);
		}
		else {

			if (true) {
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

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"after making symbols_latex" << endl;
	}


	int f_first = false;

	string label;


	monomial_symbols.clear();
	for (i = 0; i < nb_monomials; i++) {
		label = "";
		f_first = true;
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			if (a) {
				if (!f_first) {
					label += "*";
				}
				else {
					f_first = false;
				}
				label += symbols[j];
				if (a > 1) {
					label += "^" + std::to_string(a);
				}
			}
		}
		monomial_symbols.push_back(label);

	}


	monomial_symbols_latex.clear();
	for (i = 0; i < nb_monomials; i++) {
		label = "";
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			if (a) {
				label += symbols_latex[j];
				if (a > 1) {
					if (a >= 10) {
						label += "^{" + std::to_string(a) + "}";
					}
					else {
						label += "^" + std::to_string(a);
					}
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

	monomial_symbols_easy.clear();
	for (i = 0; i < nb_monomials; i++) {
		label = "";
		label += "X";
		for (j = 0; j < degree; j++) {
			a = Variables[i * degree + j];
			label += std::to_string(a);
		}
		monomial_symbols_easy.push_back(label);

	}


	if (false) {
		cout << "homogeneous_polynomial_domain::make_monomials the "
				"variable lists are:" << endl;
		if (nb_monomials < 100) {
			for (i = 0; i < nb_monomials; i++) {
				cout << i << " : " << monomial_symbols[i] << " : ";
				Int_vec_print(
						cout, Variables + i * degree, degree);
				cout << endl;
			}
		}
		else {
			cout << "too many to print" << endl;
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials  "
				"before making Affine" << endl;
	}


	ring_theory::longinteger_object Nb_affine, Bound;
	ring_theory::longinteger_domain Longint;

	NT.i_power_j_longinteger(
			nb_variables, degree, Nb_affine);

	Bound.create(ONE_MILLION);

	int f_ok;

	if (Longint.is_less_than(Nb_affine, Bound)) {
		f_ok = true;
		nb_affine = Nb_affine.as_lint();
	}
	else {
		f_ok = false;
		nb_affine = -1;
	}


	//nb_affine = NT.i_power_j(nb_variables, degree);
		// could be negative in case of overflow!

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials  "
				"f_ok = " << f_ok << endl;
		cout << "homogeneous_polynomial_domain::make_monomials  "
				"nb_affine = " << nb_affine << endl;
		cout << "homogeneous_polynomial_domain::make_monomials  "
				"nb_affine * degree = " << nb_affine * degree << endl;
	}


	if (f_ok) {
		Affine = NEW_int(nb_affine * degree);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::make_monomials  "
					"Affine, nb_affine=" << nb_affine << endl;
		}
		for (h = 0; h < nb_affine; h++) {
			Gg.AG_element_unrank(
					nb_variables /* q */,
					Affine + h * degree, 1, degree, h);
		}
		if (false) {
			cout << "homogeneous_polynomial_domain::make_monomials  "
					"Affine" << endl;
			Int_matrix_print(
					Affine, nb_affine, degree);
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

	if (false) {
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
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type "
				"before Sorting.Heapsort_general" << endl;
	}
	Sorting.Heapsort_general(Monomials, nb_monomials,
		homogeneous_polynomial_domain_compare_monomial, 
		homogeneous_polynomial_domain_swap_monomial, 
		this);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type "
				"after Sorting.Heapsort_general" << endl;
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type done" << endl;
	}
}

int homogeneous_polynomial_domain::index_of_monomial(
		int *v)
{
	other::data_structures::sorting Sorting;

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
	
	if (!Sorting.search_general(
			Monomials, nb_monomials, v, idx,
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
			Int_vec_print(
					cout, Monomials + i * nb_variables, nb_variables);
			cout << endl;
		}
		cout << "homogeneous_polynomial_domain::index_of_monomial "
				"Did not find the monomial v=";
		Int_vec_print(cout, v, nb_variables);
		cout << endl;
		Sorting.search_general(
				Monomials, nb_monomials, v, idx,
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
		f_kernel = false;
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
				f_kernel = true;
			}
		}
		if (f_kernel) {
			if (false) {
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
		f_kernel = false;
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
				f_kernel = true;
			}
		}
		if (f_kernel) {
			if (false) {
				cout << "homogeneous_polynomial_domain::affine_evaluation_kernel "
						"monomial ";
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

void homogeneous_polynomial_domain::get_quadratic_form_matrix(
		int *eqn, int *M)
{
	int h, i, j, a;

	if (degree != 2) {
		cout << "homogeneous_polynomial_domain::get_quadratic_form_matrix "
				"degree != 2" << endl;
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


void homogeneous_polynomial_domain::print_symbols(
		std::ostream &ost)
{
	int i;

	cout << "homogeneous_polynomial_domain::print_symbols symbols:" << endl;
	for (i = 0; i < nb_variables; i++) {

		cout << i << " : " << symbols[i] << endl;
	}
	cout << "homogeneous_polynomial_domain::print_symbols symbols_latex:" << endl;
	for (i = 0; i < nb_variables; i++) {

		cout << i << " : " << symbols_latex[i] << endl;
	}

}

std::string homogeneous_polynomial_domain::stringify_monomial(
		int i)
{
	int j, a, f_first = true;
	string output;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			output += "*";
		}
		else {
			f_first = false;
		}
		output += symbols[j];
		if (a > 1) {
			output += "^" + std::to_string(a);
		}
	}
	return output;
}

void homogeneous_polynomial_domain::print_monomial(
		std::ostream &ost, int i)
{
	int j, a, f_first = true;
	
	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = false;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial(
		std::ostream &ost, int *mon)
{
	int j, a, f_first = true;
	
	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = false;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial_latex(
		std::ostream &ost, int *mon)
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

void homogeneous_polynomial_domain::print_monomial_latex(
		std::ostream &ost, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;
	print_monomial_latex(ost, mon);
}

void homogeneous_polynomial_domain::print_monomial_relaxed(
		std::ostream &ost, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;

	string s;
	print_monomial_relaxed(s, mon);
	ost << s;
}

void homogeneous_polynomial_domain::print_monomial_latex(
		std::string &s, int *mon)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		s += symbols_latex[j];

		if (a >= 10) {
			s += "^{" + std::to_string(a) + "}";
		}
		else if (a > 1) {
			s += "^" + std::to_string(a);
		}
	}
}

void homogeneous_polynomial_domain::print_monomial_relaxed(
		std::string &s, int *mon)
{
	int j, a;
	int f_first = true;

	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			s += "*";
		}

		s += symbols[j];

		if (a > 1) {
			s += "^" + std::to_string(a);
		}
	}
}



void homogeneous_polynomial_domain::print_monomial_latex(
		std::string &s, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;
	print_monomial_latex(s, mon);
}


void homogeneous_polynomial_domain::print_monomial_str(
		std::stringstream &ost, int i)
{
	int j, a, f_first = true;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = false;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial_for_gap_str(
		std::stringstream &ost, int i)
{
	int j, a, f_first = true;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = false;
		}
		//ost << symbols[j];
		ost << "(r." << j + 1 << ")";
		if (a > 1) {
			ost << "^" << a;
		}
	}
}



void homogeneous_polynomial_domain::print_monomial_latex_str(
		std::stringstream &ost, int i)
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

void homogeneous_polynomial_domain::print_equation(
		std::ostream &ost, int *coeffs)
{
	std::string s;

	s = stringify_equation(coeffs);
	ost << s;

#if 0
	int i, c;
	int f_first = true;

	//cout << "homogeneous_polynomial_domain::print_equation" << endl;
	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			//F->print_element(ost, c);
			ost << c << "*";
		}
		print_monomial(ost, i);
	}
#endif
}


std::string homogeneous_polynomial_domain::stringify_equation(
		int *coeffs)
{
	int i, c;
	int f_first = true;
	string s;

	//cout << "homogeneous_polynomial_domain::print_equation" << endl;
	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			s += " + ";
		}
		if (c > 1) {
			//F->print_element(ost, c);
			s += std::to_string(c) + "*";
		}
		s += stringify_monomial(i);
		//print_monomial(ost, i);
	}
	return s;
}


void homogeneous_polynomial_domain::print_equation_simple(
		std::ostream &ost, int *coeffs)
{

	Int_vec_print_fully(cout, coeffs, nb_monomials);
}


void homogeneous_polynomial_domain::print_equation_tex(
		std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->Io->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
	}

	if (f_first) {
		ost << "0";
	}
}

void homogeneous_polynomial_domain::print_equation_relaxed(
		std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->Io->print_element(ost, c);
			ost << "*";
		}
		print_monomial_relaxed(ost, i);
	}
}


void homogeneous_polynomial_domain::print_equation_numerical(
		std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
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

void homogeneous_polynomial_domain::print_equation_lint(
		std::ostream &ost, long int *coeffs)
{
	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->Io->print_element(ost, c);
			//ost << c;
		}
		print_monomial(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_lint_tex(
		std::ostream &ost, long int *coeffs)
{
	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->Io->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_str(
		std::stringstream &ost, int *coeffs)
{
	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			ost << "+";
		}
		if (c > 1) {
			F->Io->print_element_str(ost, c);
			//ost << c;
		}
		print_monomial_str(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_for_gap_str(
		std::stringstream &ost, int *coeffs)
{
	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			ost << "+";
		}
		if (c > 1) {
			F->Io->print_element_str(ost, c);
			//ost << c;
		}
		print_monomial_for_gap_str(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_with_line_breaks_tex(
		std::ostream &ost, int *coeffs, int nb_terms_per_line,
	const char *new_line_text)
{
	int i, c, cnt = 0;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}

		if ((cnt % nb_terms_per_line) == 0 && cnt) {
			ost << new_line_text;
		}


		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->Io->print_element(ost, c);
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
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {

		c = coeffs[i];

		if (c == 0) {
			continue;
		}

		if ((cnt % nb_terms_per_line) == 0 && cnt) {
			ost << new_line_text;
		}

		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->Io->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
		cnt++;
	}
}

void homogeneous_polynomial_domain::algebraic_set(
		int *Eqns, int nb_eqns,
		long int *Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, a, i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::algebraic_set" << endl;
	}

	long int N_points;
	geometry::other_geometry::geometry_global Gg;

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

void homogeneous_polynomial_domain::polynomial_function(
		int *coeff, int *f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, a;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::polynomial_function" << endl;
	}
	long int N_points;
	geometry::other_geometry::geometry_global Gg;

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
	geometry::other_geometry::geometry_global Geo;
	algebra::number_theory::number_theory_domain NT;

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


void homogeneous_polynomial_domain::enumerate_points(
		int *coeff,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	long int rk;
	int a;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points" << endl;
	}

	long int N_points;
	geometry::other_geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);

	if (f_vv) {
		cout << "homogeneous_polynomial_domain::enumerate_points "
				"N_points=" << N_points << endl;
		cout << "homogeneous_polynomial_domain::enumerate_points "
				"coeff=" << endl;
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
			cout << "homogeneous_polynomial_domain::enumerate_points "
					"point " << rk << " / " << N_points << " :";
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


void homogeneous_polynomial_domain::enumerate_points_in_intersection(
		int *coeff1,
		int *coeff2,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	long int rk;
	int a, b;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection" << endl;
	}

	long int N_points;
	geometry::other_geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);

	if (f_vv) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection "
				"N_points=" << N_points << endl;
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection "
				"coeff1=" << endl;
		Int_vec_print(cout, coeff1, nb_monomials);
		cout << endl;
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection "
				"coeff2=" << endl;
		Int_vec_print(cout, coeff2, nb_monomials);
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
		a = evaluate_at_a_point(coeff1, v);
		b = evaluate_at_a_point(coeff2, v);
		if (f_vv) {
			cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection "
					"point " << rk << " / " << N_points << " :";
			Int_vec_print(cout, v, nb_variables);
			cout << " evaluates to (" << a << "," << b << ")" << endl;
		}
		if (a == 0 && b == 0) {
			Pts.push_back(rk);
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection "
				"done" << endl;
	}
}



void homogeneous_polynomial_domain::enumerate_points_lint(
		int *coeff,
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint" << endl;
	}

	vector<long int> Points;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint "
				"before enumerate_points" << endl;
	}
	enumerate_points(coeff, Points, verbose_level - 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint "
				"after enumerate_points" << endl;
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_lint "
				"The object has " << Points.size() << " points" << endl;
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

void homogeneous_polynomial_domain::enumerate_points_in_intersection_lint(
		int *coeff1, int *coeff2,
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection_lint" << endl;
	}

	vector<long int> Points;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection_lint "
				"before enumerate_points" << endl;
	}
	enumerate_points_in_intersection(coeff1, coeff2, Points, verbose_level - 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection_lint "
				"after enumerate_points" << endl;
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection_lint "
				"The object has " << Points.size() << " points" << endl;
	}
	int i;

	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points_in_intersection_lint done" << endl;
	}
}



void homogeneous_polynomial_domain::enumerate_points_zariski_open_set(
		int *coeff,
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
	geometry::other_geometry::geometry_global Gg;

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
		b = F->Linear_algebra->evaluate_monomial(
				Monomials + i * nb_variables,
				pt_vec,
				nb_variables);
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

	substitute_semilinear(
			coeff_in, coeff_out,
		false /* f_semilinear */, 0 /* frob_power */,
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
	geometry::other_geometry::geometry_global Gg;

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
	algebra::number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;


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
			Gg.AG_element_unrank(
					2 /* q */, my_affine, 1, degree, rk);
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


int homogeneous_polynomial_domain::is_zero(
		int *coeff)
{
	int i;
	
	for (i = 0; i < nb_monomials; i++) {
		if (coeff[i]) {
			return false;
		}
	}
	return true;
}

void homogeneous_polynomial_domain::unrank_point(
		int *v, long int rk)
{
	//P->unrank_point(v, rk);
	F->Projective_space_basic->PG_element_unrank_modified_lint(
			v, 1, nb_variables, rk);
}

long int homogeneous_polynomial_domain::rank_point(
		int *v)
{
	long int rk;

	//rk = P->rank_point(v);
	F->Projective_space_basic->PG_element_rank_modified_lint(
			v, 1, nb_variables, rk);
	return rk;
}

void homogeneous_polynomial_domain::unrank_coeff_vector(
		int *v, long int rk)
{
	F->Projective_space_basic->PG_element_unrank_modified_lint(
			v, 1, nb_monomials, rk);
}

long int homogeneous_polynomial_domain::rank_coeff_vector(
		int *v)
{
	long int rk;

	F->Projective_space_basic->PG_element_rank_modified_lint(
			v, 1, nb_monomials, rk);
	return rk;
}

int homogeneous_polynomial_domain::test_weierstrass_form(
		int rk,
	int &a1, int &a2, int &a3, int &a4, int &a6,
	int verbose_level)
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
		return false;
	}
	F->Projective_space_basic->PG_element_normalize_from_front(
			coeff2, 1, nb_monomials);
	if (coeff2[0] != 1) {
		return false;
	}
	m_one = F->negate(1);
	if (coeff2[7] != m_one) {
		return false;
	}
	a1 = F->negate(coeff2[4]);
	a2 = coeff2[2];
	a3 = F->negate(coeff2[8]);
	a4 = coeff2[5];
	a6 = coeff2[9];
	return true;
}

int homogeneous_polynomial_domain::test_potential_algebraic_degree(
		int *eqn, int eqn_size,
		long int *Pts, int nb_pts,
		int d,
		other::data_structures::int_matrix *Subspace_wgr,
		int *eqn_reduced,
		int *eqn_kernel,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree d = " << d << endl;
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"the input set has size " << nb_pts << endl;

		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"the input set is: " << endl;
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;

		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"the input equation is: " << endl;
		Int_vec_print(cout, eqn, eqn_size);
		cout << endl;

		print_equation(cout, eqn);
		cout << endl;

	}

	other::data_structures::int_matrix *Ideal;
	int rk;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"before vanishing_ideal" << endl;
	}
	vanishing_ideal(
			Pts, nb_pts,
			rk, Ideal,
			verbose_level - 1);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"after vanishing_ideal" << endl;
	}
	int r1, r2, ret = false;

	r1 = Ideal->m;
	r2 = Subspace_wgr->m;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"d = " << d << " r1=" << r1 << " r2=" << r2 << endl;
	}

	if (Ideal->n != eqn_size) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"Ideal->n != eqn_size" << endl;
		exit(1);
	}

	algebra::linear_algebra::linear_algebra Linear_algebra;

	Linear_algebra.init(
			F, verbose_level);

	other::data_structures::int_matrix *Intersection;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree Ideal=" << endl;
		Ideal->print();
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree Subspace_wgr=" << endl;
		Subspace_wgr->print();
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"before Linear_algebra.subspace_intersection" << endl;
	}
	Linear_algebra.subspace_intersection(
			Ideal,
			Subspace_wgr,
			Intersection,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"after Linear_algebra.subspace_intersection" << endl;
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree Intersection=" << endl;
		Intersection->print();
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
				"dimension of intersection = " << Intersection->m << endl;
	}

	if (Intersection->m == 0) {
		if (f_v) {
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
					"dimension of intersection is zero" << endl;
		}
		ret = false;
	}
	else {

		int *Basis_UV;
		int *base_cols;
		int n;
		int k1, k2;

		n = Intersection->n;
		k1 = Intersection->m;
		k2 = Ideal->m ;


		if (f_v) {
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
					"before Linear_algebra.extend_basis_of_subspace" << endl;
		}
		Linear_algebra.extend_basis_of_subspace(
				n, k1, Intersection->M, k2, Ideal->M,
				Basis_UV,
				base_cols,
				verbose_level);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
					"after Linear_algebra.extend_basis_of_subspace" << endl;
		}
		if (f_v) {
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
					"Basis_UV=" << endl;
			Int_matrix_print(Basis_UV, k2, n);
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
					"base_cols=";
			Int_vec_print(cout, base_cols, k2);
			cout << endl;
		}

		int *eqn2;
		int *eqn3;
		int *coeffs;

		eqn2 = NEW_int(n);
		eqn3 = NEW_int(n);
		coeffs = NEW_int(k2);

		if (f_v) {
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree eqn=";
			Int_vec_print(cout, eqn, n);
			cout << endl;
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree n = " << n << endl;
		}

		Int_vec_copy(eqn, eqn2, n);

		int i, a, b, col;

		for (i = 0; i < k2; i++) {

			col = base_cols[i];
			if (f_v) {
				cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree i=" << i << ", cleaning column " << col << endl;
			}

			a = eqn2[col];
			coeffs[i] = a;
			b = F->negate(a);

			Linear_algebra.linear_combination_of_vectors(
					1, eqn2, b, Basis_UV + i * n, eqn3, n);

			Int_vec_copy(eqn3, eqn2, n);
			if (f_v) {
				cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree eqn=";
				Int_vec_print(cout, eqn2, n);
				cout << endl;
			}

		}

		if (!Int_vec_is_zero(eqn2, n)) {
			cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
					"eqn does not reduce to zero" << endl;
			ret = false;
		}
		else {
			if (f_v) {
				cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
						"Basis_UV=" << endl;
				Int_matrix_print(Basis_UV, k2, n);
				cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
						"coeffs=";
				Int_vec_print(cout, coeffs, k2);
				cout << endl;
			}

			if (Int_vec_is_zero(coeffs, k1)) {
				if (f_v) {
					cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
							"coeffs[k1] is zero" << endl;
				}
				ret = false;
			}
			else {
				Linear_algebra.mult_vector_from_the_left(
						coeffs, Basis_UV,
						eqn_reduced, k1, n);

				Linear_algebra.mult_vector_from_the_left(
						coeffs + k1, Basis_UV + k1 * n,
						eqn_kernel, k2 - k1, n);

				if (f_v) {
					cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
							"eqn        =";
					Int_vec_print(cout, eqn, n);
					cout << endl;

					print_equation(cout, eqn);
					cout << endl;


					cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
							"eqn_reduced=";
					Int_vec_print(cout, eqn_reduced, n);
					cout << endl;

					print_equation(cout, eqn_reduced);
					cout << endl;


					cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree "
							"eqn_kernel =";
					Int_vec_print(cout, eqn_kernel, n);
					cout << endl;

					print_equation(cout, eqn_kernel);
					cout << endl;

				}
				ret = true;
			}
		}


		FREE_int(eqn2);
		FREE_int(eqn3);
		FREE_int(coeffs);
		FREE_int(Basis_UV);
		FREE_int(base_cols);
	}

	FREE_OBJECT(Ideal);
	FREE_OBJECT(Intersection);



	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_potential_algebraic_degree done" << endl;
	}

	return ret;
}


int homogeneous_polynomial_domain::dimension_of_ideal(
		long int *Pts,
		int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::dimension_of_ideal" << endl;
	}
	//int *Kernel;
	//int r, rk;

	//Kernel = NEW_int(get_nb_monomials() * get_nb_monomials());


	if (f_v) {
		cout << "homogeneous_polynomial_domain::dimension_of_ideal "
				"the input set has size " << nb_pts << endl;
		cout << "homogeneous_polynomial_domain::dimension_of_ideal "
				"the input set is: " << endl;
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		//P->print_set_numerical(cout, GOC->Pts, GOC->nb_pts);
	}

	other::data_structures::int_matrix *Kernel;
	int rk, r;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::dimension_of_ideal "
				"before vanishing_ideal" << endl;
	}
	vanishing_ideal(
			Pts, nb_pts,
			rk, Kernel,
			verbose_level - 1);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::dimension_of_ideal "
				"after vanishing_ideal" << endl;
	}

	//r = get_nb_monomials() - rk;
	r = Kernel->m;

	FREE_OBJECT(Kernel);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::dimension_of_ideal done" << endl;
	}

	return r;
}

void homogeneous_polynomial_domain::explore_vanishing_ideal(
		long int *Pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal" << endl;
	}

	//int *Kernel;
	//int r, rk;

	//Kernel = NEW_int(get_nb_monomials() * get_nb_monomials());


	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal "
				"the input set has size " << nb_pts << endl;
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal "
				"the input set is: " << endl;
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		//P->print_set_numerical(cout, GOC->Pts, GOC->nb_pts);
	}

	other::data_structures::int_matrix *Kernel;
	int rk, r;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal "
				"before vanishing_ideal" << endl;
	}
	vanishing_ideal(
			Pts, nb_pts,
			rk,
			Kernel,
			verbose_level - 1);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal "
				"after vanishing_ideal" << endl;
	}

	int h;
	int nb_pts2;
	long int *Pts2;

	r = get_nb_monomials() - rk;

	for (h = 0; h < r; h++) {
		cout << "generator " << h << " / " << r << " is ";
		print_equation_relaxed(cout, Kernel->M + h * get_nb_monomials());
		cout << endl;

	}


	cout << "looping over all generators of the ideal:" << endl;
	for (h = 0; h < r; h++) {
		cout << "generator " << h << " / " << r << " is ";
		Int_vec_print(cout, Kernel->M + h * get_nb_monomials(), get_nb_monomials());
		cout << " : " << endl;

#if 0
		vector<long int> Points;
		int i;

		enumerate_points(Kernel + h * get_nb_monomials(),
				Points, verbose_level);
		nb_pts2 = Points.size();


		Pts2 = NEW_lint(nb_pts2);
		for (i = 0; i < nb_pts2; i++) {
			Pts2[i] = Points[i];
		}
#endif
		enumerate_points_lint(Kernel->M + h * get_nb_monomials(), Pts2, nb_pts2, verbose_level);


		cout << "We found " << nb_pts2 << " points "
				"on the generator of the ideal" << endl;
		cout << "They are : ";
		Lint_vec_print(cout, Pts2, nb_pts2);
		cout << endl;
		//P->print_set_numerical(cout, Pts, nb_pts);

		FREE_lint(Pts2);

	} // next h

	FREE_OBJECT(Kernel);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::explore_vanishing_ideal done" << endl;
	}

}


void homogeneous_polynomial_domain::evaluate_point_on_all_monomials(
		int *pt_coords,
		int *evaluation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::evaluate_point_on_all_monomials" << endl;
	}
	for (j = 0; j < nb_monomials; j++) {
		evaluation[j] =
				F->Linear_algebra->evaluate_monomial(
						Monomials + j * nb_variables,
						pt_coords, nb_variables);
	}
}


void homogeneous_polynomial_domain::make_system(
		int *Pt_coords, int nb_pts,
		int *&System, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_system" << endl;
	}

	nb_cols = nb_variables;

	System = NEW_int(nb_pts * nb_monomials);

	for (i = 0; i < nb_pts; i++) {

		evaluate_point_on_all_monomials(
				Pt_coords + i * nb_variables,
				System + i * nb_monomials,
				verbose_level - 2);

#if 0
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] =
					F->Linear_algebra->evaluate_monomial(
							Monomials + j * nb_variables, Pt_coords + i * nb_variables, nb_variables);
		}
#endif

	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_system done" << endl;
	}
}

void homogeneous_polynomial_domain::vanishing_ideal(
		long int *Pts, int nb_pts, int &rank,
		other::data_structures::int_matrix *&Kernel,
		int verbose_level)
// Kernel[(nb_monomials - r) * nb_monomials]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int *System;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal" << endl;
	}

	System = NEW_int(MAX(nb_pts, nb_monomials) * nb_monomials);
	for (i = 0; i < nb_pts; i++) {

		unrank_point(v, Pts[i]);


		evaluate_point_on_all_monomials(
				v,
				System + i * nb_monomials,
				verbose_level - 2);

#if 0
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] =
					F->Linear_algebra->evaluate_monomial(
							Monomials + j * nb_variables, v, nb_variables);
		}
#endif

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
	rank = F->Linear_algebra->RREF_and_kernel(
			nb_monomials,
			nb_pts, System, 0 /* verbose_level */);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system has rank " << rank << endl;
	}
	if (f_vv) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system in RREF:" << endl;
		Int_matrix_print(
				System, rank, nb_monomials);
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The kernel:" << endl;
		Int_matrix_print(
				System + rank * nb_monomials,
				nb_monomials - rank, nb_monomials);
	}
#if 0
	Int_vec_copy(
			System + rank * nb_monomials, Kernel,
			(nb_monomials - r) * nb_monomials);
#endif


	Kernel = NEW_OBJECT(other::data_structures::int_matrix);

	Kernel->allocate_and_init(
			nb_monomials - rank /* m */, nb_monomials /* n */,
			System + rank * nb_monomials);

	FREE_int(System);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal done" << endl;
	}
}

void homogeneous_polynomial_domain::subspace_with_good_reduction(
		int degree, int modulus,
		other::data_structures::int_matrix *&Subspace_wgr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction" << endl;
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction "
				"degree = " << degree
				<< " modulus=" << modulus << endl;
	}
	int rk, j;
	int *basis;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction "
				"before allocating basis" << endl;
	}
	basis = NEW_int(nb_monomials);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction "
				"before computing basis" << endl;
	}
	rk = 0;
	for (j = 0; j < nb_monomials; j++) {
		if (monomial_has_good_reduction(
				j, degree, modulus,
				0 /*verbose_level - 3*/)) {
			basis[rk++] = j;
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction "
				"rk = " << rk << endl;
	}

	Subspace_wgr = NEW_OBJECT(other::data_structures::int_matrix);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction "
				"before Subspace_wgr->allocate_and_initialize_with_zero" << endl;
	}
	Subspace_wgr->allocate_and_initialize_with_zero(
			rk, nb_monomials);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction "
				"after Subspace_wgr->allocate_and_initialize_with_zero" << endl;
	}

	int i;

	for (i = 0; i < rk; i++) {
		Subspace_wgr->M[i * nb_monomials + basis[i]] = 1;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction "
				"degree = " << degree
				<< " modulus=" << modulus
				<< " rk=" << rk << endl;
	}

	FREE_int(basis);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::subspace_with_good_reduction done" << endl;
	}
}

int homogeneous_polynomial_domain::monomial_has_good_reduction(
		int mon_idx, int degree, int modulus,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::monomial_has_good_reduction, "
				"modulus = " << modulus << endl;
	}

	int *reduced_monomial;
	int i, s;
	int ret = false;

	reduced_monomial = NEW_int(nb_variables);


	monomial_reduction(mon_idx, modulus, reduced_monomial, verbose_level - 1);

	s = 0;
	for (i = 0; i < nb_variables; i++) {
		s += reduced_monomial[i];
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::monomial_has_good_reduction, "
				"s = " << s << endl;
	}
	if (s == degree) {
		ret = true;
	}
	else if (s < degree) {

		// we reduced it down too much.
		// Test if it could reduce to degree:

		if (((degree - s) % modulus) == 0) {
			ret = true;
		}
	}

	FREE_int(reduced_monomial);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::monomial_has_good_reduction done" << endl;
	}
	return ret;
}

void homogeneous_polynomial_domain::monomial_reduction(
		int mon_idx, int modulus,
		int *reduced_monomial,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::monomial_reduction, "
				"modulus = " << modulus << endl;
	}
	int *mon;
	int i, a, b;

	Int_vec_zero(reduced_monomial, nb_variables);

	mon = Monomials + mon_idx * nb_variables;
	if (f_v) {
		cout << "homogeneous_polynomial_domain::monomial_reduction, "
				"monomial = ";
		Int_vec_print(cout, mon, nb_variables);
		cout << endl;
	}
	for (i = 0; i < nb_variables; i++) {
		a = mon[i];
		if (a > modulus) {
			b = a % modulus;
			if (b == 0) {
				b = modulus;
			}
		}
		else {
			b = a;
		}
		reduced_monomial[i] = b;
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::monomial_reduction done" << endl;
	}
}

void homogeneous_polynomial_domain::equation_reduce(
		int modulus,
		homogeneous_polynomial_domain *HPD,
		int *eqn_in,
		int *eqn_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::equation_reduce" << endl;
	}

	int *reduced_monomial;
	int mon_idx;
	int mon_idx_d;
	int nb_mon_d;

	reduced_monomial = NEW_int(nb_variables);
	nb_mon_d = HPD->get_nb_monomials();
	Int_vec_zero(eqn_out, nb_mon_d);

	for (mon_idx = 0; mon_idx < nb_monomials; mon_idx++) {

		if (f_v) {
			cout << "homogeneous_polynomial_domain::equation_reduce "
					"mon_idx = " << mon_idx << " / " << nb_monomials << endl;
		}

		if (eqn_in[mon_idx] == 0) {
			continue;
		}

#if 0
		int *mon;
		mon = Monomials + mon_idx * nb_variables;
		if (f_v) {
			cout << "homogeneous_polynomial_domain::monomial_reduction, monomial = ";
			Int_vec_print(cout, mon, nb_variables);
			cout << endl;
		}
#endif

		monomial_reduction(
				mon_idx, modulus,
				reduced_monomial,
				verbose_level - 2);

		mon_idx_d = HPD->index_of_monomial(reduced_monomial);

		eqn_out[mon_idx_d] = F->add(eqn_out[mon_idx_d], eqn_in[mon_idx]);
	}

	FREE_int(reduced_monomial);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::equation_reduce done" << endl;
	}
}

int homogeneous_polynomial_domain::compare_monomials(
		int *M1, int *M2)
{
	other::data_structures::sorting Sorting;

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

int homogeneous_polynomial_domain::compare_monomials_PART(
		int *M1, int *M2)
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


void homogeneous_polynomial_domain::print_monomial_ordering_latex(
		std::ostream &ost)
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

int *homogeneous_polynomial_domain::read_from_string_coefficient_pairs(
		std::string &str, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::read_from_string_coefficient_pairs" << endl;
	}

	int *coeff;
	algebra::number_theory::number_theory_domain NT;

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

int *homogeneous_polynomial_domain::read_from_string_coefficient_vector(
		std::string &str, int verbose_level)
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
	F->Io->read_from_string_coefficient_vector(str,
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
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::number_of_conditions_satisfied" << endl;
	}


	if (f_v) {
		cout << "Reading file " << number_of_conditions_satisfied_fname << " of size "
				<< Fio.file_size(number_of_conditions_satisfied_fname) << endl;
	}
	Fio.read_set_from_file(
			number_of_conditions_satisfied_fname, Pts, nb_pts,
			verbose_level);

	int *Cnt;

	Cnt = NEW_int(nb_pts);
	Int_vec_zero(Cnt, nb_pts);


	algebra::number_theory::number_theory_domain NT;
	int h, i, a;
	long int rk;
	int *v;

	v = NEW_int(get_nb_variables());


	label_txt = variety_label_txt;
	label_tex = variety_label_tex;



	for (h = 0; h < Variety_coeffs.size(); h++) {

		if (f_v) {
			cout << "homogeneous_polynomial_domain::number_of_conditions_satisfied "
					"h=" << h << " / " << Variety_coeffs.size() << " : ";
			cout << Variety_coeffs[h] << endl;
		}

		int *coeff;

		coeff = read_from_string_coefficient_pairs(
				Variety_coeffs[h], verbose_level - 2);

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


	other::data_structures::tally T;

	T.init(Cnt, nb_pts, false, 0);

	cout << "Number of conditions satisfied:" << endl;
	T.print_bare(true);
	cout << endl;

	//T.save_classes_individually(fname);

	int f, l, t, j, pos;

	// go through classes in reverse order:
	for (i = T.nb_types - 1; i >= 0; i--) {

		f = T.type_first[i];
		l = T.type_len[i];
		t = T.data_sorted[f];


		string fname2;

		fname2 = number_of_conditions_satisfied_fname
				+ std::to_string(t) + ".csv";



		long int *the_class;

		the_class = NEW_lint(l);
		for (j = 0; j < l; j++) {
			pos = T.sorting_perm_inv[f + j];
			the_class[j] = Pts[pos];
		}
		string label;

		label.assign("case");

		Fio.Csv_file_support->lint_vec_write_csv(
				the_class, l, fname2, label);

		cout << "class of type " << t
				<< " contains " << l << " elements:" << endl;
		F->Io->display_table_of_projective_points(
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
	algebra::number_theory::number_theory_domain NT;
	int h;
	long int *Pts1;
	int sz1;
	long int *Pts2;
	int sz2;
	other::data_structures::sorting Sorting;
	long int N_points;
	geometry::other_geometry::geometry_global Gg;

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

		coeff = read_from_string_coefficient_pairs(
				Variety_coeffs[h], verbose_level - 2);
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

		enumerate_points_zariski_open_set(
				coeff, Points, verbose_level);

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
				if (Sorting.lint_vec_search(
						Pts1, sz1, a, idx, 0)) {
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

	F->Io->display_table_of_projective_points(
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
		int *coeff, int sz,
		//std::string &variety_coeffs,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety" << endl;
	}

	algebra::number_theory::number_theory_domain NT;
	long int N_points;
	geometry::other_geometry::geometry_global Gg;

	N_points = Gg.nb_PG_elements(nb_variables - 1, q);


	label_txt = variety_label;
	label_tex = variety_label_tex;

	//int *coeff;
	//int sz;

	//Get_int_vector_from_label(variety_coeffs, coeff, sz, verbose_level);

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
		print_equation_tex(cout, coeff);
		cout << endl;

		print_equation_relaxed(cout, coeff);
		cout << endl;


	}

	Pts = NEW_lint(N_points);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety "
				"before enumerate_points_lint" << endl;
	}

#if 0
	vector<long int> Points;
	int i;

	enumerate_points(coeff, Points, verbose_level);



	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}
#endif

	enumerate_points_lint(coeff, Pts, nb_pts, verbose_level);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_variety "
				"after enumerate_points_lint, "
				"nb_pts = " << nb_pts << endl;
	}


	F->Io->display_table_of_projective_points(
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
		int &dim_kernel, int &nb_monomials,
		other::data_structures::int_matrix *&Kernel,
		//int *&Kernel,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_ideal" << endl;
	}

	algebra::number_theory::number_theory_domain NT;


	nb_monomials = get_nb_monomials();


	long int *Pts;
	int nb_pts;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_ideal "
				"ideal_point_set_label=" << ideal_point_set_label << endl;
	}

	Get_lint_vector_from_label(ideal_point_set_label, Pts, nb_pts, verbose_level);

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

	//Kernel = NEW_int(nb_monomials * nb_monomials);

	int r;

	vanishing_ideal(
			Pts, nb_pts, r,
			Kernel,
			verbose_level - 3);


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
		cout << "homogeneous_polynomial_domain::create_projective_curve "
				"number of variables must be 3" << endl;
		exit(1);
	}

	coeff = NEW_int(get_nb_monomials());
	Int_vec_zero(coeff, get_nb_monomials());

	label_txt.assign(variety_label_txt);
	label_tex.assign(variety_label_tex);
	int *coeffs;
	int len;
	long int i, j, a, b, c, s, t;
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
		F->Projective_space_basic->PG_element_unrank_modified(
				v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		for (j = 0; j < get_nb_variables(); j++) {
			a = get_monomial(j, 0);
			b = get_monomial(j, 1);
			v[j] = F->mult3(coeffs[j], F->power(s, a), F->power(t, b));
		}
		F->Projective_space_basic->PG_element_rank_modified(
				v, 1, get_nb_variables(), c);
		Pts[i] = c;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, get_nb_variables());
			cout << " : " << setw(5) << c << endl;
		}
	}

	F->Io->display_table_of_projective_points(
			cout, Pts, nb_pts, get_nb_variables());


	FREE_int(v);
	FREE_int(coeffs);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::create_projective_curve done" << endl;
	}
}


void homogeneous_polynomial_domain::get_coefficient_vector(
		algebra::expression_parser::formula *Formula,
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
		Formula->print(cout);
	}


	algebra::expression_parser::syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(
			this, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"after Formula->get_subtrees" << endl;
	}

	if (Formula->f_Sajeeb) {

		if (f_v) {
			cout << "homogeneous_polynomial_domain::get_coefficient_vector "
					"before Formula->evaluate" << endl;
		}
		Formula->evaluate(
				this,
				Subtrees, evaluate_text, Coefficient_vector,
				verbose_level);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::get_coefficient_vector "
					"after Formula->evaluate" << endl;
		}


	}
	else {

		int i;


		for (i = 0; i < nb_monomials; i++) {
			cout << "homogeneous_polynomial_domain::get_coefficient_vector "
					"Monomial " << i << " : ";
			if (Subtrees[i]) {
				Subtrees[i]->print_expression(cout);
				cout << " * ";
				print_monomial(cout, i);
				cout << endl;
			}
			else {
				cout << "homogeneous_polynomial_domain::get_coefficient_vector "
						"no subtree" << endl;
			}
		}


		//int *Coefficient_vector;

		//Coefficient_vector = NEW_int(nb_monomials);

		if (f_v) {
			cout << "homogeneous_polynomial_domain::get_coefficient_vector "
					"before Formula->evaluate" << endl;
		}
		Formula->evaluate(
				this,
				Subtrees, evaluate_text, Coefficient_vector,
				verbose_level);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::get_coefficient_vector "
					"after Formula->evaluate" << endl;
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::get_coefficient_vector "
				"coefficient vector:" << endl;
		Int_vec_print(cout, Coefficient_vector, homogeneous_polynomial_domain::nb_monomials);
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
		geometry::projective_geometry::projective_space *P,
		long int *&Image_pts, int &N_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::evaluate_regular_map" << endl;
	}

	if (nb_eqns != P->Subspaces->n + 1) {
		cout << "homogeneous_polynomial_domain::evaluate_regular_map "
				"nb_eqns != P->n + 1" << endl;
		exit(1);
	}

	int *v;
	int *w;
	int h;
	long int i, j;
	int f_vv = false;

	N_points = P->Subspaces->N_points;
	Image_pts = NEW_lint(N_points);
	v = NEW_int(P->Subspaces->n + 1);
	w = NEW_int(P->Subspaces->n + 1);

	for (i = 0; i < N_points; i++) {

		P->unrank_point(v, i);

		if (f_vv) {
			cout << "homogeneous_polynomial_domain::evaluate_regular_map "
					"point " << i << " is ";
			Int_vec_print(cout, v, P->Subspaces->n + 1);
			cout << endl;
		}

		for (h = 0; h < P->Subspaces->n + 1; h++) {
			w[h] = evaluate_at_a_point(
					Coefficient_vector + h * nb_monomials, v);
		}


		if (!Int_vec_is_zero(w, P->Subspaces->n + 1)) {
			j = P->rank_point(w);
		}
		else {
			j = -1;
		}

		if (f_vv) {
			cout << "homogeneous_polynomial_domain::evaluate_regular_map maps to ";
			Int_vec_print(cout, w, P->Subspaces->n + 1);
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

std::string homogeneous_polynomial_domain::stringify(
		int *eqn)
{
	string output;

	output = Int_vec_stringify(eqn, get_nb_monomials());
	return output;
}


std::string homogeneous_polynomial_domain::stringify_algebraic_notation(
		int *eqn)
{
	string output;

	//output = Int_vec_stringify(eqn, get_nb_monomials());


	int i, c;
	int f_first = true;


	for (i = 0; i < nb_monomials; i++) {
		c = eqn[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
		}
		else {
			output += " + ";
		}
		if (c > 1) {
			output += std::to_string(c) + "*";
			//F->Io->print_element(ost, c);
			//ost << c;
		}

		output += stringify_monomial(i);
		//print_monomial_latex(ost, i);
	}

	if (f_first) {
		output = "0";
	}


	return output;
}


void homogeneous_polynomial_domain::parse_equation_wo_parameters(
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		std::string &equation_text,
		int *&eqn, int &eqn_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters" << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"name_of_formula=" << name_of_formula << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"name_of_formula_tex=" << name_of_formula_tex << endl;
		//cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
		//		"managed_variables=" << managed_variables << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"equation_text=" << equation_text << endl;
	}


	// create a symbolic object containing the general formula:

	algebra::expression_parser::symbolic_object_builder_description *Descr1;


	Descr1 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder_description);
	Descr1->f_field_pointer = true;
	Descr1->field_pointer = F;
	Descr1->f_text = true;
	Descr1->text_txt = equation_text;




	algebra::expression_parser::symbolic_object_builder *SB;

	SB = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder);



	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"before SB->init" << endl;
	}

	string s1;

	s1 = name_of_formula + "_raw";

	SB->init(Descr1, s1, verbose_level);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"after SB->init" << endl;
	}



	// Perform simplification

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"before SB->Formula_vector->V[0].simplify" << endl;
	}
	SB->Formula_vector->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"after SB->Formula_vector->V[0].simplify" << endl;
	}

	// Perform expansion.
	// The result will be in the temporary object Formula_vector_after_expand


	algebra::expression_parser::formula_vector *Formula_vector_after_expand;

	Formula_vector_after_expand = NEW_OBJECT(algebra::expression_parser::formula_vector);

	int f_write_trees_during_expand = false;

	std::string managed_variables;

	managed_variables = list_of_variables();


	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"managed_variables=" << managed_variables << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"before Formula_vector->expand" << endl;
	}
	Formula_vector_after_expand->expand(
			SB->Formula_vector,
			F,
			name_of_formula, name_of_formula_tex,
			true /*f_has_managed_variables*/,
			managed_variables,
			f_write_trees_during_expand,
			verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"after Formula_vector->expand" << endl;
	}

	// Perform simplification



	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"before Formula_vector_after_expand->V[0].simplify" << endl;
	}
	Formula_vector_after_expand->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"after Formula_vector_after_expand->V[0].simplify" << endl;
	}


	// collect the coefficients of the monomials:


	other::data_structures::int_matrix *I;
	int *Coeff;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"before collect_monomial_terms" << endl;
	}
	Formula_vector_after_expand->V[0].collect_monomial_terms(
			I, Coeff,
			verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"after collect_monomial_terms" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"data collected:" << endl;
		int i;

		for (i = 0; i < I->m; i++) {
			cout << Coeff[i] << " : ";
			Int_vec_print(cout, I->M + i * I->n, I->n);
			cout << endl;
		}
		cout << "variables: ";
		Formula_vector_after_expand->V[0].tree->print_variables_in_line(cout);
		cout << endl;
	}

#if 1
	if (I->n != nb_variables) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"I->n != nb_variables" << endl;
		exit(1);
	}
#endif

	eqn = NEW_int(nb_monomials);
	eqn_size = nb_monomials;
	Int_vec_zero(eqn, nb_monomials);

	int i, idx;

	for (i = 0; i < I->m; i++) {
		idx = index_of_monomial(I->M + i * I->n);
		eqn[idx] = F->add(eqn[idx], Coeff[i]);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters "
				"equation: ";
		Int_vec_print(cout, eqn, eqn_size);
		cout << endl;
	}

	FREE_OBJECT(I);
	FREE_int(Coeff);
	FREE_OBJECT(Formula_vector_after_expand);
	//FREE_OBJECT(Formula_vector_after_sub);
	//FREE_OBJECT(SB1);
	//FREE_OBJECT(SB2);
	FREE_OBJECT(Descr1);
	//FREE_OBJECT(Descr2);





	FREE_OBJECT(SB);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_wo_parameters done" << endl;
	}
}


void homogeneous_polynomial_domain::parse_equation_and_substitute_parameters(
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameter_values,
		int *&eqn, int &eqn_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters" << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"name_of_formula=" << name_of_formula << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"name_of_formula_tex=" << name_of_formula_tex << endl;
		//cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
		//		"managed_variables=" << managed_variables << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"equation_text=" << equation_text << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"equation_parameters=" << equation_parameters << endl;
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"equation_parameter_values=" << equation_parameter_values << endl;
	}


	// create a symbolic object containing the general formula:

	algebra::expression_parser::symbolic_object_builder_description *Descr1;


	Descr1 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder_description);
	Descr1->f_field_pointer = true;
	Descr1->field_pointer = F;
	Descr1->f_text = true;
	Descr1->text_txt = equation_text;




	algebra::expression_parser::symbolic_object_builder *SB1;

	SB1 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder);



	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"before SB1->init" << endl;
	}

	string s1;

	s1 = name_of_formula + "_raw";

	SB1->init(Descr1, s1, verbose_level);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"after SB1->init" << endl;
	}



	// create a second symbolic object containing the specific values
	// to be substituted.

	algebra::expression_parser::symbolic_object_builder_description *Descr2;


	Descr2 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder_description);
	Descr2->f_field_pointer = true;
	Descr2->field_pointer = F;
	Descr2->f_text = true;
	Descr2->text_txt = equation_parameter_values;



	algebra::expression_parser::symbolic_object_builder *SB2;

	SB2 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder);

	string s2;

	s2 = name_of_formula + "_param_values";


	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"before SB2->init" << endl;
	}

	SB2->init(Descr2, s2, verbose_level);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"after SB2->init" << endl;
	}


	// Perform the substitution.
	// Create temporary object Formula_vector_after_sub

	algebra::expression_parser::symbolic_object_builder *O_target = SB1;
	algebra::expression_parser::symbolic_object_builder *O_source = SB2;

	//O_target = Get_symbol(Descr->substitute_target);
	//O_source = Get_symbol(Descr->substitute_source);


	algebra::expression_parser::formula_vector *Formula_vector_after_sub;


	Formula_vector_after_sub = NEW_OBJECT(algebra::expression_parser::formula_vector);


	std::string managed_variables;
	int f_has_managed_variables = false;

	int i;

#if 0
	for (i = 0; i < nb_variables; i++) {
		managed_variables += symbols[i];
		if (i < nb_variables - 1) {
			managed_variables += ",";
		}
	}
#endif
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"managed_variables = " << managed_variables << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"before Formula_vector_after_sub->substitute" << endl;
	}
	Formula_vector_after_sub->substitute(
			O_source->Formula_vector,
			O_target->Formula_vector,
			equation_parameters /*Descr->substitute_variables*/,
			name_of_formula, name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"after Formula_vector_after_sub->substitute" << endl;
	}


	// Perform simplification

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"before Formula_vector_after_sub->V[0].simplify" << endl;
	}
	Formula_vector_after_sub->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"after Formula_vector_after_sub->V[0].simplify" << endl;
	}

	// Perform expansion.
	// The result will be in the temporary object Formula_vector_after_expand


	algebra::expression_parser::formula_vector *Formula_vector_after_expand;

	Formula_vector_after_expand = NEW_OBJECT(algebra::expression_parser::formula_vector);

	int f_write_trees_during_expand = false;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"before Formula_vector->expand" << endl;
	}
	Formula_vector_after_expand->expand(
			Formula_vector_after_sub,
			F,
			name_of_formula, name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			f_write_trees_during_expand,
			verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"after Formula_vector->expand" << endl;
	}

	// Perform simplification



	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"before Formula_vector_after_expand->V[0].simplify" << endl;
	}
	Formula_vector_after_expand->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"after Formula_vector_after_expand->V[0].simplify" << endl;
	}


	// collect the coefficients of the monomials:


	other::data_structures::int_matrix *I;
	int *Coeff;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"before collect_monomial_terms" << endl;
	}
	Formula_vector_after_expand->V[0].collect_monomial_terms(
			I, Coeff,
			verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"after collect_monomial_terms" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"data collected:" << endl;
		int i;

		for (i = 0; i < I->m; i++) {
			cout << Coeff[i] << " : ";
			Int_vec_print(cout, I->M + i * I->n, I->n);
			cout << endl;
		}
		cout << "variables: ";
		Formula_vector_after_expand->V[0].tree->print_variables_in_line(cout);
		cout << endl;
	}

#if 0
	if (I->n != 3) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"we need exactly 3 variables" << endl;
		exit(1);
	}
#endif

	eqn = NEW_int(nb_monomials);
	eqn_size = nb_monomials;
	Int_vec_zero(eqn, nb_monomials);

	int idx;
	for (i = 0; i < I->m; i++) {
		idx = index_of_monomial(I->M + i * I->n);
		eqn[idx] = F->add(eqn[idx], Coeff[i]);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters "
				"equation: ";
		Int_vec_print(cout, eqn, eqn_size);
		cout << endl;
	}

	FREE_OBJECT(I);
	FREE_int(Coeff);
	FREE_OBJECT(Formula_vector_after_expand);
	FREE_OBJECT(Formula_vector_after_sub);
	FREE_OBJECT(SB1);
	FREE_OBJECT(SB2);
	FREE_OBJECT(Descr1);
	FREE_OBJECT(Descr2);




	if (f_v) {
		cout << "homogeneous_polynomial_domain::parse_equation_and_substitute_parameters done" << endl;
	}
}


void homogeneous_polynomial_domain::compute_singular_points_projectively(
		geometry::projective_geometry::projective_space *P,
		int *equation,
		std::vector<long int> &Singular_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_singular_points_projectively" << endl;
	}

	homogeneous_polynomial_domain *Poly_reduced_degree;
	int *gradient;

	Poly_reduced_degree = NEW_OBJECT(homogeneous_polynomial_domain);


	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_singular_points_projectively "
				"before Poly_reduced_degree->init" << endl;
	}
	Poly_reduced_degree->init(
			F,
			nb_variables, degree - 1,
			Monomial_ordering_type,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_singular_points_projectively "
				"after Poly_reduced_degree->init" << endl;
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_singular_points_projectively "
				"before compute_gradient" << endl;
	}
	compute_gradient(
			Poly_reduced_degree,
			equation, gradient, verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_singular_points_projectively "
				"after compute_gradient" << endl;
	}

	long int nb_points;
	int *v;
	long int rk;
	int a;

	v = NEW_int(nb_variables);

	nb_points = P->Subspaces->N_points;

	for (rk = 0; rk < nb_points; rk++) {
		P->unrank_point(
			v, rk);

		// check if the point lies on the variety:

		a = evaluate_at_a_point(equation, v);

		if (a == 0) {

			// yes, it does.

			// check if the point is singular:
			for (i = 0; i < nb_variables; i++) {
				a = Poly_reduced_degree->evaluate_at_a_point(
						gradient + i * Poly_reduced_degree->get_nb_monomials(),
						v);
				if (a) {
					break;
				}
			}
			if (i == nb_variables) {
				Singular_points.push_back(rk);
			}
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_singular_points_projectively "
				"number of singular points = " << Singular_points.size() << endl;
	}

	FREE_int(v);
	FREE_int(gradient);

	FREE_OBJECT(Poly_reduced_degree);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_singular_points_projectively done" << endl;
	}
}

void homogeneous_polynomial_domain::compute_partials(
		homogeneous_polynomial_domain *Poly_reduced_degree,
		ring_theory::partial_derivative *&Partials,
		int verbose_level)
// Partials[nb_variables]
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_partials" << endl;
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_partials "
				"nb_variables = " << nb_variables << endl;
		cout << "homogeneous_polynomial_domain::compute_partials "
				"nb_monomials = " << get_nb_monomials() << endl;
	}

	//ring_theory::partial_derivative *Partials; // [nb_variables]

	Partials = NEW_OBJECTS(ring_theory::partial_derivative, nb_variables);


	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_partials "
				"initializing partials" << endl;
	}
	for (i = 0; i < nb_variables; i++) {
		Partials[i].init(this, Poly_reduced_degree, i, verbose_level);
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_partials "
				"initializing partials done" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_partials done" << endl;
	}

}

void homogeneous_polynomial_domain::compute_and_export_partials(
		homogeneous_polynomial_domain *Poly_reduced_degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_and_export_partials" << endl;
	}



	ring_theory::partial_derivative *Partials; // [nb_variables]


	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_and_export_partials "
				"before compute_partials" << endl;
	}
	compute_partials(Poly_reduced_degree, Partials, verbose_level - 2);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_and_export_partials "
				"after compute_partials" << endl;
	}


	for (i = 0; i < nb_variables; i++) {

		string fname_base;

		fname_base = "partial_" + std::to_string(degree)
				+ "_" + std::to_string(Poly_reduced_degree->degree)
				+ "_" + std::to_string(i);

		Partials[i].do_export(fname_base, verbose_level);
	}


	FREE_OBJECTS(Partials);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_and_export_partials done" << endl;
	}

}




void homogeneous_polynomial_domain::compute_gradient(
		homogeneous_polynomial_domain *Poly_reduced_degree,
		int *equation, int *&gradient, int verbose_level)
// gradient[nb_variables * Poly_reduced_degree->get_nb_monomials()]
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_gradient "
				"nb_variables = " << nb_variables << endl;
		cout << "homogeneous_polynomial_domain::compute_gradient "
				"nb_monomials = " << get_nb_monomials() << endl;
	}

	ring_theory::partial_derivative *Partials; // [nb_variables]


	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_gradient "
				"before compute_partials" << endl;
	}
	compute_partials(Poly_reduced_degree, Partials, verbose_level - 2);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::compute_gradient "
				"after compute_partials" << endl;
	}

	gradient = NEW_int(nb_variables * Poly_reduced_degree->get_nb_monomials());

	for (i = 0; i < nb_variables; i++) {
		if (f_v) {
			cout << "homogeneous_polynomial_domain::compute_gradient i=" << i << endl;
		}
		if (f_v) {
			cout << "homogeneous_polynomial_domain::compute_gradient "
					"eqn_in=";
			Int_vec_print(cout, equation, get_nb_monomials());
			cout << " = " << endl;
			print_equation(cout, equation);
			cout << endl;
		}
		Partials[i].apply(
				equation,
				gradient + i * Poly_reduced_degree->get_nb_monomials(),
				verbose_level - 2);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::compute_gradient "
					"partial=";
			Int_vec_print(cout,
					gradient + i * Poly_reduced_degree->get_nb_monomials(),
					Poly_reduced_degree->get_nb_monomials());
			cout << " = ";
			Poly_reduced_degree->print_equation(cout,
					gradient + i * Poly_reduced_degree->get_nb_monomials());
			cout << endl;
		}
	}

	FREE_OBJECTS(Partials);

	if (f_v) {
		cout << "surface_polynomial_domains::compute_gradient done" << endl;
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




}}}}



