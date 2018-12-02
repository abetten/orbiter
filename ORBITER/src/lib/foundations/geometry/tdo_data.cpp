// tdo_data.C
// Anton Betten
//
// started: January 30, 2007

#include "foundations.h"

tdo_data::tdo_data()
{
	types_first = NULL;
	types_len = NULL;
	only_one_type = NULL;
	nb_only_one_type = 0;
	multiple_types = NULL;
	nb_multiple_types = 0;
	types_first2 = NULL;
	D1 = NULL;
	D2 = NULL;
}

tdo_data::~tdo_data()
{
	free();
}

void tdo_data::free()
{
	if (types_first) {
		FREE_int(types_first);
		types_first = NULL;
		}
	if (types_len) {
		FREE_int(types_len);
		types_len = NULL;
		}
	if (only_one_type) {
		FREE_int(only_one_type);
		only_one_type = NULL;
		}
	if (multiple_types) {
		FREE_int(multiple_types);
		multiple_types = NULL;
		}
	if (types_first2) {
		FREE_int(types_first2);
		types_first2 = NULL;
		}
	if (D1) {
		FREE_OBJECT(D1);
		D1 = NULL;
		}
	if (D2) {
		FREE_OBJECT(D2);
		D2 = NULL;
		}
}

void tdo_data::allocate(int R)
{
	types_first = NEW_int(R + 1);
	types_len = NEW_int(R);
	only_one_type = NEW_int(R);
	multiple_types = NEW_int(R);
	types_first2 = NEW_int(R);
	D1 = NEW_OBJECT(diophant);
	D2 = NEW_OBJECT(diophant);
}

int tdo_data::solve_first_system(int verbose_level, 
	int *&line_types, int &nb_line_types, int &line_types_allocated)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, nb_sol, nb_vars;
	
	if (f_v) {
		cout << "solve_first_system" << endl;
		}
	nb_vars = D1->n;
	nb_sol = 0;
	if (D1->solve_first(0/*verbose_level - 4*/)) {
	
		while (TRUE) {
			if (nb_line_types >= line_types_allocated) {
				int new_nb_line_types = line_types_allocated + 100;
				if (f_vvv) {
					cout << "reallocating to " << new_nb_line_types << endl;
					}
				int *new_line_types = NEW_int(new_nb_line_types * nb_vars);
				for (i = 0; i < nb_line_types * nb_vars; i++) {
					new_line_types[i] = line_types[i];
					}
				FREE_int(line_types);
				line_types = new_line_types;
				line_types_allocated = new_nb_line_types;
				}

			if (f_vvv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D1->x[i];
					}
				cout << endl;
				}
			for (i = 0; i < nb_vars; i++) {
				line_types[nb_line_types * nb_vars + i] = D1->x[i];
				}
			nb_line_types++;
			nb_sol++;
			if (!D1->solve_next())
				break;
			}
		}
	if (f_vv) {
		cout << "solve_first_system: found " << nb_sol
			<< " refined types" << endl;
		}
	return nb_sol;
}

void tdo_data::solve_second_system_omit(int verbose_level,
	int *classes_len, 
	int *&line_types, int &nb_line_types, 
	int *&distributions, int &nb_distributions,
	int omit)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_sol;
	
	if (f_v) {
		cout << "tdo_data::solve_second_system_omit omit=" << omit << endl;
		}
	int s, i, r = 0, f, l, h, j, u, first, len, N, a, ii, f_bad;
	
	f = types_first2[0];
	l = 0;
	s = 0;
	for (i = 0; i < nb_multiple_types - omit; i++) {
		r = multiple_types[i];
		l += types_len[r];
		s += classes_len[r];
		}
	diophant D;
	int nb_eqns_replaced;
	int *eqns_replaced;
	int *eqn_number;
		
	if (f_v) {
		cout << r << " : " << setw(3) << classes_len[r] << " : "
			<< setw(3) << f << " : " << setw(3) << l << endl;
		cout << "nb_multiple_types=" << nb_multiple_types << endl;
		cout << "omit=" << omit << endl;
		cout << "calling D2->project f=" << f << " l=" << l << endl;
		}
	D2->project(&D, f, l, eqn_number, nb_eqns_replaced,
			eqns_replaced, verbose_level - 1);
	D.f_has_sum = TRUE;
	D.sum = s;
	if (f_vv) {
		cout << "after projection:" << endl;
		D.print();
		D.write_xml(cout, "projected");
		}
	D.nb_steps_betten = 0;
	//nb_sol = D.solve_once_mckay(verbose_level);
	//nb_sol = D.solve_all_mckay(verbose_level);
	nb_sol = D.solve_all_betten(0 /*verbose_level*/);
	if (f_v) {
		cout << "number of solutions = " << nb_sol << " in "
			<< D.nb_steps_betten << " steps" << endl;
		}
	nb_distributions = 0;
	distributions = NEW_int(nb_sol * nb_line_types);


	for (N = 0; N < nb_sol; N++) {
	
		if (f_v) {
			cout << "tdo_data::solve_second_system_omit "
					"N=" << N << " / " << nb_sol << endl;
			}
		f_bad = FALSE;
		for (j = 0; j < nb_line_types; j++) {
			distributions[nb_distributions * nb_line_types + j] = 0;
			}
		for (j = 0; j < D.n; j++) {
			D.x[j] = D._results.front()[j];
			}
		if (f_v) {
			cout << "solution " << N << ":" << endl;
			for (j = 0; j < D.n; j++) {
				cout << setw(3) << D.x[j] << " ";
				}
			cout << endl;
			}
		D._results.pop_front();
		D.multiply_A_x_to_RHS1();
		for (i = 0; i < D2->m; i++) {
			D2->RHS1[i] = 0;
			}
		for (i = 0; i < D.m; i++) {
			a = D.RHS1[i];
			ii = eqn_number[i];
			D2->RHS1[ii] = a;
			}
		if (f_vv) {
			cout << "RHS1" << endl;
			for (i = 0; i < D.m; i++) {
				cout << "equation ";
				if (D.eqn_label[i])
					cout << D.eqn_label[i];
				cout << " : " << D.RHS1[i] << endl;
				}
			}

		diophant DD;
		int nb_eqns_replaced2;
		int *eqns_replaced2;
		int *eqn_number2;
		int F, L, nb_sol2;
		F = l;
		L = 0;
		s = 0;
		for (i = nb_multiple_types - omit;
				i < nb_multiple_types; i++) {
			r = multiple_types[i];
			L += types_len[r];
			s += classes_len[r];
			}
		if (f_v) {
			cout << "tdo_data::solve_second_system_omit "
					"N=" << N << " / " << nb_sol << endl;
			cout << "calling D2->project F=" << F << " L=" << L << endl;
			}
		D2->project(&DD, F, L, eqn_number2,
				nb_eqns_replaced2, eqns_replaced2,
				verbose_level - 1);
		DD.f_has_sum = TRUE;
		DD.sum = s;
		for (i = 0; i < DD.m; i++) {
			ii = eqn_number2[i];
			a = D2->RHS1[ii];
			DD.RHS[i] -= a;
			if (DD.RHS[i] < 0) {
				f_bad = TRUE;
				break;
				}
			}
		if (f_vv) {
			cout << "after projection:" << endl;
			DD.print();
			}
		if (f_bad) {
			cout << "solutions does not extend (bad RHS in eqn "
					<< i << "), skipping" << endl;
			continue;
			}
		
		if (FALSE) {
			// here we test if a givemn solution of the projected system 
			// extends to a global solution.
			// Since out solver finds in fact all solutions, this test is 
			// too expensive, hence we do not do it any more.
			
			int nb_backtrack;
			nb_sol2 = diophant_solve_all_mckay(&DD,
					nb_backtrack, 0/*verbose_level*/);
			if (f_v) {
				cout << "N=" << N << " / " << nb_sol
						<< " number of solutions = " << nb_sol2 << endl;
				}
			if (nb_sol2 == 0) {
				cout << "solutions does not extend "
						"(no solution), skipping" << endl;
				continue;
				}
			}
		FREE_int(eqns_replaced2);
		FREE_int(eqn_number2);


		for (h = 0; h < nb_only_one_type; h++) {
			r = only_one_type[h];
			u = types_first[r];
			//cout << "only one type, r=" << r << " u=" << u << endl;
			distributions[nb_distributions * nb_line_types + u] =
					classes_len[r];
			}
		for (i = 0; i < nb_multiple_types - omit; i++) {
			r = multiple_types[i];
			F = types_first2[i];
			first = types_first[r];
			len = types_len[r];
			//cout << "multiple types, r=" << r
			//<< " first=" << first << endl;
			for (j = 0; j < len; j++) {
				a = D.x[F + j];
				distributions[nb_distributions *
							  nb_line_types + first + j] = a;
				}
			}
		nb_distributions++;
		}
	if (f_v) {
		if (nb_distributions == 0) {
			cout << "tdo_data::solve_second_system_omit "
					"system has no solution" << endl;
			}
		else {
			cout << "tdo_data::solve_second_system_omit "
					"system has " << nb_distributions
					<< " solutions" << endl;
			}
		}

	FREE_int(eqns_replaced);
	FREE_int(eqn_number);

#if 0
	for (i = 0; i < nb_multiple_types; i++) {
		diophant D;
		int nb_eqns_replaced;
		int *eqns_replaced;
		
		r = multiple_types[i];
		f = types_first2[i];
		l = types_len[r];
		if (f_v) {
			cout << r << " : " << setw(3) << classes_len[r]
				<< " : " << setw(3) << f << " : "
				<< setw(3) << l << endl;
			}
		D2->project(&D, f, l, nb_eqns_replaced,
				eqns_replaced, verbose_level);
		D.f_has_sum = TRUE;
		D.sum = classes_len[r];
		D.print();
		D.solve_first(verbose_level);
		FREE_int(eqns_replaced);
		}
#endif
}

void tdo_data::solve_second_system_with_help(int verbose_level,
	int f_use_mckay_solver, int f_once, 
	int *classes_len, int f_scale, int scaling,
	int *&line_types, int &nb_line_types, 
	int *&distributions, int &nb_distributions,
	int cnt_second_system, solution_file_data *Sol)
{
	int i;
	
	if (Sol) {
		for (i = 0; i < Sol->nb_solution_files; i++) {
			if (cnt_second_system == Sol->system_no[i]) {
				cout << "reading solutions from file "
					<< Sol->solution_file[i] << endl;
				solve_second_system_from_file(
					verbose_level, classes_len,
					f_scale, scaling, line_types, nb_line_types, 
					distributions, nb_distributions,
					Sol->solution_file[i]);
				return;
				}
			}
		}
	solve_second_system(verbose_level,
		f_use_mckay_solver, f_once, classes_len,
		f_scale, scaling, line_types, nb_line_types, 
		distributions, nb_distributions);
}

void tdo_data::solve_second_system_from_file(int verbose_level,
	int *classes_len, int f_scale, int scaling,
	int *&line_types, int &nb_line_types, 
	int *&distributions, int &nb_distributions,
	char *solution_file_name)
{
	int f_v = (verbose_level >= 1);
	int cnt, i, j, a, nb_sol, *the_solution;
	int h, r, u, f, l, first, distributions_allocated;
	int Nb_vars; //, Nb_eqns;
	
	//Nb_eqns = D2->m;
	Nb_vars = D2->n;
	the_solution = NEW_int(Nb_vars);
	{
	ifstream ff(solution_file_name);
	
	for (i = 0; TRUE; i++) {
		if (ff.eof()) {
			break;
			}
		ff >> a;
		if (a == -1)
			break;
		for (j = 1; j < Nb_vars; j++) 
			ff >> a;
		}
	}
	nb_sol = i;
	cout << "the solution file " << solution_file_name
			<< " contains " << nb_sol << " solutions" << endl;
	distributions_allocated = nb_sol;
	nb_distributions = 0;
	distributions = NEW_int(distributions_allocated * nb_line_types);
	{
	ifstream ff(solution_file_name);
	for (cnt = 0; cnt < nb_sol; cnt++) {
		for (j = 0; j < Nb_vars; j++) 
			ff >> the_solution[j];
		
		for (h = 0; h < nb_only_one_type; h++) {
			r = only_one_type[h];
			u = types_first[r];
			//cout << "only one type, r=" << r << " u=" << u << endl;
			distributions[nb_distributions * nb_line_types + u] = 
				classes_len[r];
			}
		for (i = 0; i < nb_multiple_types; i++) {
			r = multiple_types[i];
			f = types_first2[i];
			first = types_first[r];
			l = types_len[r];
			//cout << "multiple types, r=" << r
			//<< " first=" << first << endl;
			for (j = 0; j < l; j++) {
				a = the_solution[f + j];
				if (f_scale)
					a *= scaling;
				distributions[nb_distributions * nb_line_types
							  + first + j] = a;
				}
			}
		nb_distributions++;
		
		}
	
	}
	FREE_int(the_solution);
	if (f_v) {
		cout << "solve_second_system_from_file: found "
				<< nb_distributions << " distributions." << endl;
		}
}

void tdo_data::solve_second_system(int verbose_level,
	int f_use_mckay_solver, int f_once, 
	int *classes_len, int f_scale, int scaling,
	int *&line_types, int &nb_line_types, 
	int *&distributions, int &nb_distributions)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int distributions_allocated, nb_sol, a;
	int h, r, u, i, f, l, j, first, ret;
	int Nb_vars, /*Nb_eqns,*/ nb_steps = 0;
	
	if (f_v) {
		cout << "tdo_data::solve_second_system" << endl;
		cout << "f_use_mckay_solver=" << f_use_mckay_solver << endl;
		cout << "f_once=" << f_once << endl;
		}
	//Nb_eqns = D2->m;
	Nb_vars = D2->n;

	distributions_allocated = 100;
	nb_distributions = 0;
	distributions = NEW_int(distributions_allocated * nb_line_types);
		
	
	nb_sol = 0;

	if (f_use_mckay_solver) {
		ret = diophant_solve_first_mckay/*betten*/(D2,
			f_once, 0/*verbose_level - 4*/);
		}
	else {
		ret = D2->solve_first_betten(0/*verbose_level - 4*/);
		}
	if (ret) {
	
		nb_steps += D2->nb_steps_betten;
		while (TRUE) {
			if (nb_distributions && (nb_distributions % 1000) == 0) {
				cout << "solve_second_system: " << nb_distributions
					<< " distributions" << endl;
				}
			if (nb_distributions >= distributions_allocated) {
				int new_nb_distributions = distributions_allocated + 100;
				if (f_vv) {
					cout << "reallocating to "
						<< new_nb_distributions << endl;
					}
				
				int *new_distributions = NEW_int(
						new_nb_distributions * nb_line_types);
				for (i = 0; i < nb_distributions * nb_line_types; i++) {
					new_distributions[i] = distributions[i];
					}
				FREE_int(distributions);
				distributions = new_distributions;
				distributions_allocated = new_nb_distributions;
				}
			if (f_vvv) {
				cout << nb_sol << " : ";
				for (i = 0; i < Nb_vars; i++) {
					cout << " " << setw(2) << D2->x[i];
					}
				cout << endl;
				}
			

			for (h = 0; h < nb_only_one_type; h++) {
				r = only_one_type[h];
				u = types_first[r];
				//cout << "only one type, r=" << r << " u=" << u << endl;
				distributions[nb_distributions * nb_line_types + u] = 
					classes_len[r];
				}
			for (i = 0; i < nb_multiple_types; i++) {
				r = multiple_types[i];
				f = types_first2[i];
				first = types_first[r];
				l = types_len[r];
				//cout << "multiple types, r=" << r
				// << " first=" << first << endl;
				for (j = 0; j < l; j++) {
					a = D2->x[f + j];
					if (f_scale) {
						a *= scaling;
						}
					distributions[nb_distributions * nb_line_types
						+ first + j] = a;
					}
				}
			nb_distributions++;
			
			nb_sol++;
			if (f_once) {
				ret = FALSE;
				}
			else {
				if (f_use_mckay_solver) {
					ret = diophant_solve_next_mckay(D2,
							0/*verbose_level - 5*/);
					}
				else {
					ret = D2->solve_next_betten(0/*verbose_level - 5*/);
					}
				}
			if (!ret) {
				break;
				}
			nb_steps += D2->nb_steps_betten;
			}
		}
	
	nb_steps += D2->nb_steps_betten;
	if (f_v) {
		cout << "solve_second_system: found " << nb_distributions
			<< " distributions in " << nb_steps << " steps" << endl;
		}
}


