// buekenhout_metz.cpp
// 
// Anton Betten
// 12/13/2010
//
// creates Buekenhout Metz unitals in PG(2,q^2).
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace finite_geometries {



buekenhout_metz::buekenhout_metz()
{
	Record_birth();
	FQ = NULL;
	Fq = NULL;
	q = 0;
	Q = 0;

	SubS = NULL;

	f_classical = false;
	f_Uab = false;
	parameter_a = 0;
	parameter_b = 0;

	P2 = NULL;
	P3 = NULL;
	v = NULL;
	w1 = w2 = w3 = w4 = w5 = NULL;

	ovoid = NULL;
	U = NULL;
	sz = 0;
	alpha = t0 = t1 = T0 = T1 = 0;
	theta_3 = 0;
	minus_t0 = 0;
	sz_ovoid = 0;
	e1 = one_1 = one_2 = 0;

	secant_lines = NULL;
	nb_secant_lines = 0;

	tangent_lines = NULL;
	nb_tangent_lines = 0;

	Intersection_sets = NULL;
	Design_blocks = NULL;
	block = NULL;
	block_size = 0;
	idx_in_unital = NULL;
	idx_in_secants = NULL;
	tangent_line_at_point = NULL;
	point_of_tangency = NULL;
	f_is_tangent_line = NULL;
	f_is_Baer = NULL;

	nb_good_points = 0;
	good_points = NULL;
}




buekenhout_metz::~buekenhout_metz()
{
	Record_death();
	if (SubS) {
		FREE_OBJECT(SubS);
	}
	if (f_is_Baer) {
		FREE_int(f_is_Baer);
	}
	if (idx_in_unital) {
		FREE_int(idx_in_unital);
	}
	if (idx_in_secants) {
		FREE_int(idx_in_secants);
	}
	if (tangent_line_at_point) {
		FREE_int(tangent_line_at_point);
	}
	if (f_is_tangent_line) {
		FREE_int(f_is_tangent_line);
	}
	if (point_of_tangency) {
		FREE_int(point_of_tangency);
	}
	if (Intersection_sets) {
		FREE_lint(Intersection_sets);
	}
	if (Design_blocks) {
		FREE_int(Design_blocks);
	}
	if (secant_lines) {
		FREE_lint(secant_lines);
	}
	if (tangent_lines) {
		FREE_lint(tangent_lines);
	}

	if (v) {
		FREE_int(v);
		FREE_int(w1);
		FREE_int(w2);
		FREE_int(w3);
		FREE_int(w4);
		FREE_int(w5);
	}
	if (U) {
		FREE_lint(U);
	}
	if (ovoid) {
		FREE_lint(ovoid);
	}
	if (P2) {
		FREE_OBJECT(P2);
	}
	if (P3) {
		FREE_OBJECT(P3);
	}
	if (good_points) {
		FREE_int(good_points);
	}
}

void buekenhout_metz::buekenhout_metz_init(
		algebra::field_theory::finite_field *Fq,
		algebra::field_theory::finite_field *FQ,
		int f_Uab, int a, int b, 
		int f_classical, int verbose_level)
// creates P2 over FQ and P3 over Fq,
// calls FQ->subfield_embedding_2dimensional
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "buekenhout_metz::buekenhout_metz_init "
				"q=" << q << " Q=" << Q << endl;
	}
	buekenhout_metz::Fq = Fq;
	buekenhout_metz::FQ = FQ;
	buekenhout_metz::q = Fq->q;
	buekenhout_metz::Q = Fq->q;
	if (Q != q * q) {
		cout << "buekenhout_metz::buekenhout_metz_init "
				"Q != q * q" << endl;
		exit(1);
	}
	buekenhout_metz::f_Uab = f_Uab;
	buekenhout_metz::parameter_a = a;
	buekenhout_metz::parameter_b = b;
	buekenhout_metz::f_classical = f_classical;
	Q = q * q;

	if (f_v) {
		cout << "buekenhout_metz::buekenhout_metz_init "
				"f_Uab=" << f_Uab << endl;
		if (f_Uab) {
			cout << "buekenhout_metz::buekenhout_metz_init "
					"a=" << parameter_a << endl;
			cout << "buekenhout_metz::buekenhout_metz_init "
					"b=" << parameter_b << endl;
		}
		cout << "buekenhout_metz::buekenhout_metz_init "
				"f_classical=" << f_classical << endl;
	}

	P2 = NEW_OBJECT(projective_geometry::projective_space);
	P3 = NEW_OBJECT(projective_geometry::projective_space);
	

	P2->projective_space_init(2, FQ, true, verbose_level);
	P3->projective_space_init(3, Fq, true, verbose_level);


	SubS = NEW_OBJECT(algebra::field_theory::subfield_structure);

	if (f_v) {
		cout << "buekenhout_metz::buekenhout_metz_init "
				"before SubS->init" << endl;
	}
	SubS->init(
			FQ,
			Fq, verbose_level);
	if (f_v) {
		cout << "buekenhout_metz::buekenhout_metz_init "
				"after SubS->init" << endl;
	}

	
	string s;

	s.assign("\\beta");
	Fq->init_symbol_for_print(s);
	
	alpha = FQ->p;
	T0 = FQ->negate(FQ->N2(alpha));
	T1 = FQ->T2(alpha);
	
#if 0
	if (f_v) {
		cout << "buekenhout_metz::buekenhout_metz_init "
				"before FQ->subfield_embedding_2dimensional" << endl;
		}
	FQ->subfield_embedding_2dimensional(*Fq, 
		components, embedding, pair_embedding, verbose_level - 2);

		// we think of FQ as two dimensional vector space 
		// over Fq with basis (1,alpha)
		// for i,j \in Fq, with x = i + j * alpha \in FQ, we have 
		// pair_embedding[i * q + j] = x;
		// also, 
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in Fq, embedding[i] is the element 
		// in FQ that corresponds to i 
		
		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]
#endif

	e1 = SubS->embedding_2D[1];
	one_1 = SubS->components_2D[e1 * 2 + 0];
	one_2 = SubS->components_2D[e1 * 2 + 1];

	if (f_v) {
#if 0
		FQ->print_embedding(*Fq, 
			components, embedding, pair_embedding);

		cout << "embedding table:" << endl;
		FQ->print_embedding_tex(*Fq, 
			components, embedding, pair_embedding);

#endif

		cout << "buekenhout_metz::buekenhout_metz_init "
				"e1=" << e1 << " one_1=" << one_1
				<< " one_2=" << one_2 << endl;
	}
	
	for (i = 0; i < q; i++) {
		if (SubS->embedding_2D[i] == T0) {
			t0 = i;
		}
		if (SubS->embedding_2D[i] == T1) {
			t1 = i;
		}
	}
	minus_t0 = Fq->negate(t0);
	if (f_v) {
		cout << "buekenhout_metz::buekenhout_metz_init "
				"t0=" << t0 << " t1=" << t1
				<< " minus_t0=" << minus_t0 << endl;
	}
	

	v = NEW_int(3);
	w1 = NEW_int(6);
	w2 = NEW_int(6);
	w3 = NEW_int(6);
	w4 = NEW_int(6);
	w5 = NEW_int(6);
	if (f_v) {
		cout << "buekenhout_metz::buekenhout_metz_init done" << endl;
	}
}

void buekenhout_metz::init_ovoid(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "buekenhout_metz::init_ovoid" << endl;
	}
	int X0, X1, Y1, Z, a;
	long int i;
	
	theta_3 = Gg.nb_PG_elements(3, q);
	ovoid = NEW_lint(theta_3);
	sz_ovoid = 0;
	for (i = 0; i < theta_3; i++) {
		Fq->Projective_space_basic->PG_element_unrank_modified(
				w1, 1, 4, i);
		if (f_vv) {
			cout << "testing point " << i << endl;
		}
		X0 = w1[0];
		X1 = w1[1];
		Y1 = w1[2];
		Z = w1[3];
		if (f_classical) {
#if 1
			// works in general:
			// X0^2 + t1*X0*X1 - t0*X1^2 + t1*Y1*Z:
			a = Fq->add4(
				Fq->mult(X0, X0), 
				Fq->product3(t1, X0, X1), 
				Fq->product3(minus_t0, X1, X1), 
				Fq->product3(t1, Y1, Z)
				);
#else
			// works only for GF(16):
			// 3 * X0^2 + X1^2 + Y1*Z + X0*X1 + 3*X0*Z + X1*Z
			a = Fq->add6(
				Fq->mult(3, Fq->mult(X0, X0)),
				Fq->mult(X1, X1), 
				Fq->mult(Y1, Z),
				Fq->mult(X0, X1),
				Fq->mult(3, Fq->mult(X0, Z)),
				Fq->mult(X1, Z)
				);
#endif
		}
		else {
			// works only for GF(16):
			// 2 * X0^2 + X1^2 + Y1*Z + X0*X1 + 2*X0*Z + X1*Z
			a = Fq->add6(
				Fq->mult(2, Fq->mult(X0, X0)),
				Fq->mult(X1, X1), 
				Fq->mult(Y1, Z),
				Fq->mult(X0, X1),
				Fq->mult(2, Fq->mult(X0, Z)),
				Fq->mult(X1, Z)
				);
		}
		if (a == 0) {
			ovoid[sz_ovoid++] = i;
		}
	}
	if (f_v) {
		cout << "found an ovoid of size " << sz_ovoid << ":" << endl;
		Lint_vec_print(cout, ovoid, sz_ovoid);
		cout << endl;
	}
	if (f_vv) {
		P3->Reporting->print_set(ovoid, sz_ovoid);
	}

	if (sz_ovoid != q * q + 1) {
		cout << "we need the ovoid to be of size q * q + 1" << endl;
		exit(1);
	}
}

void buekenhout_metz::init_ovoid_Uab_even(
		int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "buekenhout_metz::init_ovoid_Uab_even" << endl;
	}

	if (Fq->p != 2) {
		cout << "buekenhout_metz::init_ovoid_Uab_even, "
				"characteristic must be even" << endl;
		exit(1);
	}
	
	int X0, X1, Y1, Z, i, aa;
	int a1, a2, /*b1,*/ b2, delta, delta2;
	int lambda, lambda_big, c1, c2, c3;
	other_geometry::geometry_global Gg;
	
	a1 = SubS->components_2D[a * 2 + 0];
	a2 = SubS->components_2D[a * 2 + 1];
	//b1 = components[b * 2 + 0]; // b1 is unused
	b2 = SubS->components_2D[b * 2 + 1];
	delta = 2;
	delta2 = FQ->mult(delta, delta);
	lambda_big = FQ->add(delta, delta2);
	lambda = 0;
	for (i = 0; i < q; i++) {
		if (SubS->embedding_2D[i] == lambda_big) {
			lambda = i;
			break;
		}
	}
	c1 = Fq->add(a2, b2);
	c2 = b2;
	c3 = Fq->add3(a1, a2, Fq->mult(Fq->add(a2, b2), lambda));
	if (f_v) {
		cout << "buekenhout_metz::init_ovoid_Uab_even" << endl;
		cout << "delta=" << delta << endl;
		cout << "delta2=" << delta2 << endl;
		cout << "lambda_big=" << lambda_big << endl;
		cout << "lambda=" << lambda << endl;
		cout << "c1=" << c1 << endl;
		cout << "c2=" << c2 << endl;
		cout << "c3=" << c3 << endl;
	}
	
	theta_3 = Gg.nb_PG_elements(3, q);
	ovoid = NEW_lint(theta_3);
	sz_ovoid = 0;
	for (i = 0; i < theta_3; i++) {
		Fq->Projective_space_basic->PG_element_unrank_modified(
				w1, 1, 4, i);
		if (f_v) {
			cout << "testing point " << i << endl;
		}
		X0 = w1[0];
		X1 = w1[1];
		Y1 = w1[2];
		Z = w1[3];

		aa = Fq->add4(
			Fq->product3(c1, X0, X0), 
			Fq->product3(c2, X0, X1), 
			Fq->product3(c3, X1, X1), 
			Fq->mult(Y1, Z)
			);



		if (aa == 0) {
			ovoid[sz_ovoid++] = i;
		}
	}
	cout << "found an ovoid of size " << sz_ovoid << ":" << endl;
	Lint_vec_print(cout, ovoid, sz_ovoid);
	cout << endl;
	P3->Reporting->print_set(ovoid, sz_ovoid);

	if (sz_ovoid != q * q + 1) {
		cout << "we need the ovoid to be of size q * q + 1" << endl;
		exit(1);
	}
}

void buekenhout_metz::create_unital(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	int f_vvv = (verbose_level >= 4);
	int i, j, a, c1, c2, h;
	long int b;

	if (f_v) {
		cout << "buekenhout_metz::create_unital" << endl;
	}

	w1[0] = 0;
	w1[1] = 0;
	w1[2] = 1;
	w1[3] = 0;
	w1[4] = 0;

	if (f_v) {
		cout << "The vertex is:" << endl;
		Int_vec_print(cout, w1, 5);
		cout << endl;
	}
	
	U = NEW_lint(q * q * q + 1);
	sz = 0;
	for (i = 1; i < q * q + 1; i++) {
		a = ovoid[i];
		P3->unrank_point(w2, a);
		if (f_vv) {
			cout << i << "-th ovoidal point is " << a << " : ";
			Int_vec_print(cout, w2, 4);
			cout << endl;
		}

		if (w2[3] != 1) {
			cout << "we need the Z coordinate to be one" << endl;
			exit(1);
		}

		// Now, w2[4] is a point of the ovoid in PG(3,q)
		// We will now create the generator in PG(4,q)
		// This is just the line that joins the vertex to this point.

		w3[0] = w2[0];
		w3[1] = w2[1];
		w3[2] = 0;
		w3[3] = w2[2];
		w3[4] = w2[3];
		if (f_vv) {
			cout << "after embedding:" << endl;
			Int_vec_print(cout, w3, 5);
			cout << endl;
		}
		
		for (j = 0; j < q; j++) {
			if (f_vvv) {
				cout << "j=" << j << " : ";
			}
			for (h = 0; h < 5; h++) {
				w4[h] = Fq->mult(j, w1[h]);
			}
			if (f_vvv) {
				cout << "w4:" << endl;
				Int_vec_print(cout, w4, 5);
				cout << endl;
			}
			
			for (h = 0; h < 5; h++) {
				w5[h] = Fq->add(w4[h], w3[h]);
			}
			w5[5] = 0;
			if (f_vvv) {
				cout << "w5 (with added 0):" << endl;
				Int_vec_print(cout, w5, 6);
				cout << endl;
			}



			for (h = 0; h < 3; h++) {
				c1 = w5[2 * h + 0];
				c2 = w5[2 * h + 1];
				v[h] = SubS->pair_embedding_2D[c1 * q + c2];
			}

			if (f_vvv) {
				cout << " : ";
				Int_vec_print(cout, v, 3);
				//cout << endl;
			}

			b = P2->rank_point(v);

			if (f_vvv) {
				cout << " : " << b << endl;
			}
			U[sz++] = b;
		}
	}

	// the unique point (0,1,0) at infinity:
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	b = P2->rank_point(v);
	U[sz++] = b;

	if (f_v) {
		cout << "the Buekenhout Metz unital of size " << sz << " : ";
		Lint_vec_print(cout, U, sz);
		cout << endl;

		for (i = 0; i < sz; i++) {
			cout << U[i] << " ";
		}
		cout << endl;
		P2->Reporting->print_set(U, sz);
	}


}

void buekenhout_metz::create_unital_tex(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 3);
	int i, j, a, /*b,*/ c1, c2, h;
	string symbol;

	if (f_v) {
		cout << "buekenhout_metz::create_unital_tex" << endl;
	}

	symbol.assign("\\beta");

	w1[0] = 0;
	w1[1] = 0;
	w1[2] = 1;
	w1[3] = 0;
	w1[4] = 0;


	//cout << "The vertex is:" << endl;
	//int_vec_print(cout, w1, 5);
	//cout << endl;
	
	//U = NEW_int(q * q * q + 1);
	//sz = 0;
	for (i = 1; i < q * q + 1; i++) {
		a = ovoid[i];
		P3->unrank_point(w2, a);
		if (false) {
			cout << i << "-th ovoidal point is " << a << " : ";
			Int_vec_print(cout, w2, 4);
			cout << endl;
		}

		if (w2[3] != 1) {
			cout << "we need the Z coordinate to be one" << endl;
			exit(1);
		}

		w3[0] = w2[0];
		w3[1] = w2[1];
		w3[2] = 0;
		w3[3] = w2[2];
		w3[4] = w2[3];
		if (false) {
			cout << "after embedding:" << endl;
			Int_vec_print(cout, w3, 5);
			cout << endl;
		}
		cout << "(";
		Fq->Io->print_element_with_symbol(cout, w3[0],
				true /* f_exponential */, 8, symbol);
		cout << ",";
		Fq->Io->print_element_with_symbol(cout, w3[1],
				true /* f_exponential */, 8, symbol);
		cout << ",*,";
		Fq->Io->print_element_with_symbol(cout, w3[3],
				true /* f_exponential */, 8, symbol);
		cout << ",";
		Fq->Io->print_element_with_symbol(cout, w3[4],
				true /* f_exponential */, 8, symbol);
		cout << ") ";

		
		for (j = 0; j < q; j++) {
			cout << " & ";
			if (false) {
				cout << "j=" << j << " : ";
			}
			for (h = 0; h < 5; h++) {
				w4[h] = Fq->mult(j, w1[h]);
			}
			if (false) {
				cout << "w4:" << endl;
				Int_vec_print(cout, w4, 5);
				cout << endl;
			}
			
			for (h = 0; h < 5; h++) {
				w5[h] = Fq->add(w4[h], w3[h]);
			}
			w5[5] = 0;
			if (false) {
				cout << "w5 (with added 0):" << endl;
				Int_vec_print(cout, w5, 6);
				cout << endl;
			}



			for (h = 0; h < 3; h++) {
				c1 = w5[2 * h + 0];
				c2 = w5[2 * h + 1];
				v[h] = SubS->pair_embedding_2D[c1 * q + c2];
			}

			if (false) {
				cout << " : ";
				Int_vec_print(cout, v, 3);
				//cout << endl;
			}
			FQ->Io->int_vec_print_field_elements(cout, v, 3);

#if 0
			cout << "(";
			FQ->print_element_with_symbol(cout, v[0],
					true /* f_exponential */, 8, "\\alpha");
			cout << ",";
			FQ->print_element_with_symbol(cout, v[1],
					true /* f_exponential */, 8, "\\alpha");
			cout << ",";
			FQ->print_element_with_symbol(cout, v[2],
					true /* f_exponential */, 8, "\\alpha");
			cout << ") ";
#endif

#if 0
			int rk;
			
			rk = P2->rank_point(v);

			//cout << rk << " ";

			int x, y, t1, t2, t3;
			x = v[0];
			t1 = FQ->mult(parameter_a, FQ->mult(x, x));
			t2 = FQ->mult(parameter_b, FQ->power(x, q + 1));
			t3 = embedding[j];
			y = FQ->add3(
				t1, 
				t2, 
				t3
				);
			if (y != v[1]) {
				cout << "y != v[1]" << endl;
				cout << "y = ";
				FQ->print_element_with_symbol(cout, y,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				cout << "a*x^2 = ";
				FQ->print_element_with_symbol(cout, t1,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				cout << "b*x^(q+1) = ";
				FQ->print_element_with_symbol(cout, t2,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				cout << "r = ";
				FQ->print_element_with_symbol(cout, t3,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				//exit(1);
			}
			FQ->print_element_with_symbol(cout, y,
					true /* f_exponential */, 8, "\\alpha");

			b = P2->rank_point(v);
#endif
		}

		cout << "\\\\" << endl;
	}
}

void buekenhout_metz::create_unital_Uab_tex(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 3);
	int i, r, x, y;
	string symbol;

	if (f_v) {
		cout << "buekenhout_metz::create_unital_Uab_tex" << endl;
	}

	symbol.assign("\\beta");

	for (r = 0; r < q; r++) {
		cout << " & ";
		Fq->Io->print_element_with_symbol(cout, r,
				true /* f_exponential */, 8, symbol);
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	for (i = 0; i < q * q; i++) {
		
		if (i == 0) {
			x = 0;
		}
		else {
			x = FQ->alpha_power(i - 1);
		}
		FQ->Io->print_element_with_symbol(cout, x,
				true /* f_exponential */, 8, symbol);

		for (r = 0; r < q; r++) {
			cout << " & ";
			
			int t1, t2, t3;

			t1 = FQ->mult(parameter_a, FQ->mult(x, x));
			t2 = FQ->mult(parameter_b, FQ->power(x, q + 1));
			t3 = SubS->embedding_2D[r];
			y = FQ->add3(
				t1, 
				t2, 
				t3
				);

			v[0] = x;
			v[1] = y;
			v[2] = 1;


			FQ->Io->int_vec_print_field_elements(cout, v, 3);
#if 0
			cout << "(";
			FQ->print_element_with_symbol(cout, v[0],
					true /* f_exponential */, 8, "\\alpha");
			cout << ",";
			FQ->print_element_with_symbol(cout, v[1],
					true /* f_exponential */, 8, "\\alpha");
			cout << ",";
			FQ->print_element_with_symbol(cout, v[2],
					true /* f_exponential */, 8, "\\alpha");
			cout << ") ";
#endif


			int rk;
			
			rk = P2->rank_point(v);

			cout << " = P_{" << rk << "} ";

#if 0
			if (y != v[1]) {
				cout << "y != v[1]" << endl;
				cout << "y = ";
				FQ->print_element_with_symbol(cout, y,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				cout << "a*x^2 = ";
				FQ->print_element_with_symbol(cout, t1,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				cout << "b*x^(q+1) = ";
				FQ->print_element_with_symbol(cout, t2,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				cout << "r = ";
				FQ->print_element_with_symbol(cout, t3,
						true /* f_exponential */, 8, "\\alpha");
				cout << endl;
				//exit(1);
			}
#endif
			//FQ->print_element_with_symbol(cout, y,
			// true /* f_exponential */, 8, "\\alpha");

		}

		cout << "\\\\" << endl;
	}
}

void buekenhout_metz::compute_the_design(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, h, a, b;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "buekenhout_metz::compute_the_design" << endl;
	}


	tangent_lines = NEW_lint(P2->Subspaces->N_lines);
	secant_lines = NEW_lint(P2->Subspaces->N_lines);
	block = NEW_lint(q + 1);


	P2->Subspaces->find_k_secant_lines(U, sz, q + 1,
		secant_lines, nb_secant_lines, verbose_level - 1);

	if (f_vv) {
		cout << "There are " << nb_secant_lines
				<< " secant lines, they are:" << endl;
		Lint_vec_print(cout, secant_lines, nb_secant_lines);
		cout << endl;
	}

	P2->Subspaces->find_k_secant_lines(U, sz, 1,
		tangent_lines, nb_tangent_lines, verbose_level - 1);

	if (f_vv) {
		cout << "There are " << nb_tangent_lines
				<< " tangent lines, they are:" << endl;
		Lint_vec_print(cout, tangent_lines, nb_tangent_lines);
		cout << endl;
	}


	
	tangent_line_at_point = NEW_int(P2->Subspaces->N_points);
	f_is_tangent_line = NEW_int(P2->Subspaces->N_lines);
	point_of_tangency = NEW_int(P2->Subspaces->N_lines);
	for (i = 0; i < P2->Subspaces->N_points; i++) {
		tangent_line_at_point[i] = -1;
	}
	for (i = 0; i < P2->Subspaces->N_lines; i++) {
		f_is_tangent_line[i] = false;
		point_of_tangency[i] = -1;
	}
	for (h = 0; h < nb_tangent_lines; h++) {
		a = tangent_lines[h];
		f_is_tangent_line[a] = true;
		P2->Subspaces->intersect_with_line(U, sz,
				a /* line_rk */, block, block_size,
				0 /* verbose_level*/);
		//Sorting.int_vec_intersect(P2->Lines + a * P2->k, P2->k,
		//		U, sz, block, block_size);
		if (block_size != 1) {
			cout << "block_size != 1" << endl;
			exit(1);
		}
		b = block[0];
		if (f_vv) {
			cout << "line " << a << " is tangent at point " << b << endl;
		}
		tangent_line_at_point[b] = a;
		point_of_tangency[a] = b;
	}
	for (b = 0; b < P2->Subspaces->N_points; b++) {
		if (tangent_line_at_point[b] == -1) {
			continue;
		}
		if (f_vv) {
			cout << "The tangent line at point " << b
					<< " is line " << tangent_line_at_point[b] << endl;
		}
	}

	idx_in_unital = NEW_int(P2->Subspaces->N_points);
	idx_in_secants = NEW_int(P2->Subspaces->N_lines);
	for (i = 0; i < P2->Subspaces->N_points; i++) {
		idx_in_unital[i] = -1;
	}
	for (i = 0; i < P2->Subspaces->N_lines; i++) {
		idx_in_secants[i] = -1;
	}
	for (i = 0; i < sz; i++) {
		a = U[i];
		idx_in_unital[a] = i;
	}
	for (i = 0; i < nb_secant_lines; i++) {
		a = secant_lines[i];
		idx_in_secants[a] = i;
	}

	Intersection_sets = NEW_lint(nb_secant_lines * (q + 1));
	Design_blocks = NEW_int(nb_secant_lines * (q + 1));

	for (h = 0; h < nb_secant_lines; h++) {
		a = secant_lines[h];
		P2->Subspaces->intersect_with_line(U, sz,
				a /* line_rk */, block, block_size,
				0 /* verbose_level*/);
		//Sorting.int_vec_intersect(P2->Lines + a * P2->k,
		//		P2->k, U, sz, block, block_size);
		if (block_size != q + 1) {
			cout << "block_size != q + 1" << endl;
			exit(1);
		}
		for (j = 0; j < q + 1; j++) {
			b = idx_in_unital[block[j]];
			if (b == -1) {
				cout << "b == -1" << endl;
				exit(1);
			}
			Intersection_sets[h * (q + 1) + j] = block[j];
			Design_blocks[h * (q + 1) + j] = b;
		}
	}

	if (f_vv) {
		cout << "The blocks of the design are:" << endl;
		Int_vec_print_integer_matrix_width(cout, Design_blocks,
				nb_secant_lines, q + 1, q + 1, 3);
	}


	f_is_Baer = NEW_int(nb_secant_lines);
	for (j = 0; j < nb_secant_lines; j++) {
		f_is_Baer[j] = P2->is_contained_in_Baer_subline(
				Intersection_sets + j * (q + 1), q + 1,
				0 /*verbose_level - 1*/);
	}

	if (f_vv) {
		cout << "The intersection sets are:" << endl;
		cout << "i : line(i) : block(i) : is Baer" << endl;
		for (i = 0; i < nb_secant_lines; i++) {
			cout << setw(3) << i << " : ";
			cout << setw(3) << secant_lines[i] << " : ";
			Lint_vec_print(cout, Intersection_sets + i * (q + 1), q + 1);
			cout << " : ";
			cout << setw(3) << f_is_Baer[i] << endl;
		}
	}
}

#if 0
void buekenhout_metz::compute_automorphism_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "buekenhout_metz::compute_automorphism_group" << endl;
		cout << "computing the automorphism group of the design:" << endl;
	}

	

	A = create_automorphism_group_of_block_system(
		sz /* nb_points */, nb_secant_lines /* nb_blocks */,
		q + 1 /* block_size */,
		Design_blocks /* Blocks */, 
		verbose_level - 2);
	A->group_order(ago);

	if (f_v) {
		cout << "The automorphism group of the design "
				"has order " << ago << endl;
	}

	if (f_v) {
		cout << "Computing the automorphism group again" << endl;
	}



	get_name(fname_stab);
	strcat(fname_stab, "_stab.txt");

	if (file_size(fname_stab) <= 0) {

		if (f_v) {
			cout << "file " << fname_stab << " does not exist, "
					"we will now compute the stabilizer" << endl;
		}
		S = create_sims_for_stabilizer(P2->A, U, sz, verbose_level - 1);
		S->write_sgs(fname_stab, verbose_level - 1);
	}
	else {
		vector_ge *SG;
		
		if (f_v) {
			cout << "file " << fname_stab << " exists, we will "
					"now read the stabilizer from file" << endl;
		}
		S = NEW_OBJECT(sims);
		S->init(P2->A);
		SG = NEW_OBJECT(vector_ge);
		S->read_sgs(fname_stab, SG, verbose_level);
		FREE_OBJECT(SG);
	}

	S->group_order(ago2);

	if (f_v) {
		cout << "The stabilizer of the unital has order " << ago2 << endl;
	}


	gens = NEW_OBJECT(vector_ge);
	tl = NEW_int(P2->A->base_len);
	S->extract_strong_generators_in_order(*gens, tl, verbose_level);

	if (f_v) {
		cout << "strong generators for the stabilizer are:" << endl;
		gens->print(cout);

		S->print_generators_tex(cout);
	}
}


void buekenhout_metz::compute_orbits(int verbose_level)
// Computes the orbits on points and on lines,
// then calls investigate_line_orbit for all line orbits
{
	int f_v = (verbose_level >= 1);
	int h;
	

	if (f_v) {
		cout << "buekenhout_metz::compute_orbits" << endl;
		cout << "computing orbits on points and on lines:" << endl;
	}


	Orb = NEW_OBJECT(schreier);
	Orb->init(P2->A);
	Orb->init_generators(*gens);
	Orb->compute_all_point_orbits(verbose_level - 2);
	if (f_v) {
		cout << "Orbits on points:" << endl;
		Orb->print_and_list_orbits(cout);
	}
	

	Orb2 = NEW_OBJECT(schreier);
	Orb2->init(P2->A2);
	Orb2->init_generators(*gens);

	if (f_prefered_line_reps) {
		Orb2->compute_all_point_orbits_with_prefered_reps(
			prefered_line_reps, nb_prefered_line_reps, verbose_level - 2);
	}
	else {
		Orb2->compute_all_point_orbits(verbose_level - 2);
	}
	if (f_v) {
		cout << "Orbits on lines:" << endl;
		Orb2->print_and_list_orbits(cout);
	}
	

	for (h = 0; h < Orb2->nb_orbits; h++) {
		
		if (f_v) {
			cout << "buekenhout_metz::compute_orbits "
					"before investigate_line_orbit " << h << endl;
		}
		
		investigate_line_orbit(h, verbose_level);
			
		if (f_v) {
			cout << "buekenhout_metz::compute_orbits "
					"after investigate_line_orbit " << h << endl;
		}
	}
		

}

void buekenhout_metz::investigate_line_orbit(int h, int verbose_level)
// Investigates a secant line orbit
// We compute the stabilizer of the secant line.
// We compute the orbits of the stabilizer
// on pairs of points from the block.
// Let (PP1,PP2) be a pair
// Let (T1,T2) be the corresponding tangent lines
// Let Q be the intersection point of T1 and T2
// Let T1_set be the tangent lines on Q that do not pass through PP1 or PP2
// Finally, we compute the stabilizer of T1_set in the stabilizer of the pair.
// This is done in a rather crude way, namely by testing each group element.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int u;

	if (f_v) {
		cout << "buekenhout_metz::investigate_line_orbit" << endl;
	}
	longinteger_object stab_order;
	int the_line;
	int idx, f_hit_favorite;
	int PP1, PP2, T1, T2, T, Q, PP, vv;

		
	the_line = Orb2->orbit[Orb2->orbit_first[h]];
		

	// make sure the line is a secant line:
	if (!int_vec_search_linear(secant_lines,
			nb_secant_lines, the_line, idx)) {
		if (f_v) {
			cout << "line-orbit " << h << " represented by line "
				<< the_line << " does not consist of "
						"secant lines, skip" << endl;
		}
		return;
	}
		 


	if (f_v) {
		cout << "looking at secant line-orbit " << h
				<< " represented by line " << the_line << ":" << endl;
	}
		

	int_vec_intersect(P2->Lines + the_line * P2->k,
			P2->k, U, sz, good_points, nb_good_points);
		
	if (f_v) {
		cout << "the block is ";
		int_vec_print(cout, good_points, nb_good_points);
		cout << endl;
	}

	if (nb_good_points != q + 1) {
		cout << "nb_good_points != q + 1" << endl;
		exit(1);
	}

	Orb2->point_stabilizer(P2->A /* default_action */, ago2, 
		Stab, h /* orbit_no */, verbose_level);

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "stabilizer of orbit " << h
				<< " has order " << stab_order << endl;
	}
	

	if (f_vv) {
		cout << "the stabilizer is generated by" << endl;
		Stab->print_generators();
	}
	
	C = NEW_OBJECT(choose_points_or_lines);

		
	C->init("pairs", this /*void *data*/, 
		P2->A, P2->A2, 
		false /* f_choose_lines */, 
		2 /*nb_points_or_lines */, 
		buekenhout_metz_check_good_points, 
		t0, 
		verbose_level - 1);
		
	if (f_v) {
		cout << "computing orbits" << endl;
	}
	C->compute_orbits_from_sims(Stab, verbose_level - 1);

	if (f_v) {
		cout << "We found " << C->nb_orbits
				<< " orbits on 2-subsets" << endl;
		cout << "They are:" << endl;
	}
	for (u = 0; u < C->nb_orbits; u++) {

		if (f_v) {
			cout << "choosing orbit rep " << u << endl;
		}

		C->choose_orbit(u, f_hit_favorite, verbose_level);

		if (f_v) {
			cout << "orbit rep " << u << " is" << endl;
			C->print_rep();
			cout << endl;
		}
#if 0
		if (f_vv) {
			cout << "with stabilizer tl=";
			int_vec_print(cout, C->stab_tl, C->A->base_len);
			cout << endl;
			cout << "generated by" << endl;
			C->stab_gens->print(cout);
		}
#endif


		PP1 = C->representative[0];
		PP2 = C->representative[1];
		T1 = tangent_line_at_point[PP1];
		T2 = tangent_line_at_point[PP2];
		Q = P2->line_intersection(T1, T2);

		if (f_v) {
			cout << "P1=" << PP1 << " P2=" << PP2
					<< " T1=" << T1 << " T2=" << T2 << " Q=" << Q << endl;
		}



		int *T1_set;
		int T1_set_size;

		T1_set = NEW_int(q + 1);
		T1_set_size = 0;
			
		t1 = 0;			
		for (vv = 0; vv < P2->r; vv++) {
			T = P2->Lines_on_point[Q * P2->r + vv];
			if (!f_is_tangent_line[T]) {
				continue;
			}
			PP = P2->line_intersection(T, the_line);
			if (PP == PP1 || PP == PP2) {
			}
			else {
				T1_set[T1_set_size++] = T;
			}
			if (int_vec_search_linear(good_points,
					nb_good_points, PP, idx)) {
				if (f_v) {
					cout << vv << "-th line " << T << " on Q is "
						"tangent line and intersects in "
						"good point " << PP << endl;
				}
				t1++;
			}
			else {
				if (f_v) {
					cout << vv << "-th line " << T << " on Q "
						"is tangent line and its point of tangency "
						"is " << point_of_tangency[T]
						<< " which is not a good point" << endl;
				}
			}
		}
		if (f_v) {
			cout << "t1=" << t1 << endl;
			cout << "T1_set = ";
			int_vec_print(cout, T1_set, T1_set_size);
			cout << endl;
		}

		longinteger_object go;
		int *Elt;
		int goi, go1, ii;
		sims *Stab0;

		Stab0 = create_sims_from_generators_with_target_group_order_factorized(
			P2->A, 
			C->Stab_Strong_gens->gens,
			C->Stab_Strong_gens->tl,
			P2->A->base_len,
			verbose_level - 2);
		if (f_v) {
			cout << "computing stabilizer of the line set" << endl;
			Stab0->group_order(go);
			cout << "in a group of order " << go << endl;
		}
		goi = go.as_int();
		go1 = 0;
		Elt = NEW_int(P2->A->elt_size_in_int);
			
		for (ii = 0; ii < goi; ii++) {
			Stab0->element_unrank_int(ii, Elt);
			if (P2->A2->check_if_in_set_stabilizer(Elt, 
				T1_set_size, T1_set, 0 /*verbose_level*/)) {
				if (f_vv) {
					cout << "element " << ii << " stabilizes the set" << endl;
					P2->A2->element_print_as_permutation(Elt, cout);
					cout << endl;
				}
				go1++;
			}
		}
		if (f_v) {
			cout << "stabilizer order " << go1 << endl;
		}
		FREE_int(Elt);

#if 0
		int i;
		cout << "computing stabilizer using set stabilizer routine:" << endl;
		sims *T1_set_stab;
		T1_set_stab = create_sims_for_stabilizer_with_input_group(P2->A2, 
			P2->A, C->stab_gens, C->stab_tl, 
			T1_set, T1_set_size, verbose_level - 3);
		T1_set_stab->group_order(go);
		cout << "stabilizer order " << go << endl;
			
		vector_ge *set_stab_gens;
		int *set_stab_tl;
			
		set_stab_gens = NEW_OBJECT(vector_ge);
		set_stab_tl = NEW_int(P2->A->base_len);
		T1_set_stab->extract_strong_generators_in_order(
				*set_stab_gens, set_stab_tl, 0/*verbose_level*/);
		cout << "strong generators are:" << endl;
		for (i = 0; i < set_stab_gens->len; i++) {
			P2->A->element_print_quick(set_stab_gens->ith(i), cout);
			cout << endl;
		}
		FREE_OBJECT(T1_set_stab);
#endif
	} // next u
}
#endif


void buekenhout_metz::write_unital_to_file()
{
	string fname_unital;
	other::orbiter_kernel_system::file_io Fio;
	
	get_name(fname_unital);
	fname_unital += ".txt";

	Fio.write_set_to_file(fname_unital, U, sz, 0 /* verbose_level */);

	cout << "written file " << fname_unital << " of size "
			<< Fio.file_size(fname_unital) << endl;
}


void buekenhout_metz::get_name(
		std::string &name)
{

	if (f_Uab) {
		name = "U_" + std::to_string(parameter_a) + "_"
				+ std::to_string(parameter_b) + "_"
				+ std::to_string(q);
	}
	else {
		if (f_classical) {
			name = "H" + std::to_string(q);
		}
		else {
			name = "BM" + std::to_string(q);
		}
	}
}

#if 0
int buekenhout_metz_check_good_points(int len,
		int *S, void *data, int verbose_level)
// used in buekenhout_metz::investigate_line_orbit
{
	int i, a, idx;
	int f_v = false;
	buekenhout_metz *BM = (buekenhout_metz *) data;
	sorting Sorting;

	if (f_v) {
		cout << "buekenhout_metz_check_good_points checking the set ";
		Orbiter->Int_vec.print(cout, S, len);
		cout << endl;
	}
	for (i = 0; i < len; i++) {
		a = S[i];
		if (!Sorting.int_vec_search_linear(BM->good_points,
				BM->nb_good_points, a, idx)) {
			if (f_v) {
				cout << "The set is rejected" << endl;
			}
			return false;
		}
	}
	if (f_v) {
		cout << "The set is accepted" << endl;
	}
	return true;
}
#endif

}}}}



