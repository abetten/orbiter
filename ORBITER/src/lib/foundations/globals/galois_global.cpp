// galois_global.C
//
// Anton Betten
//
// started: Oct 16, 2013
//
// unipoly stuff:
// started:  November 16, 2002
// moved here: Oct 16, 2013




#include "foundations.h"

namespace orbiter {
namespace foundations {


void test_unipoly()
{
	finite_field GFp;
	int p = 2;
	unipoly_object m, a, b, c;
	unipoly_object elts[4];
	int i, j;
	int verbose_level = 0;
	
	GFp.init(p, verbose_level);
	unipoly_domain FX(&GFp);
	
	FX.create_object_by_rank(m, 7);
	FX.create_object_by_rank(a, 5);
	FX.create_object_by_rank(b, 55);
	FX.print_object(a, cout); cout << endl;
	FX.print_object(b, cout); cout << endl;

	unipoly_domain Fq(&GFp, m);
	Fq.create_object_by_rank(c, 2);
	for (i = 0; i < 4; i++) {
		Fq.create_object_by_rank(elts[i], i);
		cout << "elt_" << i << " = ";
		Fq.print_object(elts[i], cout); cout << endl;
		}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			Fq.print_object(elts[i], cout);
			cout << " * ";
			Fq.print_object(elts[j], cout);
			cout << " = ";
			Fq.mult(elts[i], elts[j], c);
			Fq.print_object(c, cout); cout << endl;
			
			FX.mult(elts[i], elts[j], a);
			FX.print_object(a, cout); cout << endl;
			}
		}
	
}

void test_unipoly2()
{
	finite_field Fq;
	int q = 4, p = 2, i;
	int verbose_level = 0;
	
	Fq.init(q, verbose_level);
	unipoly_domain FX(&Fq);
	
	unipoly_object a;
	
	FX.create_object_by_rank(a, 0);
	for (i = 1; i < q; i++) {
		FX.minimum_polynomial(a, i, p, TRUE);
		//cout << "minpoly_" << i << " = ";
		//FX.print_object(a, cout); cout << endl;
		}
	
}

char *search_for_primitive_polynomial_of_given_degree(
		int p, int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field Fp;
	
	Fp.init(p, 0 /*verbose_level*/);
	unipoly_domain FX(&Fp);
	
	unipoly_object m;
	longinteger_object rk;
	
	FX.create_object_by_rank(m, 0);
	
	if (f_v) {
		cout << "search_for_primitive_polynomial_of_given_degree "
				"p=" << p << " degree=" << degree << endl;
		}
	FX.get_a_primitive_polynomial(m, degree, verbose_level - 1);
	FX.rank_longinteger(m, rk);
	
	char *s;
	int i, j;
	if (f_v) {
		cout << "found a polynomial. It's rank is " << rk << endl;
		}
	
	s = NEW_char(rk.len() + 1);
	for (i = rk.len() - 1, j = 0; i >= 0; i--, j++) {
		s[j] = '0' + rk.rep()[i];
		}
	s[rk.len()] = 0;
	
	if (f_v) {
		cout << "created string " << s << endl;
		}
	
	return s;
}


void search_for_primitive_polynomials(
		int p_min, int p_max, int n_min, int n_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, p;
	

	longinteger_f_print_scientific = FALSE;

	
	if (f_v) {
		cout << "search_for_primitive_polynomials "
				"p_min=" << p_min << " p_max=" << p_max
				<< " n_min=" << n_min << " n_max=" << n_max << endl;
		}
	for (p = p_min; p <= p_max; p++) {
		if (!is_prime(p)) {
			continue;
			}
		if (f_v) {
			cout << "considering the prime " << p << endl;
			}

			{
			finite_field Fp;
			Fp.init(p, 0 /*verbose_level*/);
			unipoly_domain FX(&Fp);
	
			unipoly_object m;
			longinteger_object rk;
	
			FX.create_object_by_rank(m, 0);
	
			for (d = n_min; d <= n_max; d++) {
				if (f_v) {
					cout << "d=" << d << endl;
					}
				FX.get_a_primitive_polynomial(m, d, verbose_level - 1);
				FX.rank_longinteger(m, rk);
				//cout << d << " : " << rk << " : ";
				cout << "\"" << rk << "\", // ";
				FX.print_object(m, cout);
				cout << endl;
				}
			FX.delete_object(m);
			}
		}
}


void make_linear_irreducible_polynomials(int q, int &nb,
		int *&table, int verbose_level)
{
	int i;
	
	finite_field F;
	F.init(q, 0 /*verbose_level*/);
#if 0
	if (f_no_eigenvalue_one) {
		nb = q - 2;
		table = NEW_int(nb * 2);
		for (i = 0; i < nb; i++) {
			table[i * 2 + 0] = F.negate(i + 2);
			table[i * 2 + 1] = 1;
			}
		}
	else {
#endif
		nb = q - 1;
		table = NEW_int(nb * 2);
		for (i = 0; i < nb; i++) {
			table[i * 2 + 0] = F.negate(i + 1);
			table[i * 2 + 1] = 1;
			}
#if 0
		}
#endif
}



void gl_random_matrix(int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int *M;
	int *M2;
	finite_field F;
	unipoly_object char_poly;

	if (f_v) {
		cout << "gl_random_matrix" << endl;
		}
	F.init(q, 0 /*verbose_level*/);
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);
	
	F.random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	int_matrix_print(M, k, k);

	
	{
	unipoly_domain U(&F);



	U.create_object_by_rank(char_poly, 0);
		
	U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

	cout << "The characteristic polynomial is ";
	U.print_object(char_poly, cout);
	cout << endl;

	U.substitute_matrix_in_polynomial(char_poly, M, M2, k, verbose_level);
	cout << "After substitution, the matrix is " << endl;
	int_matrix_print(M2, k, k);

	U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

}

void save_as_colored_graph_easy(const char *fname_base,
		int n, int *Adj, int verbose_level)
{
	char fname[1000];
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "save_as_colored_graph_easy" << endl;
		}
	sprintf(fname, "%s.colored_graph", fname_base);


	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(n, Adj, 0 /*verbose_level*/);

	CG->save(fname, verbose_level);

	if (f_v) {
		cout << "save_as_colored_graph_easy Written file "
				<< fname << " of size " << file_size(fname) << endl;
		}
}

void save_colored_graph(const char *fname,
	int nb_vertices, int nb_colors,
	int *points, int *point_color, 
	int *data, int data_sz, 
	uchar *bitvector_adjacency, int bitvector_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	FILE *fp;
	int i;

	if (f_v) {
		cout << "save_colored_graph" << endl;
		cout << "save_colored_graph fname=" << fname << endl;
		cout << "save_colored_graph nb_vertices=" << nb_vertices << endl;
		cout << "save_colored_graph nb_colors=" << nb_colors << endl;
		//cout << "points:";
		//int_vec_print(cout, points, nb_vertices);
		//cout << endl;
		}

	
	fp = fopen(fname, "wb");

	fwrite_int4(fp, nb_vertices);
	fwrite_int4(fp, nb_colors);
	fwrite_int4(fp, data_sz);
	for (i = 0; i < data_sz; i++) {
		fwrite_int4(fp, data[i]);
		}
	for (i = 0; i < nb_vertices; i++) {
		if (points) {
			fwrite_int4(fp, points[i]);
			}
		else {
			fwrite_int4(fp, 0);
			}
		fwrite_int4(fp, point_color[i]);
		}
	fwrite_uchars(fp, bitvector_adjacency, bitvector_length);
	fclose(fp);


	if (f_v) {
		cout << "save_colored_graph done" << endl;
		}
}


void load_colored_graph(const char *fname,
	int &nb_vertices, int &nb_colors,
	int *&vertex_labels, int *&vertex_colors, 
	int *&user_data, int &user_data_size, 
	uchar *&bitvector_adjacency, int &bitvector_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	FILE *fp;
	int i, L;

	if (f_v) {
		cout << "load_colored_graph" << endl;
		}
	fp = fopen(fname, "rb");

	nb_vertices = fread_int4(fp);
	nb_colors = fread_int4(fp);
	if (f_v) {
		cout << "load_colored_graph nb_vertices=" << nb_vertices
			<< " nb_colors=" << nb_colors << endl;
		}


	L = (nb_vertices * (nb_vertices - 1)) >> 1;

	bitvector_length = (L + 7) >> 3;

	if (f_v) {
		cout << "load_colored_graph user_data_size="
				<< user_data_size << endl;
		}
	user_data_size = fread_int4(fp);
	user_data = NEW_int(user_data_size);
	
	for (i = 0; i < user_data_size; i++) {
		user_data[i] = fread_int4(fp);
		}

	vertex_labels = NEW_int(nb_vertices);
	vertex_colors = NEW_int(nb_vertices);
	
	for (i = 0; i < nb_vertices; i++) {
		vertex_labels[i] = fread_int4(fp);
		vertex_colors[i] = fread_int4(fp);
		if (vertex_colors[i] >= nb_colors) {
			cout << "colored_graph::load" << endl;
			cout << "vertex_colors[i] >= nb_colors" << endl;
			cout << "vertex_colors[i]=" << vertex_colors[i] << endl;
			cout << "i=" << i << endl;
			cout << "nb_colors=" << nb_colors << endl;
			exit(1);
			}
		}

	bitvector_adjacency = NEW_uchar(bitvector_length);
	fread_uchars(fp, bitvector_adjacency, bitvector_length);


	fclose(fp);
	if (f_v) {
		cout << "load_colored_graph done" << endl;
		}
}

int is_diagonal_matrix(int *A, int n)
{
	int i, j;
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				continue;
				}
			else {
				if (A[i * n + j]) {
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}

int is_association_scheme(int *color_graph, int n, int *&Pijk, 
	int *&colors, int &nb_colors, int verbose_level)
// color_graph[n * n]
// added Dec 22, 2010.
//Originally in BLT_ANALYZE/analyze_plane_invariant.C
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int N;
	int *M1;
	int k, i, j;
	int ret = FALSE;
	
	if (f_v) {
		cout << "is_association_scheme" << endl;
		}
	N = (n * (n - 1)) / 2;
	M1 = NEW_int(N);
	k = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			M1[k++] = color_graph[i * n + j];
			}
		}
	if (k != N) {
		cout << "N=" << N << endl;
		cout << "k=" << k << endl;
		exit(1);
		}

	classify Cl;

	Cl.init(M1, N, FALSE, 0);
	nb_colors = Cl.nb_types + 1;
	colors = NEW_int(nb_colors);
	colors[0] = color_graph[0];
	for (i = 0; i < Cl.nb_types; i++) {
		colors[i + 1] = Cl.data_sorted[Cl.type_first[i]];
		}

	if (f_vv) {
		cout << "colors (the 0-th color is the diagonal color): ";
		int_vec_print(cout, colors, nb_colors);
		cout << endl;
		}

	int C = nb_colors;
	int *M = color_graph;
	int pijk, pijk1, u, v, w, u0 = 0, v0 = 0;
	
	Pijk = NEW_int(C * C * C);
	int_vec_zero(Pijk, C * C * C);
	for (k = 0; k < C; k++) {
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				pijk = -1;
				for (u = 0; u < n; u++) {
					for (v = 0; v < n; v++) {
						//if (v == u) continue;
						if (M[u * n + v] != colors[k])
							continue;
						// now: edge (u,v) is colored k
						pijk1 = 0;
						for (w = 0; w < n; w++) {
							//if (w == u)continue;
							//if (w == v)continue;
							if (M[u * n + w] != colors[i])
								continue;
							if (M[v * n + w] != colors[j])
								continue;
							//cout << "i=" << i << " j=" << j << " k=" << k
							//<< " u=" << u << " v=" << v << " w=" << w
							//<< " increasing pijk" << endl;
							pijk1++;
							} // next w
						//cout << "i=" << i << " j=" << j << " k=" << k
						//<< " u=" << u << " v=" << v
						//<< " pijk1=" << pijk1 << endl;
						if (pijk == -1) {
							pijk = pijk1;
							u0 = u;
							v0 = v;
							//cout << "u=" << u << " v=" << v
							//<< " p_{" << i << "," << j << ","
							//<< k << "}="
							//<< Pijk[i * C * C + j * C + k] << endl;
							}
						else {
							if (pijk1 != pijk) {
								//FREE_int(Pijk);
								//FREE_int(colors);

								cout << "not an association scheme" << endl;
								cout << "k=" << k << endl;
								cout << "i=" << i << endl;
								cout << "j=" << j << endl;
								cout << "u0=" << u0 << endl;
								cout << "v0=" << v0 << endl;
								cout << "pijk=" << pijk << endl;
								cout << "u=" << u << endl;
								cout << "v=" << v << endl;
								cout << "pijk1=" << pijk1 << endl;
								//exit(1);

								goto done;
								}
							}
						} // next v
					} // next u
				Pijk[i * C * C + j * C + k] = pijk;
				} // next j
			} // next i
		} // next k

	ret = TRUE;

	if (f_v) {
		cout << "it is an association scheme" << endl;


		if (f_v) {
			print_Pijk(Pijk, C);
			}

		if (C == 3 && colors[1] == 0 && colors[2] == 1) {
			int k, lambda, mu;

			k = Pijk[2 * C * C + 2 * C + 0]; // p220;
			lambda = Pijk[2 * C * C + 2 * C + 2]; // p222;
			mu = Pijk[2 * C * C + 2 * C + 1]; // p221;
			cout << "it is an srg(" << n << "," << k << ","
					<< lambda << "," << mu << ")" << endl;
			}


		}
	

done:
	FREE_int(M1);
	return ret;
}

void print_Pijk(int *Pijk, int nb_colors)
{
	int i, j, k;
	int C = nb_colors;
	
	for (k = 0; k < C; k++) {
		int *Mtx;

		Mtx = NEW_int(C * C);
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				Mtx[i * C + j] = Pijk[i * C * C + j * C + k];
				}
			}
		cout << "P^{(" << k << ")}=(p_{i,j," << k << "})_{i,j}:" << endl;
		print_integer_matrix_width(cout, Mtx, C, C, C, 3);
		FREE_int(Mtx);
		}
}




void write_colored_graph(ofstream &ost, char *label, 
	int point_offset, 
	int nb_points, 
	int f_has_adjacency_matrix, int *Adj, 
	int f_has_adjacency_list, int *adj_list, 
	int f_has_bitvector, uchar *bitvector_adjacency, 
	int f_has_is_adjacent_callback, 
	int (*is_adjacent_callback)(int i, int j, void *data), 
	void *is_adjacent_callback_data, 
	int f_colors, int nb_colors, int *point_color, 
	int f_point_labels, int *point_label)
{
	int i, j, d, aij = 0;
    int w;
	
	cout << "write_graph " << label 
		<< " with " << nb_points 
		<< " points, point_offset=" <<  point_offset
		<< endl;
	w = (int) int_log10(nb_points);
	cout << "w=" << w << endl;
	ost << "<GRAPH label=\"" << label << "\" num_pts=\"" << nb_points 
		<< "\" f_has_colors=\"" <<  f_colors
		<< "\" num_colors=\"" << nb_colors 
		<< "\" point_offset=\"" <<  point_offset
		<< "\" f_point_labels=\"" <<  f_point_labels
		<< "\">" 
		<< endl;
	for (i = 0; i < nb_points; i++) {
		d = 0;
		for (j = 0; j < nb_points; j++) {
			if (j == i) {
				continue;
				}
			if (f_has_adjacency_matrix) {
				aij = Adj[i * nb_points + j];
				}
			else if (f_has_adjacency_list) {
				int h;
				if (i < j) {
					h = ij2k(i, j, nb_points);
					}
				else {
					h = ij2k(j, i, nb_points);
					}
				aij = adj_list[h];
				}
			else if (f_has_bitvector) {
				int h;
				if (i < j) {
					h = ij2k(i, j, nb_points);
					}
				else {
					h = ij2k(j, i, nb_points);
					}
				aij = bitvector_s_i(bitvector_adjacency, h);
				}
			else if (f_has_is_adjacent_callback) {
				aij = (*is_adjacent_callback)(i, j,
					is_adjacent_callback_data);
				}
			else {
				cout << "write_colored_graph cannot "
						"determine adjacency" << endl;
				}

			if (aij) {
				d++;
				}
			}
		ost << setw(w) << i + point_offset << " " << setw(w) << d << " ";
		for (j = 0; j < nb_points; j++) {
			if (j == i) {
				continue;
				}
			if (f_has_adjacency_matrix) {
				aij = Adj[i * nb_points + j];
				}
			else if (f_has_adjacency_list) {
				int h;
				if (i < j) {
					h = ij2k(i, j, nb_points);
					}
				else {
					h = ij2k(j, i, nb_points);
					}
				aij = adj_list[h];
				}
			else if (f_has_bitvector) {
				int h;
				if (i < j) {
					h = ij2k(i, j, nb_points);
					}
				else {
					h = ij2k(j, i, nb_points);
					}
				aij = bitvector_s_i(bitvector_adjacency, h);
				}
			else if (f_has_is_adjacent_callback) {
				aij = (*is_adjacent_callback)(i, j,
						is_adjacent_callback_data);
				}
			else {
				cout << "write_colored_graph cannot "
						"determine adjacency" << endl;
				}
			if (aij) {
				ost << setw(w) << j + point_offset << " ";
				}
			}
		ost << endl;
	
		
		}
	
	if (f_colors) {
		ost << endl;
		for (j = 0; j < nb_colors; j++) {
			d = 0;
			for (i = 0; i < nb_points; i++) {
				if (point_color[i] == j)
					d++;
				}
			ost << setw(w) << j + point_offset << " "
					<< setw(w) << d << " ";
			for (i = 0; i < nb_points; i++) {
				if (point_color[i] == j)
					ost << setw(w) << i + point_offset << " ";
				}
			ost << endl;
			}
		}
	
	if (f_point_labels) {
		ost << endl;
		for (i = 0; i < nb_points; i++) {
			ost << setw(w) << i + point_offset << " " 
				<< setw(6) << point_label[i] << endl;
			}
		}

	ost << "</GRAPH>" << endl;
	
}


int str2int(string &str)
{
	int i, res, l;
	
	l = (int) str.length();
	res = 0;
	for (i = 0; i < l; i++) {
		res = (res * 10) + (str[i] - 48);
		}
	return res;
}

void print_longinteger_after_multiplying(
		ostream &ost, int *factors, int len)
{
	longinteger_domain D;
	longinteger_object a;

	D.multiply_up(a, factors, len);
	ost << a;
}

void andre_preimage(projective_space *P2, projective_space *P4, 
	int *set2, int sz2, int *set4, int &sz4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	finite_field *FQ;
	finite_field *Fq;
	int /*Q,*/ q;
	int *v, *w1, *w2, *w3, *v2;
	int *components;
	int *embedding;
	int *pair_embedding;
	int i, h, k, a, a0, a1, b, b0, b1, e, alpha;

	if (f_v) {
		cout << "andre_preimage" << endl;
		}
	FQ = P2->F;
	//Q = FQ->q;
	alpha = FQ->p;
	if (f_vv) {
		cout << "alpha=" << alpha << endl;
		//FQ->print(TRUE /* f_add_mult_table */);
		}

	
	Fq = P4->F;
	q = Fq->q;
	
	v = NEW_int(3);
	w1 = NEW_int(5);
	w2 = NEW_int(5);
	w3 = NEW_int(5);
	v2 = NEW_int(2);
	e = P2->F->e >> 1;
	if (f_vv) {
		cout << "e=" << e << endl;
		}

	FQ->subfield_embedding_2dimensional(*Fq, 
		components, embedding, pair_embedding, verbose_level - 3);

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

	if (f_vv) {
		FQ->print_embedding(*Fq, 
			components, embedding, pair_embedding);
		}


	sz4 = 0;
	for (i = 0; i < sz2; i++) {
		if (f_vv) {
			cout << "input point " << i << " : ";
			}
		P2->unrank_point(v, set2[i]);
		FQ->PG_element_normalize(v, 1, 3);
		if (f_vv) {
			int_vec_print(cout, v, 3);
			cout << " becomes ";
			}

		if (v[2] == 0) {

			// we are dealing with a point on the
			// line at infinity.
			// Such a point corresponds to a line of the spread. 
			// We create the line and then create all
			// q + 1 points on that line.
			
			if (f_vv) {
				cout << endl;
				}
			// w1[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2]
			// w2[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2] * alpha
			// where v[2] runs through the points of PG(1,q^2). 
			// That way, w1[4] and w2[4] are a GF(q)-basis for the 
			// 2-dimensional subspace v[2] (when viewed over GF(q)), 
			// which is an element of the regular spread.
						
			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				b = FQ->mult(a, alpha);
				b0 = components[b * 2 + 0];
				b1 = components[b * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				w2[2 * h + 0] = b0;
				w2[2 * h + 1] = b1;
				}
			if (FALSE) {
				cout << "w1=";
				int_vec_print(cout, w1, 4);
				cout << "w2=";
				int_vec_print(cout, w2, 4);
				cout << endl;
				}
			
			// now we create all points on the line
			// spanned by w1[4] and w2[4]:
			// There are q + 1 of these points.
			// We make sure that the coordinate vectors
			// have a zero in the last spot.
			
			for (h = 0; h < q + 1; h++) {
				Fq->PG_element_unrank_modified(v2, 1, 2, h);
				if (FALSE) {
					cout << "v2=";
					int_vec_print(cout, v2, 2);
					cout << " : ";
					}
				for (k = 0; k < 4; k++) {
					w3[k] = Fq->add(Fq->mult(v2[0], w1[k]),
							Fq->mult(v2[1], w2[k]));
					}
				w3[4] = 0;
				if (f_vv) {
					cout << " ";
					int_vec_print(cout, w3, 5);
					}
				a = P4->rank_point(w3);
				if (f_vv) {
					cout << " rank " << a << endl;
					}
				set4[sz4++] = a;
				}
			}
		else {

			// we are dealing with an affine point:
			// We make sure that the coordinate vector
			// has a zero in the last spot.


			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				}
			w1[4] = 1;
			if (f_vv) {
				//cout << "w1=";
				int_vec_print(cout, w1, 5);
				}
			a = P4->rank_point(w1);
			if (f_vv) {
				cout << " rank " << a << endl;
				}
			set4[sz4++] = a;
			}
		}
	if (f_v) {
		cout << "we found " << sz4 << " points:" << endl;	
		int_vec_print(cout, set4, sz4);
		cout << endl;
		P4->print_set(set4, sz4);
		for (i = 0; i < sz4; i++) {
			cout << set4[i] << " ";
			}
		cout << endl;
		}


	FREE_int(components);
	FREE_int(embedding);
	FREE_int(pair_embedding);
	if (f_v) {
		cout << "andre_preimage done" << endl;
		}
}

void determine_conic(int q, const char *override_poly,
		int *input_pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	finite_field F;
	projective_space *P;
	//int f_basis = TRUE;
	//int f_semilinear = TRUE;
	//int f_with_group = FALSE;
	int v[3];
	int len = 3;
	int six_coeffs[6];
	int i;

	if (f_v) {
		cout << "determine_conic q=" << q << endl;
		cout << "input_pts: ";
		int_vec_print(cout, input_pts, nb_pts);
		cout << endl;
		}
	F.init_override_polynomial(q, override_poly, verbose_level);

	P = NEW_OBJECT(projective_space);
	if (f_vv) {
		cout << "determine_conic before P->init" << endl;
		}
	P->init(len - 1, &F, 
		FALSE, 
		verbose_level - 2/*MINIMUM(2, verbose_level)*/);

	if (f_vv) {
		cout << "determine_conic after P->init" << endl;
		}
	P->determine_conic_in_plane(input_pts, nb_pts,
			six_coeffs, verbose_level - 2);

	if (f_v) {
		cout << "determine_conic the six coefficients are ";
		int_vec_print(cout, six_coeffs, 6);
		cout << endl;
		}
	
	int points[1000];
	int nb_points;
	//int v[3];
	
	P->conic_points(input_pts, six_coeffs,
			points, nb_points, verbose_level - 2);
	if (f_v) {
		cout << "the " << nb_points << " conic points are: ";
		int_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			int_vec_print(cout, v, 3);
			cout << endl;
			}
		}
	FREE_OBJECT(P);
}

void compute_decomposition_of_graph_wrt_partition(
	int *Adj, int N,
	int *first, int *len, int nb_parts, int *&R,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J, i, j, f1, l1, f2, l2, r0 = 0, r;

	if (f_v) {
		cout << "compute_decomposition_of_graph_wrt_partition" << endl;
		cout << "The partition is:" << endl;
		cout << "first = ";
		int_vec_print(cout, first, nb_parts);
		cout << endl;
		cout << "len = ";
		int_vec_print(cout, len, nb_parts);
		cout << endl;
		}
	R = NEW_int(nb_parts * nb_parts);
	int_vec_zero(R, nb_parts * nb_parts);
	for (I = 0; I < nb_parts; I++) {
		f1 = first[I];
		l1 = len[I];
		for (J = 0; J < nb_parts; J++) {
			f2 = first[J];
			l2 = len[J];
			for (i = 0; i < l1; i++) {
				r = 0;
				for (j = 0; j < l2; j++) {
					if (Adj[(f1 + i) * N + f2 + j]) {
						r++;
						}
					}
				if (i == 0) {
					r0 = r;
					}
				else {
					if (r0 != r) {
						cout << "compute_decomposition_of_graph_"
							"wrt_partition not tactical" << endl;
						cout << "I=" << I << endl;
						cout << "J=" << J << endl;
						cout << "r0=" << r0 << endl;
						cout << "r=" << r << endl;
						exit(1); 
						}
					}
				}
			R[I * nb_parts + J] = r0;
			}
		}
	if (f_v) {
		cout << "compute_decomposition_of_graph_wrt_partition done" << endl;
		}
}

void int_vec_print_classified(int *v, int len)
{
	classify C;

	C.init(v, len, FALSE /*f_second */, 0);
	C.print(TRUE /* f_backwards*/);
	cout << endl;
}

void create_Levi_graph_from_incidence_matrix(
	colored_graph *&CG, int *M, int nb_rows, int nb_cols,
	int f_point_labels, int *point_labels,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	uchar *bitvector_adjacency;
	int L, /*bitvector_length_in_bits,*/ bitvector_length;
	int i, j, k, r, c;
	int N;
	
	if (f_v) {
		cout << "create_Levi_graph_from_incidence_matrix" << endl;
		}
	CG = NEW_OBJECT(colored_graph);

	N = nb_rows + nb_cols;
	L = (N * (N - 1)) >> 1;

	//bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	bitvector_adjacency = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
		}


	for (r = 0; r < nb_rows; r++) {
		i = r;
		for (c = 0; c < nb_cols; c++) {
			if (M[r * nb_cols + c]) {
				j = nb_rows + c;
				k = ij2k(i, j, N);
				bitvector_m_ii(bitvector_adjacency, k, 1);
				}
			}
		}

	if (f_point_labels) {
		CG->init_with_point_labels(N, 1 /* nb_colors */, 
			NULL /*point_color*/, bitvector_adjacency,
			TRUE, point_labels, verbose_level - 2);
			// the adjacency becomes part of the colored_graph object
		}
	else {
		CG->init(N, 1 /* nb_colors */, 
			NULL /*point_color*/, bitvector_adjacency,
			TRUE, verbose_level - 2);
			// the adjacency becomes part of the colored_graph object
		}

	if (f_v) {
		cout << "create_Levi_graph_from_incidence_matrix done" << endl;
		}
}

}
}


