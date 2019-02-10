// make_something.C
// 
// Anton Betten
// June 25, 2011
//
// 
//
//
//

#include "orbiter.h"

using namespace orbiter;
using namespace orbiter::top_level;





// global data:

int t0; // the system time when the program started




int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q;
	int f_n = FALSE;
	int n;
	int f_poly = FALSE;
	const char *poly = NULL;
	int f_Q = FALSE;
	int Q;
	int f_poly_Q = FALSE;
	const char *poly_Q = NULL;

	int f_subiaco_oval = FALSE;
	int f_short = FALSE;
	int f_subiaco_hyperoval = FALSE;
	int f_adelaide_hyperoval = FALSE;

	int f_hyperoval = FALSE;
	int f_translation = FALSE;
	int translation_exponent = 0;
	int f_Segre = FALSE;
	int f_Payne = FALSE;
	int f_Cherowitzo = FALSE;
	int f_OKeefe_Penttila = FALSE;

	int f_BLT_database = FALSE;
	int BLT_k = 0;
	int f_BLT_in_PG = FALSE;
	
	int f_BLT_Linear = FALSE;
	int f_BLT_Fisher = FALSE;
	int f_BLT_Mondello = FALSE;
	int f_BLT_FTWKB = FALSE;

	int f_ovoid = FALSE;

	int f_Baer = FALSE;
	
	int f_orthogonal = FALSE;
	int orthogonal_epsilon = 0;

	int f_hermitian = FALSE;

	int f_cubic = FALSE; // twisted cubic in PG(2,q)
	int f_twisted_cubic = FALSE; // twisted cubic in PG(3,q)

	int f_elliptic_curve = FALSE;
	int elliptic_curve_b = 0;
	int elliptic_curve_c = 0;

	int f_Hill_cap_56 = FALSE;

	int f_ttp_code = FALSE;
	int f_ttp_construction_A = FALSE;
	int f_ttp_hyperoval = FALSE;
	int f_ttp_construction_B = FALSE;

	int f_unital_XXq_YZq_ZYq = FALSE;

	int f_desarguesian_line_spread_in_PG_3_q = FALSE;
	int f_embedded_in_PG_4_q = FALSE;

	int f_Buekenhout_Metz = FALSE;
	int f_classical = FALSE;
	int f_Uab = FALSE;
	int parameter_a = 0;
	int parameter_b = 0;

	int f_whole_space = FALSE;
	int f_hyperplane = FALSE;
	int pt = 0;
	
	int f_segre_variety = FALSE;
	int segre_variety_a;
	int segre_variety_b;

	int f_Maruta_Hamada_arc = FALSE;

	int f_projective_variety = FALSE;
	const char *variety_label = NULL;
	int variety_degree = 0;
	const char *variety_coeffs = NULL;


	int f_projective_curve = FALSE;
	const char *curve_label = NULL;
	int curve_nb_vars = 0;
	int curve_degree = 0;
	const char *curve_coeffs = NULL;


	t0 = os_ticks();



 	

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-Q") == 0) {
			f_Q = TRUE;
			Q = atoi(argv[++i]);
			cout << "-Q " << Q << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-hyperoval") == 0) {
			f_hyperoval = TRUE;
			cout << "-hyperoval " << endl;
			}
		else if (strcmp(argv[i], "-subiaco_oval") == 0) {
			f_subiaco_oval = TRUE;
			f_short = atoi(argv[++i]);
			cout << "-subiaco_oval " << f_short << endl;
			}
		else if (strcmp(argv[i], "-subiaco_hyperoval") == 0) {
			f_subiaco_hyperoval = TRUE;
			cout << "-subiaco_hyperoval " << endl;
			}
		else if (strcmp(argv[i], "-adelaide_hyperoval") == 0) {
			f_adelaide_hyperoval = TRUE;
			cout << "-adelaide_hyperoval " << endl;
			}
		else if (strcmp(argv[i], "-translation") == 0) {
			f_translation = TRUE;
			translation_exponent = atoi(argv[++i]);
			cout << "-translation " << translation_exponent << endl;
			}
		else if (strcmp(argv[i], "-Segre") == 0 || strcmp(argv[i], "-segre") == 0) {
			f_Segre = TRUE;
			cout << "-segre" << endl;
			}
		else if (strcmp(argv[i], "-Payne") == 0 || strcmp(argv[i], "-payne") == 0) {
			f_Payne = TRUE;
			cout << "-Payne" << endl;
			}
		else if (strcmp(argv[i], "-Cherowitzo") == 0 || strcmp(argv[i], "-cherowitzo") == 0) {
			f_Cherowitzo = TRUE;
			cout << "-Cherowitzo" << endl;
			}
		else if (strcmp(argv[i], "-OKeefe_Penttila") == 0) {
			f_OKeefe_Penttila = TRUE;
			cout << "-OKeefe_Penttila" << endl;
			}


		else if (strcmp(argv[i], "-BLT_database") == 0) {
			f_BLT_database = TRUE;
			BLT_k = atoi(argv[++i]);
			cout << "-BLT_database " << BLT_k << endl;
			}
		else if (strcmp(argv[i], "-BLT_in_PG") == 0) {
			f_BLT_in_PG = TRUE;
			cout << "-BLT_in_PG " << endl;
			}

		else if (strcmp(argv[i], "-BLT_Linear") == 0) {
			f_BLT_Linear = TRUE;
			cout << "-BLT_Linear " << endl;
			}
		else if (strcmp(argv[i], "-BLT_Fisher") == 0) {
			f_BLT_Fisher = TRUE;
			cout << "-BLT_Fisher " << endl;
			}
		else if (strcmp(argv[i], "-BLT_Mondello") == 0) {
			f_BLT_Mondello = TRUE;
			cout << "-BLT_Mondello " << endl;
			}
		else if (strcmp(argv[i], "-BLT_FTWKB") == 0) {
			f_BLT_FTWKB = TRUE;
			cout << "-BLT_FTWKB " << endl;
			}

		else if (strcmp(argv[i], "-ovoid") == 0) {
			f_ovoid = TRUE;
			cout << "-ovoid " << endl;
			}
		else if (strcmp(argv[i], "-Baer") == 0) {
			f_Baer = TRUE;
			cout << "-Baer " << endl;
			}
		else if (strcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = TRUE;
			orthogonal_epsilon = atoi(argv[++i]);
			cout << "-orthogonal " << orthogonal_epsilon << endl;
			}
		else if (strcmp(argv[i], "-hermitian") == 0) {
			f_hermitian = TRUE;
			cout << "-hermitian" << endl;
			}
		else if (strcmp(argv[i], "-cubic") == 0) {
			f_cubic = TRUE;
			cout << "-cubic " << endl;
			}
		else if (strcmp(argv[i], "-twisted_cubic") == 0) {
			f_twisted_cubic = TRUE;
			cout << "-twisted_cubic " << endl;
			}
		else if (strcmp(argv[i], "-elliptic_curve") == 0) {
			f_elliptic_curve = TRUE;
			elliptic_curve_b = atoi(argv[++i]);
			elliptic_curve_c = atoi(argv[++i]);
			cout << "-elliptic_curve " << elliptic_curve_b
					<< " " << elliptic_curve_c << endl;
			}
		else if (strcmp(argv[i], "-Hill_cap_56") == 0) {
			f_Hill_cap_56 = TRUE;
			cout << "-Hill_cap_56 " << endl;
			}
		else if (strcmp(argv[i], "-ttp_construction_A") == 0) {
			f_ttp_code = TRUE;
			f_ttp_construction_A = TRUE;
			cout << "-ttp_construction_A" << endl;
			}
		else if (strcmp(argv[i], "-ttp_construction_A_hyperoval") == 0) {
			f_ttp_code = TRUE;
			f_ttp_construction_A = TRUE;
			f_ttp_hyperoval = TRUE;
			cout << "-ttp_construction_A_hyperoval" << endl;
			}
		else if (strcmp(argv[i], "-ttp_construction_B") == 0) {
			f_ttp_code = TRUE;
			f_ttp_construction_B = TRUE;
			cout << "-ttp_construction_B" << endl;
			}
		else if (strcmp(argv[i], "-unital_XXq_YZq_ZYq") == 0) {
			f_unital_XXq_YZq_ZYq = TRUE;
			cout << "-unital_XXq_YZq_ZYq" << endl;
			}
		else if (strcmp(argv[i], "-desarguesian_line_spread_in_PG_3_q") == 0) {
			f_desarguesian_line_spread_in_PG_3_q = TRUE;
			cout << "-desarguesian_line_spread_in_PG_3_q" << endl;
			}
		else if (strcmp(argv[i], "-embedded_in_PG_4_q") == 0) {
			f_embedded_in_PG_4_q = TRUE;
			cout << "-embedded_in_PG_4_q" << endl;
			}
		else if (strcmp(argv[i], "-Buekenhout_Metz") == 0) {
			f_Buekenhout_Metz = TRUE;
			cout << "-Buekenhout_Metz " << endl;
			}
		else if (strcmp(argv[i], "-classical") == 0) {
			f_classical = TRUE;
			cout << "-classical " << endl;
			}
		else if (strcmp(argv[i], "-Uab") == 0) {
			f_Uab = TRUE;
			parameter_a = atoi(argv[++i]);
			parameter_b = atoi(argv[++i]);
			cout << "-Uab " << parameter_a << " " << parameter_b << endl;
			}
		else if (strcmp(argv[i], "-whole_space") == 0) {
			f_whole_space = TRUE;
			cout << "-whole_space " << endl;
			}
		else if (strcmp(argv[i], "-hyperplane") == 0) {
			f_hyperplane = TRUE;
			pt = atoi(argv[++i]);
			cout << "-hyperplane " << pt << endl;
			}
		else if (strcmp(argv[i], "-segre_variety") == 0) {
			f_segre_variety = TRUE;
			segre_variety_a = atoi(argv[++i]);
			segre_variety_b = atoi(argv[++i]);
			cout << "-segre_variety " << segre_variety_a
					<< " " << segre_variety_b << endl;
			}
		else if (strcmp(argv[i], "-Maruta_Hamada_arc") == 0) {
			f_Maruta_Hamada_arc = TRUE;
			cout << "-Maruta_Hamada_arc " << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-poly_Q") == 0) {
			f_poly_Q = TRUE;
			poly_Q = argv[++i];
			cout << "-poly_Q " << poly_Q << endl;
			}
		else if (strcmp(argv[i], "-projective_variety") == 0) {
			f_projective_variety = TRUE;
			variety_label = argv[++i];
			variety_degree = atoi(argv[++i]);
			variety_coeffs = argv[++i];
			cout << "-projective_variety "
					<< variety_label << " "
					<< variety_degree << " "
					<< variety_coeffs << endl;
			}
		else if (strcmp(argv[i], "-projective_curve") == 0) {
			f_projective_curve = TRUE;
			curve_label = argv[++i];
			curve_nb_vars = atoi(argv[++i]);
			curve_degree = atoi(argv[++i]);
			curve_coeffs = argv[++i];
			cout << "-projective_curve "
					<< curve_label << " "
					<< curve_nb_vars << " "
					<< curve_degree << " "
					<< curve_coeffs << endl;
			}
		}
	
	if (!f_q) {
		cout << "please specify the field order "
				"using the option -q <q>" << endl;
		exit(1);
		}

	char fname[1000];
	int nb_pts;
	int *Pts = NULL;

	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0);
	
	if (f_hyperoval) {
		create_hyperoval(F, 
			f_translation, translation_exponent, 
			f_Segre, f_Payne, f_Cherowitzo, f_OKeefe_Penttila, 
			fname, nb_pts, Pts, 
			verbose_level);
			// ACTION/geometric_object.C

		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);

		}
	else if (f_subiaco_oval) {
		create_subiaco_oval(F, 
			f_short, 
			fname, nb_pts, Pts, 
			verbose_level);
			// ACTION/geometric_object.C


		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);


		}
	else if (f_subiaco_hyperoval) {
		create_subiaco_hyperoval(F, 
			fname, nb_pts, Pts, 
			verbose_level);
			// ACTION/geometric_object.C


		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);

		}
	else if (f_adelaide_hyperoval) {

		finite_field *FQ;
		subfield_structure *S;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Q, poly_Q, 0);

		S = NEW_OBJECT(subfield_structure);
		S->init(FQ, F, verbose_level);
		
		create_adelaide_hyperoval(S, 
			fname, nb_pts, Pts, 
			verbose_level);
			// ACTION/geometric_object.C


		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);

		
		FREE_OBJECT(S);
		FREE_OBJECT(FQ);
		}
	else if (f_BLT_database) {
		create_BLT_from_database(f_BLT_in_PG /* f_embedded */, F, BLT_k, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
#if 0
	else if (f_BLT_Linear) {
		create_BLT(f_BLT_in_PG /* f_embedded */, F, 
			TRUE /* f_Linear */, 
			FALSE /* f_Fisher */, 
			FALSE /* f_Mondello */, 
			FALSE /* f_FTWKB */, 
			f_poly, poly, 
			f_poly_Q, poly_Q, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_BLT_Fisher) {
		create_BLT(f_BLT_in_PG /* f_embedded */, q, 
			FALSE /* f_Linear */, 
			TRUE /* f_Fisher */, 
			FALSE /* f_Mondello */, 
			FALSE /* f_FTWKB */, 
			f_poly, poly, 
			f_poly_Q, poly_Q, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_BLT_Mondello) {
		create_BLT(f_BLT_in_PG /* f_embedded */, q, 
			FALSE /* f_Linear */, 
			FALSE /* f_Fisher */, 
			TRUE /* f_Mondello */, 
			FALSE /* f_FTWKB */, 
			f_poly, poly, 
			f_poly_Q, poly_Q, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_BLT_FTWKB) {
		create_BLT(f_BLT_in_PG /* f_embedded */, q, 
			FALSE /* f_Linear */, 
			FALSE /* f_Fisher */, 
			FALSE /* f_Mondello */, 
			TRUE /* f_FTWKB */, 
			f_poly, poly, 
			f_poly_Q, poly_Q, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
#endif
	else if (f_ovoid) {
		create_ovoid(F, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_Baer) {
		if (!f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}

		if (!f_Q) {
			cout << "please specify the field order "
					"using the option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Q, poly_Q, 0);

		create_Baer_substructure(n, FQ, F, 
			fname, nb_pts, Pts, 
			verbose_level);
		FREE_OBJECT(FQ);
		}
	else if (f_orthogonal) {
		if (!f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		create_orthogonal(orthogonal_epsilon, n, F, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_hermitian) {
		if (!f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		create_hermitian(n, F, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_cubic) {
		create_cubic(F,
			fname, nb_pts, Pts,
			verbose_level);
		}
	else if (f_twisted_cubic) {
		create_twisted_cubic(F, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_elliptic_curve) {
		create_elliptic_curve(F,
			elliptic_curve_b, elliptic_curve_c,
			fname, nb_pts, Pts,
			verbose_level);
		}
	else if (f_Hill_cap_56) {
		Hill_cap56(argc, argv, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_ttp_code) {

		if (!f_Q) {
			cout << "please specify the field order "
					"using the option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Q, poly_Q, 0);
		create_ttp_code(FQ, F, 
			f_ttp_construction_A, f_ttp_hyperoval, f_ttp_construction_B, 
			fname, nb_pts, Pts, 
			verbose_level);
		FREE_OBJECT(FQ);
		}
	else if (f_unital_XXq_YZq_ZYq) {
		create_unital_XXq_YZq_ZYq(F, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_desarguesian_line_spread_in_PG_3_q) {

		if (!f_Q) {
			cout << "please specify the field order "
					"using the option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Q, poly_Q, 0);

		create_desarguesian_line_spread_in_PG_3_q(FQ, F, 
			f_embedded_in_PG_4_q,
			fname, nb_pts, Pts, 
			verbose_level);
		FREE_OBJECT(FQ);

		}
	else if (f_Buekenhout_Metz) {

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Q, poly_Q, 0);

		create_Buekenhout_Metz(F, FQ, 
			f_classical, f_Uab, parameter_a, parameter_b, 
			fname, nb_pts, Pts, 
			verbose_level);
		
		}
	else if (f_whole_space) {
		if (!f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		create_whole_space(n, F, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_hyperplane) {
		if (!f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		create_hyperplane(n, F, 
			pt, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_segre_variety) {
		create_segre_variety(F, segre_variety_a, segre_variety_b, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_Maruta_Hamada_arc) {
		create_Maruta_Hamada_arc(F, 
			fname, nb_pts, Pts, 
			verbose_level);
		}
	else if (f_projective_variety) {
		F->create_projective_variety(
				variety_label,
				n + 1, variety_degree,
				variety_coeffs,
				fname, nb_pts, Pts,
				verbose_level);
	}
	else if (f_projective_curve) {
		F->create_projective_curve(
				curve_label,
				curve_nb_vars, curve_degree,
				curve_coeffs,
				fname, nb_pts, Pts,
				verbose_level);
	}
	else {
		cout << "nothing to create" << endl;
		exit(1);
		}


	cout << "we created a set of " << nb_pts << " points, "
			"and we will write it to the file " << fname << endl;
	cout << "list of points:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;

	write_set_to_file(fname, Pts, nb_pts, verbose_level);
	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;


	if (Pts) {
		FREE_int(Pts);
		}
	the_end(t0);
}



