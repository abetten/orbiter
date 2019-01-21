// tdo_can.C
// Anton Betten
// Abdullah Al-Azemi
//
// started:  Dec 18 2006

#include "orbiter.h"

using namespace orbiter;


int t0;

char buf[BUFSIZE];

int main(int argc, char **argv)
{

	int verbose_level = 0;
	int f_transpose;
	//int f_nauty = FALSE;
	char *fname_in;
	char str[1000];
	char ext[1000];
	char fname_out[1000];
	char fname_out2[1000];

	int nb_V, *V;
	int nb_B, *B;
	int *row_scheme;
	int *col_scheme;
	
	int a, b, c, ra, r;
	int m, n, i, j, cnt = 0;
	longinteger_object ago;
	int nb_agos;
	longinteger_object *agos;
	int *multiplicities;
	char *p_buf;
	

	t0 = os_ticks();

	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-transpose") == 0) {
			f_transpose = TRUE;
			cout << "-transpose " << endl;
		}
	}
	fname_in = argv[argc - 1];

	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	cout << "opening file " << fname_in << " for reading" << endl;
	ifstream f(fname_in);
	
	strcpy(str, fname_in);
	get_extension_if_present(str, ext);
	chop_off_extension_if_present(str, ext);
	if (f_transpose) {
		sprintf(fname_out, "%st.tdo", str);
		sprintf(fname_out2, "%st.tdov", str);
		}
	else {
		sprintf(fname_out, "%sc.tdo", str);
		sprintf(fname_out2, "%sc.tdov", str);
		}
	{
	cout << "opening file " << fname_out << " for writing" << endl;
	ofstream g(fname_out);
	cout << "opening file " << fname_out2 << " for writing" << endl;
	ofstream TDOV(fname_out2);


	longinteger_collect_setup(nb_agos, agos, multiplicities);
	
	
	while (TRUE) {


		//partition_backtrack PB;


		if (f.eof()) {
			break;
			}
		f.getline(buf, BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			continue;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;

		s_scan_int(&p_buf, &m);
		if (m == -1) {
			cout << "\nfound a complete file with " << cnt << " solutions" << endl;
			break;
			}
		s_scan_int(&p_buf, &n);
		s_scan_int(&p_buf, &a); nb_V = a;
		s_scan_int(&p_buf, &a); nb_B = a;
		V = new int[nb_V];
		B = new int[nb_B];
		row_scheme = new int[nb_V * nb_B];
		
		for (i = 0; i < nb_V; i++) {
			s_scan_int(&p_buf, &a); V[i] = a;
		}
		for (j = 0; j < nb_B; j++) {
			s_scan_int(&p_buf, &a); B[j] = a;
		}
		for (i = 0; i < nb_V; i++) {
			for (j = 0; j < nb_B; j++) {
				s_scan_int(&p_buf, &a); row_scheme[i * nb_B + j] = a;
			}
		}
		if (f_transpose) {
			col_scheme = new int[nb_B * nb_V];
			for (i = 0; i < nb_B; i++) {
				for (j = 0; j < nb_V; j++) {
					r = row_scheme[j * nb_B + i];
					a = V[j];
					b = B[i];
					ra = r * a;
					c = ra / b;
					if (c * b != ra) {
						cout << "cannot divide, maybe the TDO wasn't tactical?" << endl;
						cout << "cnt=" << cnt << endl;
						cout << "i=" << i << endl;
						cout << "j=" << j << endl;
						cout << "B[i]=b=" << b << " = " << B[i] << endl;
						cout << "V[j]=a=" << a << " = " << V[j] << endl;
						cout << "r=" << r << endl;
						cout << "ra=" << ra << endl;
						exit(1);
						}
					col_scheme[i * nb_V + j] = c;
					}
				}
			g << n << " " << m << " " << nb_B << " " << nb_V;
			for (i = 0; i < nb_B; i++) {
				g << " " << B[i];
				}
			for (j = 0; j < nb_V; j++) {
				g << " " << V[j];
				}
			for (i = 0; i < nb_B; i++) {
				for (j = 0; j < nb_V; j++) {
					g << " " << col_scheme[i * nb_V + j];
					}
				}
			g << endl;
			
			TDOV << "TDO no " << cnt << " with " 
				<< nb_B << " row classes and " 
				<< nb_V << " column classes " << endl;
			TDOV << "     | ";
			for (j = 0; j < nb_V; j++) {
				TDOV << setw(4) << V[j];
			}
			TDOV << endl;
			TDOV << "-------";
			for (j = 0; j < nb_V; j++) {
				TDOV << "----";
			}
			TDOV << endl;
			for (i = 0; i < nb_B; i++) {
				TDOV << setw(4) << B[i] << " | ";
				for (j = 0; j < nb_V; j++) {
					TDOV << setw(4) << col_scheme[i * nb_V + j];
				}
				TDOV << endl;
			}
			TDOV << endl;
			TDOV << endl;
			
			delete [] col_scheme;
			}
		else {
			}
		
		delete [] row_scheme;
		delete [] V;
		delete [] B;
		
		
		cnt++;
		
		} // while (TRUE)
	cout << "we found the following automorphism group orders: " << endl;
	longinteger_collect_print(cout, nb_agos, agos, multiplicities);
	longinteger_collect_free(nb_agos, agos, multiplicities);

	g << "-1 " << cnt << endl;
	
	} // close ofstream


	cout << endl << endl;
	cout<< "Mashaa ALLAH " << endl << endl;
	cout<< "The program has been working for ";
	time_check(cout, t0);
	cout << " to get all the result.\n" << endl << endl;

	return 0;
}

#if 0
int int_vec_compare_interface(void *a, void *b, void *data)
{
	int *A, *B, nb_inc;
	
	A = (int *) a;
	B = (int *) b;
	nb_inc = (int) data;
	//cout << "compare " << nb_inc << endl;
	return int_vec_compare(A, B, nb_inc);
}
#endif

