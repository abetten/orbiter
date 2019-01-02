// cc2widor.C
// Anton Betten
//
// converts a "cc" file to a "widor" file
//
// started:  June 29, 2008

#include "orbiter.h"

void convert(ifstream &f, ofstream &g, int f_simple);

int main(int argc, char **argv)
{
	int i;
	char *cc_fname;
	char str[1000];
	char ext[1000];
	char fname_out[1000];
	//char label[1000];
	int verbose_level = 0;
	int f_simple = FALSE;
		
	if (argc <= 1) {
		//print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		if (strcmp(argv[i], "-simple") == 0) {
			f_simple = TRUE;
			cout << "-simple " << endl;
			}
		}
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	cc_fname = argv[argc - 1];
	strcpy(str, cc_fname);
	get_extension_if_present(str, ext);
	chop_off_extension_if_present(str, ext);
	sprintf(fname_out, "%s.widor", str);
	{
	ifstream f(cc_fname);
	ofstream g(fname_out);
	for (i = 0; ; i++) {
		if (f.eof())
			break;
		convert(f, g, f_simple);
		}
	}
}


void convert(ifstream &f, ofstream &g, int f_simple)
{
	//char type[64];
	//FILE *fp;
	int b_len = 0, *B = NULL, b = 0;
	int v_len = 0, *V = NULL, v = 0;
	int *tdo = NULL;
	int *s_type = NULL, *s_flag = NULL;
	int *r_type = NULL, *r_from = NULL, *r_len = NULL;
	int *f_flush = NULL;
	int nb_fuse = 0, *fuse = NULL;
	int i, j, ff, l, no;
	int f_v = FALSE;
	int f_vv = FALSE;
	int f_do_iso_test = FALSE;
	int f_do_aut_group = FALSE;
	int f_transpose = FALSE;
	//int gen_print_interval = 1;
	//int f_resolve = FALSE;
	//int f_resolve_with_aut_group = FALSE;
	//int f_lambda = FALSE;
	//int lambda = FALSE;
	//int f_find_square = TRUE; /* JS 120100 */
	//int f_simple = FALSE; /* JS 180100 */
	//ISO_TYPE *it;
	string str;
	char buf[100000];

	//inc_file_name[0] = 0;
	//GEO_fname[0] = 0;
	while (TRUE) {
		if (f.eof())
			return;
		f.getline(buf, 100000, '\n');
		if (buf[0] == '#')
			continue;
		if (strncmp(buf, "STARTINGPOINT", 13) == 0) {
			sscanf(buf + 13, "%d", &no);
			printf("found STARTINGPOINT %d\n", no);
			printf("%s\n", buf);
			break;				
			}
		}
	while (TRUE) {
		f >> str;
		l = str.size();
		for (i = 0; i < l; i++)
			buf[i] = str[i];
		buf[l] = 0;
		
		if (buf[0] == '#') {
			printf("%s\n", buf);
			continue;
			}
		
		if (strncmp(buf, "B", 1) == 0) {
			f >> str;
			b_len = str2int(str);
			B = new int[b_len];
			b = 0;
			for (i = 0; i < b_len; i++) {
				f >> B[i];
				b += B[i];
				}
			printf("B: ");
			for (i = 0; i < b_len; i++) {
				printf("%d ", B[i]);
				}
			printf(" (b = %d)\n", b);
			}
		
		if (strncmp(buf, "V", 1) == 0) {
			f >> str;
			v_len = str2int(str);
			V = new int[v_len];
			v = 0;
			for (i = 0; i < v_len; i++) {
				f >> V[i];
				v += V[i];
				}
			printf("V: ");
			for (i = 0; i < v_len; i++) {
				printf("%d ", V[i]);
				}
			printf(" (v = %d)\n", v);
			
			
			s_type = new int[v + 1];
			s_flag = new int[v + 1];
			for (i = 0; i <= v; i++) {
				s_type[i] = 0;
				s_flag[i] = 0;
				}
			s_type[v - 1] = 1;
			s_flag[v - 1] = 0; //s_flag_opt_zahl;
			/* urspruenglicher Wert FTFF (JS 120100) */
			
			r_type = new int[v + 1];
			r_from = new int[v + 1];
			r_len = new int[v + 1];
			f_flush = new int[v + 1];
			for (i = 0; i <= v; i++) {
				r_type[i] = 0;
				r_from[i] = 0;
				r_len[i] = 0;
				f_flush[i] = 0;
				}
			}
		
		if (strncmp(buf, "FUSE", 4) == 0) {
			if (v_len == 0) {
				printf("FUSE: V not specified !\n");
				exit(1);
				}
			f >> str;
			nb_fuse = str2int(str);
			fuse = new int[nb_fuse];
			ff = 0;
			for (i = 0; i < nb_fuse; i++) {
				f >> fuse[i];
				ff += fuse[i];
				}
			if (ff != v_len) {
				printf("FUSE: ff != v_len !\n");
				exit(1);
				}
			printf("fuse: ");
			for (i = 0; i < nb_fuse; i++) {
				printf("%d ", fuse[i]);
				}
			printf("\n");
			}
		
#if 0
		if (strncmp(buf, "FLUSH", 5) == 0) {
			sscanf(buf + 6, "%d ", &i);
			f_flush[i - 1] = TRUE;
			printf("FLUSH at line %d\n", i);
			}
		
		if (strncmp(buf, "R", 1) == 0) {
			sscanf(buf + 1, "%d %d %d", &i, &ff, &l);
			r_type[i - 1] = 1;
			r_from[i - 1] = ff;
			r_len[i - 1] = l;
			printf("R: %d [%d-%d]\n", i, ff, ff + l - 1);
			}
		
		if (strncmp(buf, "S1", 2) == 0 || 
			strncmp(buf, "S2", 2) == 0) {
			sscanf(buf + 2, "%d %s", &l, type);
			t = 0;
			if (type[0] == 'T')
				t += 8;
			if (type[1] == 'T')
				t += 4;
			if (type[2] == 'T')
				t += 2;
			if (type[3] == 'T')
				t += 1;
			if (buf[1] == '1')
				s_type[l - 1] = 1;
			else
				s_type[l - 1] = 2;
			s_flag[l - 1] = t;
			printf("S%d: line %d type %s = %d\n", s_type[l - 1], l, type, t);
			}
#endif

		if (strncmp(buf, "TDO", 3) == 0) {
			if (b_len == 0) {
				printf("B not specified !\n");
				exit(1);
				}
			if (v_len == 0) {
				printf("V not specified !\n");
				exit(1);
				}
			tdo = new int[v_len * b_len];
			for (i = 0; i < v_len; i++) {
				for (j = 0; j < b_len; j++) {
					f >> tdo[i * b_len + j];
					}
				}
			printf("TDO: \n");
			for (i = 0; i < v_len; i++) {
				for (j = 0; j < b_len; j++) {
					printf("%d ", tdo[i * b_len + j]);
					}
				printf("\n");
				}
			printf("\n");
			}
		
#if 0
		if (strncmp(buf, "GEO_file", 8) == 0) {
			fgets(buf, BUFSIZE, fp);
			l = strlen(buf);
			if (l)
				buf[l - 1] = 0; /* delete newline */
			if (!f_no_inc_files) {
				strcpy(inc_file_name, buf);
				printf("inc_file_name = %s\n", inc_file_name);
				}
			}
#endif
		if (strcmp(buf, "end") == 0) {
			break;
			}

		if (strcmp(buf, "f_v") == 0) {
			f_v = TRUE;
			printf("f_v\n");
			}

		if (strcmp(buf, "f_vv") == 0) {
			f_vv = TRUE;
			printf("f_vv\n");
			}

		if (strcmp(buf, "f_do_iso_test") == 0) {
			f_do_iso_test = TRUE;
			printf("f_do_iso_test\n");
			}

		if (strcmp(buf, "f_do_aut_group") == 0) {
			f_do_aut_group = TRUE;
			printf("f_do_aut_group\n");
			}

		if (strcmp(buf, "f_transpose") == 0) {
			f_transpose = TRUE;
			printf("f_transpose\n");
			}

#if 0
		if (strcmp(buf, "print_interval") == 0) {
			fscanf(fp, "%d", &gen_print_interval);
			printf("gen_print_interval = %d\n", gen_print_interval);
			}

		if (strcmp(buf, "f_resolve") == 0) {
			f_resolve = TRUE;
			printf("f_resolve\n");
			}

		if (strcmp(buf, "f_resolve_with_aut_group") == 0) {
			f_resolve_with_aut_group = TRUE;
			f_resolve = TRUE;
			printf("f_resolve_with_aut_group\n");
			}
		
		if (strcmp(buf, "f_lambda") == 0) {
			f_lambda = TRUE;
			fscanf(fp, "%d", &lambda);
			printf("lambda = %d\n", lambda);
			}
		if (strcmp(buf, "f_ignore_square") == 0) { /* JS 120100 */
			f_find_square = FALSE;
			puts("f_ignore_square");
			}
		if (strcmp(buf, "f_simple") == 0) { /* JS 180100 */
			f_simple = TRUE;
			puts("f_simple");
			}
#endif
		}
	
	if (no > 200) 
		return;
		
	g << "<HTDO id=" << no << " type=pt nb_V=" << v_len << " nb_B=" << b_len << " fuse=";
	if (f_simple) {
		g << "simple";
		}
	else {
		g << "double";
		}
	g << ">" << endl;
	g << endl;
	for (i = 0; i < 3; i++)
		g << " ";
	for (j = 0; j < b_len; j++) 
		g << setw(3) << B[j];
	g << endl;
	for (i = 0; i < v_len; i++) {
		g << setw(3) << V[i];
		for (j = 0; j < b_len; j++) 
			g << setw(3) << tdo[i * b_len + j];
		g << endl;
		}
	g << endl;
	if (f_simple) {
		for (i = 0; i < nb_fuse; i++) {
			for (j = 0; j < fuse[i]; j++) {
				g << setw(2) << i + 1;
				}
			}
		g << endl;
		}
	else {
		g << b_len - 2 << " 2 0 " << b_len - 2 << endl;
		for (i = 0; i < v_len; i++) {
			g << setw(2) << 1;
			}
		g << endl;
		j = 1;
		l = 0;
		for (i = 0; i < v_len; i++) {
			g << setw(2) << j;
			l += V[i];
			if (l == 7 || l == 7 + 5) {
				j++;
				}
			}
		g << endl;
		}
#if 0
	for (i = 0; i < nb_fuse; i++) {
		for (j = 0; j < fuse[i]; j++) {
			g << setw(2) << i;
			}
		}
	g << endl;
#endif
	g << "</HTDO>" << endl;
	g << endl;
}
