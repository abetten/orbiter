// tdo_print.C
// Anton Betten
//
// started:  Dec 31 2006

#include "orbiter.h"
#include <string>
#include <DISCRETA/discreta.h>

INT t0;

using std::string;


void intersection_of_columns(geo_parameter &GP, tdo_scheme &G, 
	INT j1, INT j2, Vector &V, Vector &M, INT verbose_level);

void print_usage()
{
	cout << "usage: tdo_print.out [options] <tdo_file>" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <n> " << endl;
	cout << "   verbose level <n>" << endl;
	cout << "-widor" << endl;
	cout << "   read input from a widor file" << endl;
	cout << "-range <f> <l> " << endl;
	cout << "   select the TDO in interval [f..f+l-1]" << endl;
	cout << "   where counting starts from 0" << endl;
	cout << "-select <label> " << endl;
	cout << "   select the TDO whose label is <label>" << endl;
	cout << "-C " << endl;
	cout << "   produce C-source output" << endl;
	cout << "-tex " << endl;
	cout << "   print the row-scheme in tex" << endl;
	cout << "-texfile <texfile_name> " << endl;
	cout << "   print the row-scheme in tex to the file <texfile_name>" << endl;
	cout << "-Tex " << endl;
	cout << "   print the row-scheme in tex, create separate files for each scheme" << endl;
	cout << "-nt " << endl;
	cout << "   select the non-tactical TDO" << endl;
	cout << "-w " << endl;
	cout << "   write TDO file with all TDO that were selected" << endl;
	cout << "-intersection <j1> <j2>" << endl;
	cout << "   computes intersections of lines of type j1 and j2;" << endl;
	cout << "   j1 and j2 are negative and count from the end;" << endl;
}

int main(int argc, char **argv)
{
	INT cnt;
	BYTE str[1000];
	BYTE ext[1000];
	BYTE *fname_in;
	BYTE fname_out[1000];
	INT verbose_level = 0;
	INT f_widor = FALSE;
	INT f_range = FALSE;
	INT range_first, range_len;
	INT f_select = FALSE;
	BYTE *select_label;
	INT f_C = FALSE;
	INT f_tex = FALSE;
	INT f_texfile = FALSE;
	BYTE *texfile_name = NULL;
	INT f_ROW = FALSE;
	INT f_COL = FALSE;
	INT f_Tex = FALSE;
	INT f_nt = FALSE;
	INT f_intersection = FALSE;
	INT intersection_j1, intersection_j2;
	INT i, f_doit;
	INT f_w = FALSE;
	INT nb_written = 0;
	INT f_v, f_vv, f_vvv;

	t0 = os_ticks();
	if (argc <= 1) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-widor") == 0) {
			f_widor = TRUE;
			cout << "-widor " << endl;
			}
		else if (strcmp(argv[i], "-range") == 0) {
			f_range = TRUE;
			range_first = atoi(argv[++i]);
			range_len = atoi(argv[++i]);
			cout << "-range " << range_first << " " << range_len << endl;
			}
		else if (strcmp(argv[i], "-select") == 0) {
			f_select = TRUE;
			select_label = argv[++i];
			cout << "-select " << select_label << endl;
			}
		else if (strcmp(argv[i], "-C") == 0) {
			f_C = TRUE;
			cout << "-C" << endl;
			}
		else if (strcmp(argv[i], "-ROW") == 0) {
			f_ROW = TRUE;
			cout << "-ROW" << endl;
			}
		else if (strcmp(argv[i], "-COL") == 0) {
			f_COL = TRUE;
			cout << "-COL" << endl;
			}
		else if (strcmp(argv[i], "-tex") == 0) {
			f_tex = TRUE;
			cout << "-tex" << endl;
			}
		else if (strcmp(argv[i], "-texfile") == 0) {
			f_texfile = TRUE;
			texfile_name = argv[++i];
			cout << "-texfile " << texfile_name << endl;
			}
		else if (strcmp(argv[i], "-Tex") == 0) {
			f_Tex = TRUE;
			cout << "-Tex" << endl;
			}
		else if (strcmp(argv[i], "-nt") == 0) {
			f_nt = TRUE;
			cout << "-nt" << endl;
			}
		else if (strcmp(argv[i], "-w") == 0) {
			f_w = TRUE;
			cout << "-w" << endl;
			}
		else if (strcmp(argv[i], "-intersection") == 0) {
			f_intersection = TRUE;
			intersection_j1 = atoi(argv[++i]);
			intersection_j2 = atoi(argv[++i]);
			cout << "-intersection " << intersection_j1 << " " << intersection_j2 << endl;
			}
		}
	fname_in = argv[argc - 1];
	cout << "opening file " << fname_in << " for reading" << endl;
	ifstream f(fname_in);
	ofstream *g = NULL;
	
	ofstream *texfile;


	
	strcpy(str, fname_in);
	get_extension_if_present(str, ext);
	chop_off_extension_if_present(str, ext);

	sprintf(fname_out, "%sw.tdo", str);
	if (f_w) {
		g = new ofstream(fname_out);
		}
	if (f_texfile) {
		texfile = new ofstream(texfile_name);
		}
	f_v = (verbose_level >= 1);
	f_vv = (verbose_level >= 2);
	f_vvv = (verbose_level >= 3);

	
	geo_parameter GP;
	tdo_scheme G;


	Vector vm, VM, VM_mult;
	discreta_base mu;
	
	if (f_intersection) {
		VM.m_l(0);
		VM_mult.m_l(0);
		}
	
	for (cnt = 0; ; cnt++) {
		if (f.eof()) {
			cout << "eof reached" << endl;
			break;
			}
		if (f_widor) {
			if (!GP.input(f)) {
				//cout << "GP.input returns FALSE" << endl;
				break;
				}
			}
		else {
			if (!GP.input_mode_stack(f, verbose_level - 1)) {
				//cout << "GP.input_mode_stack returns FALSE" << endl;
				break;
				}
			}
		//if (f_v) {
			//cout << "read decomposition " << cnt << endl;
			//}
		
		f_doit = TRUE;
		if (f_range) {
			if (cnt < range_first || cnt >= range_first + range_len)
				f_doit = FALSE;
			}
		if (f_select) {
			if (strcmp(GP.label, select_label))
				continue;
			}
		if (f_nt) {
			if (GP.row_level == GP.col_level)
				continue;
			}
		if (!f_doit) {
			continue;
			}
		//cout << "before convert_single_to_stack" << endl;	
		//GP.convert_single_to_stack();
		//cout << "after convert_single_to_stack" << endl;
		//sprintf(label, "%s.%ld", str, i);
		//GP.write(g, label);
		if (f_vv) {
			cout << "before init_tdo_scheme" << endl;
			}
		GP.init_tdo_scheme(G, verbose_level - 1);
		if (f_vv) {
			cout << "after init_tdo_scheme" << endl;
			}
		GP.print_schemes(G);

		if (f_C) {
			GP.print_C_source();
			}
		if (f_tex) {
			GP.print_scheme_tex(cout, G, ROW);
			GP.print_scheme_tex(cout, G, COL);
			}
		if (f_texfile) {
			if (f_ROW) {
				GP.print_scheme_tex(*texfile, G, ROW);
				}
			if (f_COL) {
				GP.print_scheme_tex(*texfile, G, COL);
				}
			}
		if (f_Tex) {
			BYTE fname[1000];
			
			sprintf(fname, "%s.tex", GP.label);
			ofstream f(fname);
			
			GP.print_scheme_tex(f, G, ROW);
			GP.print_scheme_tex(f, G, COL);
			}
		if (f_intersection) {
			Vector V, M;
			intersection_of_columns(GP, G, 
				intersection_j1, intersection_j2, V, M, verbose_level - 1);
			vm.m_l(2);
			vm.s_i(0).swap(V);
			vm.s_i(1).swap(M);
			cout << "vm:" << vm << endl;
			INT idx;
			mu.m_i_i(1);
			if (VM.search(vm, &idx)) {
				VM_mult.m_ii(idx, VM_mult.s_ii(idx) + 1);
				}
			else {
				cout << "inserting at position " << idx << endl;
				VM.insert_element(idx, vm);
				VM_mult.insert_element(idx, mu);
				}
			}
		if (f_w) {
			GP.write_mode_stack(*g, GP.label);
			nb_written++;
			}
		}

	if (f_w) {
		*g << "-1 " << nb_written << endl;
		delete g;
		
		}

	if (f_texfile) {
		delete texfile;
		}

	if (f_intersection) {
		INT cl, c, l, j, L;
		cout << "the intersection types are:" << endl;
		for (i = 0; i < VM.s_l(); i++) {
			//cout << setw(5) << VM_mult.s_ii(i) << " x " << VM.s_i(i) << endl;
			cout << "intersection type " << i + 1 << ":" << endl;
			Vector &V = VM.s_i(i).as_vector().s_i(0).as_vector();
			Vector &M = VM.s_i(i).as_vector().s_i(1).as_vector();
			//cout << "V=" << V << endl;
			//cout << "M=" << M << endl;
			cl = V.s_l();
			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					INT mult = Mc.s_ii(j);
					cout << setw(5) << mult << " x " << the_type << endl;
					}
				cout << "--------------------------" << endl;
				}
			cout << "appears " << setw(5) << VM_mult.s_ii(i) << " times" << endl;

			classify *C;
			classify *C_pencil;
			INT f_second = FALSE;
			INT *pencil_data;
			INT pencil_data_size = 0;
			INT pos, b, hh;
			
			C = new classify[cl];
			C_pencil = new classify;

			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				L = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					INT mult = Mc.s_ii(j);
					if (the_type.s_ii(1) == 1 && the_type.s_ii(0)) {
						pencil_data_size += mult;
						}
					}
				}
			//cout << "pencil_data_size=" << pencil_data_size << endl;
			pencil_data = new INT[pencil_data_size];
			pos = 0;
			
			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				L = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					INT mult = Mc.s_ii(j);
					if (the_type.s_ii(1) == 1 && the_type.s_ii(0)) {
						b = the_type.s_ii(0);
						for (hh = 0; hh < mult; hh++) {
							pencil_data[pos++] = b;
							}
						}
					}
				}
			//cout << "pencil_data: ";
			//INT_vec_print(cout, pencil_data, pencil_data_size);
			//cout << endl;
			C_pencil->init(pencil_data, pencil_data_size, FALSE /*f_second */, verbose_level - 2);
			delete [] pencil_data;
			
			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				L = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					if (the_type.s_ii(1)) 
						continue;
					INT mult = Mc.s_ii(j);
					L += mult;
					}
				INT *data;
				INT k, h, a;

				data = new INT[L];
				k = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					INT mult = Mc.s_ii(j);
					if (the_type.s_ii(1)) 
						continue;
					a = the_type.s_ii(0);
					for (h = 0; h < mult; h++) {
						data[k++] = a;
						}
					}
				//cout << "data: ";
				//INT_vec_print(cout, data, L);
				//cout << endl;
				C[c].init(data, L, f_second, verbose_level - 2);
				delete [] data;
				}
			
			cout << "Intersection type " << i + 1 << ": pencil type: (";
			C_pencil->print_naked(FALSE /*f_backwards*/);
			cout << ") ";
			cout << "intersection type: (";
			for (c = 0; c < cl; c++) {
				C[c].print_naked(FALSE /*f_backwards*/);
				if (c < cl - 1)
					cout << " | ";
				}
			cout << ") appears " << VM_mult.s_ii(i) << " times" << endl;
			//C_pencil->print();
			delete [] C;
			delete C_pencil;
			}
		}
	cout << "time: ";
	time_check(cout, t0);
	cout << endl;
}

void intersection_of_columns(geo_parameter &GP, tdo_scheme &G, 
	INT j1, INT j2, Vector &V, Vector &M, INT verbose_level)
{
	INT f_v = (verbose_level >= 3);
	INT f_vv = (verbose_level >= 4);
	INT c, i, c1, J1, J2, a1, a2, m, f, l, ii;
	INT h = ROW;
	partitionstack P;
	
	if (f_v) {
		cout << "intersection_of_columns:" << endl;
		}
	if (j1 < 0) {
		J1 = G.nb_col_classes[h] + j1;
		}
	else {
		J1 = j1;
		}
	if (j2 < 0) {
		J2 = G.nb_col_classes[h] + j2;
		}
	else {
		J2 = j2;
		}
	
	G.get_row_split_partition(0 /*verbose_level*/, P);
	if (f_vv) {
		cout << "row split partition: " << P << endl;
		P.print_raw();
		}

	Vector v;
	discreta_base mu;
	INT idx;
	
	V.m_l(P.ht);
	M.m_l(P.ht);
	for (c = 0; c < P.ht; c++) {
		V.s_i(c).change_to_vector().m_l(0);
		M.s_i(c).change_to_vector().m_l(0);
		f = P.startCell[c];
		l = P.cellSize[c];
		for (ii = 0; ii < l; ii++) {
			i = P.pointList[f + ii];
		//for (i = 0; i < G.nb_row_classes[h]; i++) {
			c1 = G.row_classes[h][i];
			m = G.row_classes_len[h][i];
			v.m_l_n(2);
			a1 = G.the_row_scheme[i * G.nb_col_classes[h] + J1];
			a2 = G.the_row_scheme[i * G.nb_col_classes[h] + J2];
			cout << setw(5) << m << " x " << setw(5) << a1 << " " << setw(5) << a2 << endl;
			mu.m_i_i(m);
			v.m_ii(0, a1);
			v.m_ii(1, a2);
			if (V.s_i(c).as_vector().search(v, &idx)) {
				M.s_i(c).as_vector().m_ii(idx, M.s_i(c).as_vector().s_ii(idx) + m);
				}
			else {
				V.s_i(c).as_vector().insert_element(idx, v);
				M.s_i(c).as_vector().insert_element(idx, mu);
				}
			}
		}
	cout << "V,M=" << V << "," << M << endl;
}
