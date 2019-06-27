// field_plot.cpp
// 
// Anton Betten
// Nov 27, 2011
//
//


#include "orbiter.h"

using namespace std;


using namespace orbiter;

// global data:

int t0; // the system time when the program started

void draw_empty_grid(int q, int type, int param_a, int xmax, int ymax, int verbose_level);
void draw_beginning(char *fname, mp_graphics *&G, int xmax, int ymax, int verbose_level);
void draw_end(char *fname, mp_graphics *G, int xmax, int ymax, int verbose_level);
void draw_grid_(mp_graphics &G, int xmax, int ymax, int q, int type, int param_a);
void prepare_latex_simple(char *fname_base, int verbose_level);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q = 0;
	int xmax = 300;
	int ymax = 300;
	int f_type = FALSE;
	int type = 0;
	int param_a = 0;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-type") == 0) {
			f_type = TRUE;
			type = atoi(argv[++i]);
			param_a = atoi(argv[++i]);
			cout << "-type " << type << endl;
			}
		else if (strcmp(argv[i], "-x") == 0) {
			xmax = atoi(argv[++i]);
			cout << "-x " << xmax << endl;
			}
		else if (strcmp(argv[i], "-y") == 0) {
			ymax = atoi(argv[++i]);
			cout << "-y " << ymax << endl;
			}
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_type) {
		cout << "please use option -type <type>" << endl;
		exit(1);
		}
	draw_empty_grid(q, type, param_a, xmax, ymax, verbose_level);

	the_end(t0);
}

void draw_empty_grid(int q, int type, int param_a, int xmax, int ymax, int verbose_level)
{
	{
	char fname[1000];
	{
	mp_graphics *G;

	sprintf(fname, "grid_%d_%d", q, type);
	draw_beginning(fname, G, xmax, ymax, verbose_level);

	draw_grid_(*G, xmax, ymax, q, type, param_a);
	draw_end(fname, G, xmax, ymax, verbose_level);
	}
	prepare_latex_simple(fname, verbose_level);
	}
		
}



void draw_beginning(char *fname, mp_graphics *&G, int xmax, int ymax, int verbose_level)
{
	int x_min = 0, x_max = ONE_MILLION;
	int y_min = 0, y_max = ONE_MILLION;
	int factor_1000 = 1000;
	char fname_full[1000];
	int f_embedded = TRUE;
	int f_sideways = FALSE;
	
	//cout << "draw_grid q=" << q << endl;
	sprintf(fname_full, "%s.mp", fname);
	//{
	G = NEW_OBJECT(mp_graphics);
	G->init(fname_full, x_min, y_min, x_max, y_max, f_embedded, f_sideways);
	G->out_xmin() = 0;
	G->out_ymin() = 0;
	G->out_xmax() = xmax;
	G->out_ymax() = ymax;
	cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
	G->header();
	G->begin_figure(factor_1000);
	
	//draw_grid_(G, q, verbose_level);
}

void draw_end(char *fname, mp_graphics *G, int xmax, int ymax, int verbose_level)
{
	//int x_min = 0, x_max = 1000;
	//int y_min = 0, y_max = 1000;
	//int factor_1000 = 1000;
	char fname_full[1000];
	file_io Fio;

	sprintf(fname_full, "%s.mp", fname);
	G->end_figure();
	G->footer();
	FREE_OBJECT(G);
	
	cout << "written file " << fname_full << " of size " << Fio.file_size(fname_full) << endl;
	
}



void draw_grid_(mp_graphics &G, int xmax, int ymax, 
	int q, int type, int param_a)
{
	grid_frame F;
	int i, j;
	int Px[10], Py[10];
	char text[1000];
	int rad = 10000;

	F.f_matrix_notation = FALSE;
	F.m = q - 1;
	F.n = q - 1;
	F.origin_x = 0.;
	F.origin_y = 0.;
	F.dx = ONE_MILLION / F.m; // goes down in matrix notation
	F.dy = ONE_MILLION / F.n; // goes accross

	cout << "math_day::draw_tree2" << endl;
	cout << "dx=" << F.dx << endl;
	cout << "dy=" << F.dy << endl;

	//int line_dashing = 5;
	//G.sl_udsty(line_dashing);

	// draw horizontal lines 
	for (i = 0; i <= q - 1; i++) {
		Px[0] = (int)(F.origin_x + F.m * F.dx);
		Py[0] = (int)(F.origin_y + i * F.dy);
		Px[1] = (int)(F.origin_x + 0 * F.dx);
		Py[1] = Py[0];
		Px[2] = (int)(F.origin_x + (-1) * F.dx);
		Py[2] = (int)(F.origin_y + i * F.dy);
		G.polygon2(Px, Py, 0, 1);
		sprintf(text, "%d", i);
		G.aligned_text(Px[2], Py[2], "", text);
		}

	// draw vertical lines 
	for (i = 0; i <= q - 1; i++) {
		Px[0] = (int)(F.origin_x + i * F.dx);
		Py[0] = (int)(F.origin_y);
		Px[1] = Px[0];
		Py[1] = (int)(F.origin_y + (F.n) * F.dy);
		Px[2] = (int)(F.origin_x + i * F.dx);
		Py[2] = (int)(F.origin_y + (-1) * F.dy);
		G.polygon2(Px, Py, 0, 1);
		sprintf(text, "%d", i);
		G.aligned_text(Px[2], Py[2], "", text);
		}

		//sprintf(text, "{\\bf \\Large %s}", time[i]);
		//G.aligned_text(Px, Py, 0, "", text);

	finite_field *Fq;

	Fq = NEW_OBJECT(finite_field);

	Fq->init(q, 0);

	for (i = 0; i < q; i++) {

		if (type == -1) {
			continue;
			}
		else if (type == 0) {
			if (param_a == -1) {
				j = Fq->alpha_power(i);
				}
			else {
				j = Fq->power(param_a, i);
				}
			}
		else if (type == 1) {
			if (param_a == -1) {
				j = Fq->mult(3, i);
				}
			else {
				j = Fq->mult(param_a, i);
				}
			}
		else if (type == 2) {
			if (i == 0) 
				continue;
			
			j = Fq->inverse(i);
			}
		

		Px[0] = (int)(F.origin_x + i * F.dx);
		Py[0] = (int)(F.origin_y + j * F.dy);

		G.nice_circle(Px[0], Py[0], rad);
		}

	FREE_OBJECT(Fq);



}

void prepare_latex_simple(char *fname_base, int verbose_level)
{
	char tex_file_name[1000];
	char dvi_file_name[1000];
	char ps_file_name[1000];
	
	sprintf(tex_file_name, "%s.tex", fname_base);
	sprintf(dvi_file_name, "%s.dvi", fname_base);
	sprintf(ps_file_name, "%s.ps", fname_base);
	
	{
	ofstream f(tex_file_name);
	
	f << "\\documentclass[]{article}" << endl;
	f << "\\usepackage{amsmath}" << endl;
	f << "\\usepackage{amssymb}" << endl;
	f << "\\usepackage{latexsym}" << endl;
	f << "\\usepackage{epsfig}" << endl;
	f << "%%\\usepackage{supertabular}" << endl;
	f << "\\evensidemargin 0in" << endl;
	f << "\\oddsidemargin 0in" << endl;
	f << "\\marginparwidth 0pt" << endl;
	f << "\\marginparsep 0pt" << endl;
	f << "\\topmargin -1in" << endl;
	f << "\\headheight 0.7cm" << endl;
	f << "\\headsep 1.8cm" << endl;
	f << "%%\\footheight 0.7cm" << endl;
	f << "\\footskip 2cm" << endl;
	f << "\\textheight 22cm" << endl;
	f << "\\textwidth 6.2in" << endl;
	f << "\\marginparpush 0pt" << endl;
	f << "%%\\newcommand{\\dominowidth}{167mm}" << endl;
	f << "\\newcommand{\\dominowidth}{190mm}" << endl;
	f << "\\newcommand{\\dominowidthsmall}{90mm}" << endl;
	//f << "\\title{" << photo_label_tex << "}" << endl;
	//f << "%%\\author{{\\sc }}" << endl;
	//f << "\\date{\\today}" << endl;
	f << "\\pagestyle{empty}" << endl;
	f << "\\begin{document}" << endl;
	//f << "\\maketitle" << endl;
	f << "\\begin{center}" << endl;
	f << "\\hspace*{-15mm}\\epsfig{file=" << fname_base << ".1,width=\\dominowidth}" << endl; 
	
	f << "\\end{center}" << endl;
	

	f << "\\end{document}" << endl;
	}
	char cmd0[1000];
	char cmd1[1000];
	char cmd2[1000];
	char cmd3[1000];
	char cmd4[1000];
	
	sprintf(cmd0, "mpost %s.mp", fname_base);
	sprintf(cmd1, "latex %s", tex_file_name);
	sprintf(cmd2, "dvips %s -o", dvi_file_name);
	sprintf(cmd4, "convert -trim -density 300 %s.ps %s.png", fname_base, fname_base);
	sprintf(cmd3, "open %s &", ps_file_name);
	system(cmd0);
	system(cmd1);
	system(cmd2);
	system(cmd4);
	system(cmd3);
	//system("latex view.tex");
	//system("dvips view.dvi -o");
	//system("open view.ps &");


}







