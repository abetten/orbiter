/*
 * latex_interface.cpp
 *
 *  Created on: Apr 13, 2019
 *      Author: betten
 */


#include "foundations.h"

#include <sstream>

using namespace std;


namespace orbiter {
namespace foundations {


latex_interface::latex_interface()
{

}

latex_interface::~latex_interface()
{

}

void latex_interface::head_easy(ostream& ost)
{
	head(ost,
		FALSE /* f_book */,
		FALSE /* f_title */,
		"", "",
		FALSE /*f_toc */,
		FALSE /* f_landscape */,
		FALSE /* f_12pt */,
		FALSE /* f_enlarged_page */,
		FALSE /* f_pagenumbers */,
		NULL /* extras_for_preamble */);

}

void latex_interface::head_easy_with_extras_in_the_praeamble(
		ostream& ost, const char *extras)
{
	head(ost,
		FALSE /* f_book */,
		FALSE /* f_title */,
		"", "",
		FALSE /*f_toc */,
		FALSE /* f_landscape */,
		FALSE /* f_12pt */,
		FALSE /* f_enlarged_page */,
		FALSE /* f_pagenumbers */,
		extras /* extras_for_preamble */);

}

void latex_interface::head_easy_sideways(ostream& ost)
{
	head(ost, FALSE /* f_book */,
		FALSE /* f_title */,
		"", "",
		FALSE /*f_toc */,
		TRUE /* f_landscape */,
		FALSE /* f_12pt */,
		FALSE /* f_enlarged_page */,
		FALSE /* f_pagenumbers */,
		NULL /* extras_for_preamble */);

}

void latex_interface::head(ostream& ost,
	int f_book, int f_title,
	const char *title, const char *author,
	int f_toc, int f_landscape, int f_12pt,
	int f_enlarged_page, int f_pagenumbers,
	const char *extras_for_preamble)
{
	if (f_12pt) {
		ost << "\\documentclass[12pt]{";
		}
	else {
		ost << "\\documentclass{";
		}
	if (f_book)
		ost << "book";
	else
		ost << "article";
	ost << "}\n";
	ost << "% a4paper\n";
	ost << endl;
	ost << "%\\usepackage[dvips]{epsfig}\n";
	ost << "%\\usepackage{cours11, cours}\n";
	ost << "%\\usepackage{fancyheadings}\n";
	ost << "%\\usepackage{calc}\n";
	ost << "\\usepackage{amsmath}\n";
	ost << "\\usepackage{amssymb}\n";
	ost << "\\usepackage{latexsym}\n";
	ost << "\\usepackage{epsfig}\n";
	ost << "\\usepackage{enumerate}\n";
	ost << "%\\usepackage{supertabular}\n";
	ost << "%\\usepackage{wrapfig}\n";
	ost << "%\\usepackage{blackbrd}\n";
	ost << "%\\usepackage{epic,eepic}\n";
	ost << "\\usepackage{rotating}\n";
	ost << "\\usepackage{multicol}\n";
	ost << "%\\usepackage{multirow}\n";
	ost << "\\usepackage{makeidx} % additional command see\n";
	ost << "\\usepackage{rotating}\n";
	ost << "\\usepackage{array}\n";
	ost << "\\usepackage{tikz}\n";
	ost << "\\usepackage{longtable}\n";
	ost << "\\usepackage{anyfontsize}\n";
	ost << "\\usepackage{t1enc}\n";
	ost << "%\\usepackage{amsmath,amsfonts} \n";
	ost << endl;
	ost << endl;
	ost << "%\\usepackage[mtbold,mtplusscr]{mathtime}\n";
	ost << "% lucidacal,lucidascr,\n";
	ost << endl;
	ost << "%\\usepackage{mathtimy}\n";
	ost << "%\\usepackage{bm}\n";
	ost << "%\\usepackage{avant}\n";
	ost << "%\\usepackage{basker}\n";
	ost << "%\\usepackage{bembo}\n";
	ost << "%\\usepackage{bookman}\n";
	ost << "%\\usepackage{chancery}\n";
	ost << "%\\usepackage{garamond}\n";
	ost << "%\\usepackage{helvet}\n";
	ost << "%\\usepackage{newcent}\n";
	ost << "%\\usepackage{palatino}\n";
	ost << "%\\usepackage{times}\n";
	ost << "%\\usepackage{pifont}\n";
	if (f_enlarged_page) {
		ost << "\\usepackage{fullpage}" << endl;
		ost << "\\usepackage[top=1in,bottom=1in,right=1in,left=1in]{geometry}" << endl;
#if 0
		ost << "%\\voffset=-1.5cm" << endl;
		ost << "\\hoffset=-2cm" << endl;
		ost << "\\textwidth=20cm" << endl;
		ost << "%\\topmargin 0.0in" << endl;
		ost << "\\textheight 25cm" << endl;
#endif
		}

	if (extras_for_preamble) {
		ost << extras_for_preamble << endl;
		}
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\parindent=0pt\n";
	ost << endl;
	//ost << "\\renewcommand{\\baselinestretch}{1.5}\n";
	ost << endl;


#if 0
	if (f_enlarged_page) {
		ost << "\\hoffset -2cm\n";
		ost << "\\voffset -1cm\n";
		ost << "\\topmargin 0.0cm\n";
		if (f_landscape) {
			ost << "\\textheight=18cm\n";
			ost << "\\textwidth=23cm\n";
			}
		else {
			ost << "\\textheight=23cm\n";
			ost << "\\textwidth=18cm\n";
			}
		}
	else {
		ost << "\\hoffset -0.7cm\n";
		ost << "%\\voffset 0cm\n";
		ost << endl;
		ost << "%\\oddsidemargin=15pt\n";
		ost << endl;
		ost << "%\\oddsidemargin 0pt\n";
		ost << "%\\evensidemargin 0pt\n";
		ost << "%\\topmargin 0pt\n";
		ost << endl;
#if 1
		if (f_landscape) {
			ost << "\\textwidth = 20cm\n";
			ost << "\\textheight= 17cm\n";
			}
		else {
			ost << "\\textwidth = 17cm\n";
			ost << "\\textheight= 21cm\n";
			}
		ost << endl;
#endif
		}
#endif


	ost << "%\\topmargin=0pt\n";
	ost << "%\\headsep=18pt\n";
	ost << "%\\footskip=45pt\n";
	ost << "%\\mathsurround=1pt\n";
	ost << "%\\evensidemargin=0pt\n";
	ost << "%\\oddsidemargin=15pt\n";
	ost << endl;

	ost << "%\\setlength{\\textheight}{\\baselineskip*41+\\topskip}\n";
	ost << endl;


	ost << "\\newcommand{\\sectionline}{" << endl;
	ost << "   \\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   \\hspace{\\fill}\\rule{0.9\\linewidth}{1.7pt}\\hspace{\\fill}" << endl;
	ost << "   \\par\\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   }" << endl;

	ost << "\\newcommand\\setTBstruts{\\def\\T{\\rule{0pt}{2.6ex}}%" << endl;
	ost << "\\def\\B{\\rule[-1.2ex]{0pt}{0pt}}}" << endl;

	ost << "\\newcommand{\\ans}[1]{\\\\{\\bf ANSWER}: {#1}}" << endl;
	ost << "\\newcommand{\\Aut}{{\\rm Aut}}\n";
	ost << "\\newcommand{\\Sym}{{\\rm Sym}}\n";
	ost << "\\newcommand{\\sFix}{{\\cal Fix}}\n";
	ost << "\\newcommand{\\sOrbits}{{\\cal Orbits}}\n";
	//ost << "\\newcommand{\\sFix}{{\\mathscr Fix}}\n";
	//ost << "\\newcommand{\\sOrbits}{{\\mathscr Orbits}}\n";
	ost << "\\newcommand{\\Stab}{{\\rm Stab}}\n";
	ost << "\\newcommand{\\Fix}{{\\rm Fix}}\n";
	ost << "\\newcommand{\\fix}{{\\rm fix}}\n";
	ost << "\\newcommand{\\Orbits}{{\\rm Orbits}}\n";
	ost << "\\newcommand{\\PG}{{\\rm PG}}\n";
	ost << "\\newcommand{\\AG}{{\\rm AG}}\n";
	ost << "\\newcommand{\\SQS}{{\\rm SQS}}\n";
	ost << "\\newcommand{\\STS}{{\\rm STS}}\n";
	//ost << "\\newcommand{\\Sp}{{\\rm Sp}}\n";
	ost << "\\newcommand{\\PSL}{{\\rm PSL}}\n";
	ost << "\\newcommand{\\PGL}{{\\rm PGL}}\n";
	ost << "\\newcommand{\\PSSL}{{\\rm P\\Sigma L}}\n";
	ost << "\\newcommand{\\PGGL}{{\\rm P\\Gamma L}}\n";
	ost << "\\newcommand{\\SL}{{\\rm SL}}\n";
	ost << "\\newcommand{\\GL}{{\\rm GL}}\n";
	ost << "\\newcommand{\\SSL}{{\\rm \\Sigma L}}\n";
	ost << "\\newcommand{\\GGL}{{\\rm \\Gamma L}}\n";
	ost << "\\newcommand{\\ASL}{{\\rm ASL}}\n";
	ost << "\\newcommand{\\AGL}{{\\rm AGL}}\n";
	ost << "\\newcommand{\\ASSL}{{\\rm A\\Sigma L}}\n";
	ost << "\\newcommand{\\AGGL}{{\\rm A\\Gamma L}}\n";
	ost << "\\newcommand{\\PSU}{{\\rm PSU}}\n";
	ost << "\\newcommand{\\HS}{{\\rm HS}}\n";
	ost << "\\newcommand{\\Hol}{{\\rm Hol}}\n";
	ost << "\\newcommand{\\SO}{{\\rm SO}}\n";
	ost << "\\newcommand{\\ASO}{{\\rm ASO}}\n";

	ost << "\\newcommand{\\la}{\\langle}\n";
	ost << "\\newcommand{\\ra}{\\rangle}\n";


	ost << "\\newcommand{\\cA}{{\\cal A}}\n";
	ost << "\\newcommand{\\cB}{{\\cal B}}\n";
	ost << "\\newcommand{\\cC}{{\\cal C}}\n";
	ost << "\\newcommand{\\cD}{{\\cal D}}\n";
	ost << "\\newcommand{\\cE}{{\\cal E}}\n";
	ost << "\\newcommand{\\cF}{{\\cal F}}\n";
	ost << "\\newcommand{\\cG}{{\\cal G}}\n";
	ost << "\\newcommand{\\cH}{{\\cal H}}\n";
	ost << "\\newcommand{\\cI}{{\\cal I}}\n";
	ost << "\\newcommand{\\cJ}{{\\cal J}}\n";
	ost << "\\newcommand{\\cK}{{\\cal K}}\n";
	ost << "\\newcommand{\\cL}{{\\cal L}}\n";
	ost << "\\newcommand{\\cM}{{\\cal M}}\n";
	ost << "\\newcommand{\\cN}{{\\cal N}}\n";
	ost << "\\newcommand{\\cO}{{\\cal O}}\n";
	ost << "\\newcommand{\\cP}{{\\cal P}}\n";
	ost << "\\newcommand{\\cQ}{{\\cal Q}}\n";
	ost << "\\newcommand{\\cR}{{\\cal R}}\n";
	ost << "\\newcommand{\\cS}{{\\cal S}}\n";
	ost << "\\newcommand{\\cT}{{\\cal T}}\n";
	ost << "\\newcommand{\\cU}{{\\cal U}}\n";
	ost << "\\newcommand{\\cV}{{\\cal V}}\n";
	ost << "\\newcommand{\\cW}{{\\cal W}}\n";
	ost << "\\newcommand{\\cX}{{\\cal X}}\n";
	ost << "\\newcommand{\\cY}{{\\cal Y}}\n";
	ost << "\\newcommand{\\cZ}{{\\cal Z}}\n";

	ost << "\\newcommand{\\rmA}{{\\rm A}}\n";
	ost << "\\newcommand{\\rmB}{{\\rm B}}\n";
	ost << "\\newcommand{\\rmC}{{\\rm C}}\n";
	ost << "\\newcommand{\\rmD}{{\\rm D}}\n";
	ost << "\\newcommand{\\rmE}{{\\rm E}}\n";
	ost << "\\newcommand{\\rmF}{{\\rm F}}\n";
	ost << "\\newcommand{\\rmG}{{\\rm G}}\n";
	ost << "\\newcommand{\\rmH}{{\\rm H}}\n";
	ost << "\\newcommand{\\rmI}{{\\rm I}}\n";
	ost << "\\newcommand{\\rmJ}{{\\rm J}}\n";
	ost << "\\newcommand{\\rmK}{{\\rm K}}\n";
	ost << "\\newcommand{\\rmL}{{\\rm L}}\n";
	ost << "\\newcommand{\\rmM}{{\\rm M}}\n";
	ost << "\\newcommand{\\rmN}{{\\rm N}}\n";
	ost << "\\newcommand{\\rmO}{{\\rm O}}\n";
	ost << "\\newcommand{\\rmP}{{\\rm P}}\n";
	ost << "\\newcommand{\\rmQ}{{\\rm Q}}\n";
	ost << "\\newcommand{\\rmR}{{\\rm R}}\n";
	ost << "\\newcommand{\\rmS}{{\\rm S}}\n";
	ost << "\\newcommand{\\rmT}{{\\rm T}}\n";
	ost << "\\newcommand{\\rmU}{{\\rm U}}\n";
	ost << "\\newcommand{\\rmV}{{\\rm V}}\n";
	ost << "\\newcommand{\\rmW}{{\\rm W}}\n";
	ost << "\\newcommand{\\rmX}{{\\rm X}}\n";
	ost << "\\newcommand{\\rmY}{{\\rm Y}}\n";
	ost << "\\newcommand{\\rmZ}{{\\rm Z}}\n";

	ost << "\\newcommand{\\bA}{{\\bf A}}\n";
	ost << "\\newcommand{\\bB}{{\\bf B}}\n";
	ost << "\\newcommand{\\bC}{{\\bf C}}\n";
	ost << "\\newcommand{\\bD}{{\\bf D}}\n";
	ost << "\\newcommand{\\bE}{{\\bf E}}\n";
	ost << "\\newcommand{\\bF}{{\\bf F}}\n";
	ost << "\\newcommand{\\bG}{{\\bf G}}\n";
	ost << "\\newcommand{\\bH}{{\\bf H}}\n";
	ost << "\\newcommand{\\bI}{{\\bf I}}\n";
	ost << "\\newcommand{\\bJ}{{\\bf J}}\n";
	ost << "\\newcommand{\\bK}{{\\bf K}}\n";
	ost << "\\newcommand{\\bL}{{\\bf L}}\n";
	ost << "\\newcommand{\\bM}{{\\bf M}}\n";
	ost << "\\newcommand{\\bN}{{\\bf N}}\n";
	ost << "\\newcommand{\\bO}{{\\bf O}}\n";
	ost << "\\newcommand{\\bP}{{\\bf P}}\n";
	ost << "\\newcommand{\\bQ}{{\\bf Q}}\n";
	ost << "\\newcommand{\\bR}{{\\bf R}}\n";
	ost << "\\newcommand{\\bS}{{\\bf S}}\n";
	ost << "\\newcommand{\\bT}{{\\bf T}}\n";
	ost << "\\newcommand{\\bU}{{\\bf U}}\n";
	ost << "\\newcommand{\\bV}{{\\bf V}}\n";
	ost << "\\newcommand{\\bW}{{\\bf W}}\n";
	ost << "\\newcommand{\\bX}{{\\bf X}}\n";
	ost << "\\newcommand{\\bY}{{\\bf Y}}\n";
	ost << "\\newcommand{\\bZ}{{\\bf Z}}\n";

#if 0
	ost << "\\newcommand{\\sA}{{\\mathscr A}}\n";
	ost << "\\newcommand{\\sB}{{\\mathscr B}}\n";
	ost << "\\newcommand{\\sC}{{\\mathscr C}}\n";
	ost << "\\newcommand{\\sD}{{\\mathscr D}}\n";
	ost << "\\newcommand{\\sE}{{\\mathscr E}}\n";
	ost << "\\newcommand{\\sF}{{\\mathscr F}}\n";
	ost << "\\newcommand{\\sG}{{\\mathscr G}}\n";
	ost << "\\newcommand{\\sH}{{\\mathscr H}}\n";
	ost << "\\newcommand{\\sI}{{\\mathscr I}}\n";
	ost << "\\newcommand{\\sJ}{{\\mathscr J}}\n";
	ost << "\\newcommand{\\sK}{{\\mathscr K}}\n";
	ost << "\\newcommand{\\sL}{{\\mathscr L}}\n";
	ost << "\\newcommand{\\sM}{{\\mathscr M}}\n";
	ost << "\\newcommand{\\sN}{{\\mathscr N}}\n";
	ost << "\\newcommand{\\sO}{{\\mathscr O}}\n";
	ost << "\\newcommand{\\sP}{{\\mathscr P}}\n";
	ost << "\\newcommand{\\sQ}{{\\mathscr Q}}\n";
	ost << "\\newcommand{\\sR}{{\\mathscr R}}\n";
	ost << "\\newcommand{\\sS}{{\\mathscr S}}\n";
	ost << "\\newcommand{\\sT}{{\\mathscr T}}\n";
	ost << "\\newcommand{\\sU}{{\\mathscr U}}\n";
	ost << "\\newcommand{\\sV}{{\\mathscr V}}\n";
	ost << "\\newcommand{\\sW}{{\\mathscr W}}\n";
	ost << "\\newcommand{\\sX}{{\\mathscr X}}\n";
	ost << "\\newcommand{\\sY}{{\\mathscr Y}}\n";
	ost << "\\newcommand{\\sZ}{{\\mathscr Z}}\n";
#else
	ost << "\\newcommand{\\sA}{{\\cal A}}\n";
	ost << "\\newcommand{\\sB}{{\\cal B}}\n";
	ost << "\\newcommand{\\sC}{{\\cal C}}\n";
	ost << "\\newcommand{\\sD}{{\\cal D}}\n";
	ost << "\\newcommand{\\sE}{{\\cal E}}\n";
	ost << "\\newcommand{\\sF}{{\\cal F}}\n";
	ost << "\\newcommand{\\sG}{{\\cal G}}\n";
	ost << "\\newcommand{\\sH}{{\\cal H}}\n";
	ost << "\\newcommand{\\sI}{{\\cal I}}\n";
	ost << "\\newcommand{\\sJ}{{\\cal J}}\n";
	ost << "\\newcommand{\\sK}{{\\cal K}}\n";
	ost << "\\newcommand{\\sL}{{\\cal L}}\n";
	ost << "\\newcommand{\\sM}{{\\cal M}}\n";
	ost << "\\newcommand{\\sN}{{\\cal N}}\n";
	ost << "\\newcommand{\\sO}{{\\cal O}}\n";
	ost << "\\newcommand{\\sP}{{\\cal P}}\n";
	ost << "\\newcommand{\\sQ}{{\\cal Q}}\n";
	ost << "\\newcommand{\\sR}{{\\cal R}}\n";
	ost << "\\newcommand{\\sS}{{\\cal S}}\n";
	ost << "\\newcommand{\\sT}{{\\cal T}}\n";
	ost << "\\newcommand{\\sU}{{\\cal U}}\n";
	ost << "\\newcommand{\\sV}{{\\cal V}}\n";
	ost << "\\newcommand{\\sW}{{\\cal W}}\n";
	ost << "\\newcommand{\\sX}{{\\cal X}}\n";
	ost << "\\newcommand{\\sY}{{\\cal Y}}\n";
	ost << "\\newcommand{\\sZ}{{\\cal Z}}\n";
#endif

	ost << "\\newcommand{\\frakA}{{\\mathfrak A}}\n";
	ost << "\\newcommand{\\frakB}{{\\mathfrak B}}\n";
	ost << "\\newcommand{\\frakC}{{\\mathfrak C}}\n";
	ost << "\\newcommand{\\frakD}{{\\mathfrak D}}\n";
	ost << "\\newcommand{\\frakE}{{\\mathfrak E}}\n";
	ost << "\\newcommand{\\frakF}{{\\mathfrak F}}\n";
	ost << "\\newcommand{\\frakG}{{\\mathfrak G}}\n";
	ost << "\\newcommand{\\frakH}{{\\mathfrak H}}\n";
	ost << "\\newcommand{\\frakI}{{\\mathfrak I}}\n";
	ost << "\\newcommand{\\frakJ}{{\\mathfrak J}}\n";
	ost << "\\newcommand{\\frakK}{{\\mathfrak K}}\n";
	ost << "\\newcommand{\\frakL}{{\\mathfrak L}}\n";
	ost << "\\newcommand{\\frakM}{{\\mathfrak M}}\n";
	ost << "\\newcommand{\\frakN}{{\\mathfrak N}}\n";
	ost << "\\newcommand{\\frakO}{{\\mathfrak O}}\n";
	ost << "\\newcommand{\\frakP}{{\\mathfrak P}}\n";
	ost << "\\newcommand{\\frakQ}{{\\mathfrak Q}}\n";
	ost << "\\newcommand{\\frakR}{{\\mathfrak R}}\n";
	ost << "\\newcommand{\\frakS}{{\\mathfrak S}}\n";
	ost << "\\newcommand{\\frakT}{{\\mathfrak T}}\n";
	ost << "\\newcommand{\\frakU}{{\\mathfrak U}}\n";
	ost << "\\newcommand{\\frakV}{{\\mathfrak V}}\n";
	ost << "\\newcommand{\\frakW}{{\\mathfrak W}}\n";
	ost << "\\newcommand{\\frakX}{{\\mathfrak X}}\n";
	ost << "\\newcommand{\\frakY}{{\\mathfrak Y}}\n";
	ost << "\\newcommand{\\frakZ}{{\\mathfrak Z}}\n";

	ost << "\\newcommand{\\fraka}{{\\mathfrak a}}\n";
	ost << "\\newcommand{\\frakb}{{\\mathfrak b}}\n";
	ost << "\\newcommand{\\frakc}{{\\mathfrak c}}\n";
	ost << "\\newcommand{\\frakd}{{\\mathfrak d}}\n";
	ost << "\\newcommand{\\frake}{{\\mathfrak e}}\n";
	ost << "\\newcommand{\\frakf}{{\\mathfrak f}}\n";
	ost << "\\newcommand{\\frakg}{{\\mathfrak g}}\n";
	ost << "\\newcommand{\\frakh}{{\\mathfrak h}}\n";
	ost << "\\newcommand{\\fraki}{{\\mathfrak i}}\n";
	ost << "\\newcommand{\\frakj}{{\\mathfrak j}}\n";
	ost << "\\newcommand{\\frakk}{{\\mathfrak k}}\n";
	ost << "\\newcommand{\\frakl}{{\\mathfrak l}}\n";
	ost << "\\newcommand{\\frakm}{{\\mathfrak m}}\n";
	ost << "\\newcommand{\\frakn}{{\\mathfrak n}}\n";
	ost << "\\newcommand{\\frako}{{\\mathfrak o}}\n";
	ost << "\\newcommand{\\frakp}{{\\mathfrak p}}\n";
	ost << "\\newcommand{\\frakq}{{\\mathfrak q}}\n";
	ost << "\\newcommand{\\frakr}{{\\mathfrak r}}\n";
	ost << "\\newcommand{\\fraks}{{\\mathfrak s}}\n";
	ost << "\\newcommand{\\frakt}{{\\mathfrak t}}\n";
	ost << "\\newcommand{\\fraku}{{\\mathfrak u}}\n";
	ost << "\\newcommand{\\frakv}{{\\mathfrak v}}\n";
	ost << "\\newcommand{\\frakw}{{\\mathfrak w}}\n";
	ost << "\\newcommand{\\frakx}{{\\mathfrak x}}\n";
	ost << "\\newcommand{\\fraky}{{\\mathfrak y}}\n";
	ost << "\\newcommand{\\frakz}{{\\mathfrak z}}\n";


	ost << "\\newcommand{\\Tetra}{{\\mathfrak Tetra}}\n";
	ost << "\\newcommand{\\Cube}{{\\mathfrak Cube}}\n";
	ost << "\\newcommand{\\Octa}{{\\mathfrak Octa}}\n";
	ost << "\\newcommand{\\Dode}{{\\mathfrak Dode}}\n";
	ost << "\\newcommand{\\Ico}{{\\mathfrak Ico}}\n";

	ost << "\\newcommand{\\bbF}{{\\mathbb F}}\n";
	ost << "\\newcommand{\\bbQ}{{\\mathbb Q}}\n";
	ost << "\\newcommand{\\bbC}{{\\mathbb C}}\n";
	ost << "\\newcommand{\\bbR}{{\\mathbb R}}\n";

	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex\n";
	ost << endl;
	ost << "\\begin{document} \n";
	ost << "\\setTBstruts" << endl;
	ost << endl;
	ost << "\\bibliographystyle{plain}\n";
	if (!f_pagenumbers) {
		ost << "\\pagestyle{empty}\n";
		}
	ost << "%\\large\n";
	ost << endl;
	ost << "{\\allowdisplaybreaks%\n";
	ost << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex\n";
	ost << endl;
	ost << "%\\renewcommand{\\labelenumi}{(\\roman{enumi})}\n";
	ost << endl;

	if (f_title) {
		ost << "\\title{" << title << "}\n";
		ost << "\\author{" << author << "}%end author\n";
		ost << "%\\date{}\n";
		ost << "\\maketitle%\n";
		}
	ost << "\\pagenumbering{roman}\n";
	ost << "%\\thispagestyle{empty}\n";
	if (f_toc) {
		ost << "\\tableofcontents\n";
		}
	ost << "%\\input et.tex%\n";
	ost << "%\\thispagestyle{empty}%\\phantom{page2}%\\clearpage%\n";
	ost << "%\\addcontentsline{toc}{chapter}{Inhaltsverzeichnis}%\n";
	ost << "%\\tableofcontents\n";
	ost << "%\\listofsymbols\n";
	if (f_toc){
		ost << "\\clearpage\n";
		ost << endl;
		}
	ost << "\\pagenumbering{arabic}\n";
	ost << "%\\pagenumbering{roman}\n";
	ost << endl;
	ost << endl;
	ost << endl;
}


void latex_interface::foot(ostream& ost)
{
	ost << endl;
	ost << endl;
	ost << "%\\bibliographystyle{gerplain}% wird oben eingestellt\n";
	ost << "%\\addcontentsline{toc}{section}{References}\n";
	ost << "%\\bibliography{../MY_BIBLIOGRAPHY/anton}\n";
	ost << "% ACHTUNG: nicht vergessen:\n";
	ost << "% die Zeile\n";
	ost << "%\\addcontentsline{toc}{chapter}{Literaturverzeichnis}\n";
	ost << "% muss per Hand in d.bbl eingefuegt werden !\n";
	ost << "% nach \\begin{thebibliography}{100}\n";
	ost << endl;
	ost << "%\\begin{theindex}\n";
	ost << endl;
	ost << "%\\clearpage\n";
	ost << "%\\addcontentsline{toc}{chapter}{Index}\n";
	ost << "%\\input{apd.ind}\n";
	ost << endl;
	ost << "%\\printindex\n";
	ost << "%\\end{theindex}\n";
	ost << endl;
	ost << "}% allowdisplaybreaks\n";
	ost << endl;
	ost << "\\end{document}\n";
	ost << endl;
	ost << endl;
}




// two functions from DISCRETA1:
// adapted to use ostream instead of FILE pointer

void latex_interface::incma_latex_picture(ostream &fp,
	int width, int width_10,
	int f_outline_thin, const char *unit_length,
	const char *thick_lines,
	const char *thin_lines,
	const char *geo_line_width,
	int v, int b,
	int V, int B, int *Vi, int *Bj,
	int *R, int *X, int dim_X,
	int f_labelling_points, const char **point_labels,
	int f_labelling_blocks, const char **block_labels)
// width for one box in 0.1mm
// width_10 is 1 10th of width
// example: width = 40, width_10 = 4
{
	int w, h, w1, h1;
	int i, j, k, r, a;
	int x0, y0, x1, y1;
	int X0, Y0, X1, Y1;
	int width_8, width_5;
	const char *tdo_line_width = thick_lines; // "0.7mm";
	const char *line_width = thin_lines; // "0.15mm";
	// char *geo_line_width = "0.25mm";

	width_8 = width - 2 * width_10;
	width_5 = width >> 1;
	fp << "\\unitlength" << unit_length << endl;
	w = b * width;
	h = v * width;
	w1 = w;
	h1 = h;
	if (f_labelling_points) {
		w1 += 2 * width;
		}
	if (f_labelling_blocks) {
		h1 += 2 * width;
		}
	fp << "\\begin{picture}(" << w1 << "," << h1 << ")" << endl;

	// the grid:
	fp << "\\linethickness{" << tdo_line_width << "}" << endl;
	k = 0;
	for (i = -1; i < B; i++) {
		if (i >= 0) {
			a = Bj[i];
			k += a;
			}
		if (f_outline_thin) {
			if (i == -1 || i == B - 1) {
				continue;
				}
			}
		fp << "\\put(" << k * width << ",0){\\line(0,1){"
				<< h << "}}" << endl;
		}
	if (k != b) {
		cout << "incma_latex_picture: k != b" << endl;
		exit(1);
		}
	k = 0;
	for (i = -1; i < V; i++) {
		if (i >= 0) {
			a = Vi[i];
			k += a;
			}
		if (f_outline_thin) {
			if (i == -1 || i == V - 1) {
				continue;
				}
			}
		fp << "\\put(0," << h - k * width << "){\\line(1,0){"
				<< w << "}}" << endl;
		}
	if (k != v) {
		cout << "incma_latex_picture: k != v" << endl;
		exit(1);
		}

	// labeling of points:
	if (f_labelling_points) {
		for (i = 0; i < v; i++) {
			fp << "\\put(0," << h - i * width - width_5
				<< "){\\makebox(0,0)[r]{"
				<< point_labels[i] << "$\\,$}}" << endl;
			}
		}
	else {
		for (i = 0; i < v; i++) {
			fp << "\\put(0," << h - i * width - width_5
				<< "){\\makebox(0,0)[r]{}}" << endl;
			}
		}

	// labeling of blocks:
	if (f_labelling_blocks) {
		for (i = 0; i < b; i++) {
			fp << "\\put(" << i * width + width_5 << "," << h + width_5
				<< "){\\makebox(0,0)[b]{"
				<< block_labels[i] << "}}" << endl;
			}
		}
	else {
		for (i = 0; i < b; i++) {
			fp << "\\put(" << i * width + width_5 << "," << h + width_5
				<< "){\\makebox(0,0)[b]{}}" << endl;
			}
		}

	// the grid:
	fp << "\\linethickness{" << line_width << "}" << endl;
	fp << "\\multiput(0,0)(" << width << ",0){" << b + 1
			<< "}{\\line(0,1){" << h
		<< "}}" << endl;
	fp << "\\multiput(0,0)(0," << width << "){" << v + 1
			<< "}{\\line(1,0){" << w << "}}" << endl;

	// the incidence matrix itself:
	fp << "\\linethickness{" << geo_line_width << "}" << endl;
	for (i = 0; i < v; i++) {
		y0 = h - i * width;
		y1 = h - (i + 1) * width;
		Y0 = y0 - width_10;
		Y1 = y1 + width_10;
		for (r = 0; r < R[i]; r++) {
			j = X[i * dim_X + r];
			// printf("%d ", j);
			x0 = j * width;
			x1 = (j + 1) * width;
			X0 = x0 + width_10;
			X1 = x1 - width_10;
			// hor. lines:
			fp << "\\put(" << X0 << "," << Y0 << "){\\line(1,0){"
					<< width_8 << "}}" << endl;
			fp << "\\put(" << X0 << "," << Y1 << "){\\line(1,0){"
					<< width_8 << "}}" << endl;

			// vert. lines:
			fp << "\\put(" << X0 << "," << Y1 << "){\\line(0,1){"
					<< width_8 << "}}" << endl;
			fp << "\\put(" << X1 << "," << Y1 << "){\\line(0,1){"
					<< width_8 << "}}" << endl;

			}
		// printf("\n");
		}

	fp << "\\end{picture}" << endl;
}


static int incma_latex_unit_length_nb = 0;
static const char *incma_latex_unit_length[100];



void latex_interface::incma_latex(ostream &fp,
	int v, int b,
	int V, int B, int *Vi, int *Bj,
	int *R, int *X, int dim_X)
{
	const char *unit_length;


	if (incma_latex_unit_length_nb == 0) {
		unit_length = "0.065mm";
		}
	else {
		unit_length = incma_latex_unit_length[incma_latex_unit_length_nb - 1];
		}


	incma_latex_picture(fp,
		40 /* width */,
		10 /* width_10 */,
		FALSE /* f_outline_thin */,
		unit_length /* unit_length */,
		"0.5mm" /* thick_lines */ ,
		"0.15mm" /* thin_lines */ ,
		"0.25mm" /* geo_line_width */ ,
		v, b, V, B, Vi, Bj, R, X, dim_X,
		FALSE /* f_labelling_points */, NULL,
		FALSE /* f_labelling_blocks */, NULL);
}


void latex_interface::incma_latex_override_unit_length(
		const char *override_unit_length)
{
	incma_latex_unit_length[incma_latex_unit_length_nb++] =
			override_unit_length;
}

void latex_interface::incma_latex_override_unit_length_drop()
{
	incma_latex_unit_length_nb--;
}


}}

