% file: PGGL_2_8_OnConic_sims_1.mp
% created by Orbiter metapost interface
% creation date: Sat Dec 19 23:54:03 MST 2020

input boxes

verbatimtex
\documentclass[10pt]{article}
\usepackage[noBBpl]{mathpazo}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\scriptsize
etex;
u=0.100000mm;
beginfig(1);
path p[];
draw (500u ,958.3u )--(500u ,875u );
draw (500u ,875u )--(500u ,791.6u );
draw (500u ,791.6u )--(200u ,708.3u );
draw (500u ,791.6u )--(700u ,708.3u );
draw (200u ,708.3u )--(200u ,625u );
draw (700u ,708.3u )--(700u ,625u );
draw (700u ,625u )--(700u ,541.6u );
path pp;
pp = (525u ,958.3u ) .. (500u ,983.3u ) .. (475u ,958.3u ) .. (500u ,933.3u ) .. (525u ,958.3u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,958.3u ) .. (500u ,983.3u ) .. (475u ,958.3u ) .. (500u ,933.3u ) .. (525u ,958.3u ) .. cycle;
draw pp;
label(btex {\scriptsize 1} etex, (500u ,958.3u ));
path pp;
pp = (525u ,875u ) .. (500u ,900u ) .. (475u ,875u ) .. (500u ,850u ) .. (525u ,875u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,875u ) .. (500u ,900u ) .. (475u ,875u ) .. (500u ,850u ) .. (525u ,875u ) .. cycle;
draw pp;
label(btex {\scriptsize 5} etex, (500u ,875u ));
path pp;
pp = (525u ,791.6u ) .. (500u ,816.6u ) .. (475u ,791.6u ) .. (500u ,766.6u ) .. (525u ,791.6u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,791.6u ) .. (500u ,816.6u ) .. (475u ,791.6u ) .. (500u ,766.6u ) .. (525u ,791.6u ) .. cycle;
draw pp;
label(btex {\scriptsize 8} etex, (500u ,791.6u ));
path pp;
pp = (225u ,708.3u ) .. (200u ,733.3u ) .. (175u ,708.3u ) .. (200u ,683.3u ) .. (225u ,708.3u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (225u ,708.3u ) .. (200u ,733.3u ) .. (175u ,708.3u ) .. (200u ,683.3u ) .. (225u ,708.3u ) .. cycle;
draw pp;
label(btex {\scriptsize 3} etex, (200u ,708.3u ));
path pp;
pp = (725u ,708.3u ) .. (700u ,733.3u ) .. (675u ,708.3u ) .. (700u ,683.3u ) .. (725u ,708.3u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (725u ,708.3u ) .. (700u ,733.3u ) .. (675u ,708.3u ) .. (700u ,683.3u ) .. (725u ,708.3u ) .. cycle;
draw pp;
label(btex {\scriptsize 4} etex, (700u ,708.3u ));
path pp;
pp = (225u ,625u ) .. (200u ,650u ) .. (175u ,625u ) .. (200u ,600u ) .. (225u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (225u ,625u ) .. (200u ,650u ) .. (175u ,625u ) .. (200u ,600u ) .. (225u ,625u ) .. cycle;
draw pp;
label(btex {\scriptsize 7} etex, (200u ,625u ));
path pp;
pp = (725u ,625u ) .. (700u ,650u ) .. (675u ,625u ) .. (700u ,600u ) .. (725u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (725u ,625u ) .. (700u ,650u ) .. (675u ,625u ) .. (700u ,600u ) .. (725u ,625u ) .. cycle;
draw pp;
label(btex {\scriptsize 6} etex, (700u ,625u ));
path pp;
pp = (725u ,541.6u ) .. (700u ,566.6u ) .. (675u ,541.6u ) .. (700u ,516.6u ) .. (725u ,541.6u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (725u ,541.6u ) .. (700u ,566.6u ) .. (675u ,541.6u ) .. (700u ,516.6u ) .. (725u ,541.6u ) .. cycle;
draw pp;
label(btex {\scriptsize 2} etex, (700u ,541.6u ));
endfig;

end

