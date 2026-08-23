% file: poset_draw_tree.mp
% created by Orbiter metapost interface
% creation date: Mon Nov 25 12:41:01 MST 2019

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
draw (500u ,958.333u )--(500u ,875u );
draw (500u ,875u )--(500u ,791.666u );
draw (500u ,791.666u )--(500u ,708.333u );
draw (500u ,708.333u )--(428.571u ,625u );
draw (500u ,708.333u )--(928.571u ,625u );
draw (428.571u ,625u )--(85.714u ,541.666u );
draw (428.571u ,625u )--(514.285u ,541.666u );
draw (514.285u ,541.666u )--(514.285u ,458.333u );
draw (514.285u ,458.333u )--(514.285u ,375u );
draw (514.285u ,375u )--(514.285u ,291.666u );
path pp;
pp = (520u ,958.333u ) .. (500u ,978.333u ) .. (480u ,958.333u ) .. (500u ,938.333u ) .. (520u ,958.333u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,958.333u ) .. (500u ,978.333u ) .. (480u ,958.333u ) .. (500u ,938.333u ) .. (520u ,958.333u ) .. cycle;
draw pp;
label(btex $\emptyset$ etex, (500u ,958.333u ));
path pp;
pp = (520u ,875u ) .. (500u ,895u ) .. (480u ,875u ) .. (500u ,855u ) .. (520u ,875u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,875u ) .. (500u ,895u ) .. (480u ,875u ) .. (500u ,855u ) .. (520u ,875u ) .. cycle;
draw pp;
label(btex 0 etex, (500u ,875u ));
path pp;
pp = (520u ,791.666u ) .. (500u ,811.666u ) .. (480u ,791.666u ) .. (500u ,771.666u ) .. (520u ,791.666u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,791.666u ) .. (500u ,811.666u ) .. (480u ,791.666u ) .. (500u ,771.666u ) .. (520u ,791.666u ) .. cycle;
draw pp;
label(btex 1 etex, (500u ,791.666u ));
path pp;
pp = (520u ,708.333u ) .. (500u ,728.333u ) .. (480u ,708.333u ) .. (500u ,688.333u ) .. (520u ,708.333u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,708.333u ) .. (500u ,728.333u ) .. (480u ,708.333u ) .. (500u ,688.333u ) .. (520u ,708.333u ) .. cycle;
draw pp;
label(btex 2 etex, (500u ,708.333u ));
path pp;
pp = (448.571u ,625u ) .. (428.571u ,645u ) .. (408.571u ,625u ) .. (428.571u ,605u ) .. (448.571u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (448.571u ,625u ) .. (428.571u ,645u ) .. (408.571u ,625u ) .. (428.571u ,605u ) .. (448.571u ,625u ) .. cycle;
draw pp;
label(btex 3 etex, (428.571u ,625u ));
path pp;
pp = (948.571u ,625u ) .. (928.571u ,645u ) .. (908.571u ,625u ) .. (928.571u ,605u ) .. (948.571u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (948.571u ,625u ) .. (928.571u ,645u ) .. (908.571u ,625u ) .. (928.571u ,605u ) .. (948.571u ,625u ) .. cycle;
draw pp;
label(btex 8 etex, (928.571u ,625u ));
path pp;
pp = (105.714u ,541.666u ) .. (85.714u ,561.666u ) .. (65.714u ,541.666u ) .. (85.714u ,521.666u ) .. (105.714u ,541.666u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (105.714u ,541.666u ) .. (85.714u ,561.666u ) .. (65.714u ,541.666u ) .. (85.714u ,521.666u ) .. (105.714u ,541.666u ) .. cycle;
draw pp;
label(btex 4 etex, (85.714u ,541.666u ));
path pp;
pp = (534.285u ,541.666u ) .. (514.285u ,561.666u ) .. (494.285u ,541.666u ) .. (514.285u ,521.666u ) .. (534.285u ,541.666u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (534.285u ,541.666u ) .. (514.285u ,561.666u ) .. (494.285u ,541.666u ) .. (514.285u ,521.666u ) .. (534.285u ,541.666u ) .. cycle;
draw pp;
label(btex 8 etex, (514.285u ,541.666u ));
path pp;
pp = (534.285u ,458.333u ) .. (514.285u ,478.333u ) .. (494.285u ,458.333u ) .. (514.285u ,438.333u ) .. (534.285u ,458.333u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (534.285u ,458.333u ) .. (514.285u ,478.333u ) .. (494.285u ,458.333u ) .. (514.285u ,438.333u ) .. (534.285u ,458.333u ) .. cycle;
draw pp;
label(btex 11 etex, (514.285u ,458.333u ));
path pp;
pp = (534.285u ,375u ) .. (514.285u ,395u ) .. (494.285u ,375u ) .. (514.285u ,355u ) .. (534.285u ,375u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (534.285u ,375u ) .. (514.285u ,395u ) .. (494.285u ,375u ) .. (514.285u ,355u ) .. (534.285u ,375u ) .. cycle;
draw pp;
label(btex 13 etex, (514.285u ,375u ));
path pp;
pp = (534.285u ,291.666u ) .. (514.285u ,311.666u ) .. (494.285u ,291.666u ) .. (514.285u ,271.666u ) .. (534.285u ,291.666u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (534.285u ,291.666u ) .. (514.285u ,311.666u ) .. (494.285u ,291.666u ) .. (514.285u ,271.666u ) .. (534.285u ,291.666u ) .. cycle;
draw pp;
label(btex 14 etex, (514.285u ,291.666u ));
endfig;

end

