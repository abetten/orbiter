Orbiter
=======

A C++ class library devoted to the classification of combinatorial objects.
Besides that Orbiter offers many funtions in the field of computational group theory.
Orbiter offers a command line interface.
The User's Guide describes the command line language.
We recommend to use shell scripts or makefiles to assemble commands. 
An example makefile showing many commands is distributed with Orbiter.


Please find the User's Guide here:

- https://www.math.colostate.edu/~betten/orbiter/users_guide.pdf


There are two programmer's guides. Both are very rudimentary:

- https://www.math.colostate.edu/~betten/orbiter/orbiter_programmers_guide.pdf
- https://www.math.colostate.edu/~betten/orbiter/html/index.html



Requirements:
- C++ 17 compiler
- bison and flex and the flex library (for Sajeeb's expression parser)
- Windows users may use Windows subsystem linux to install Orbiter.

Please see the README in src/lib/layer1_foundations/expression_parser_sajeeb for 
hints regarding the installation of flex and bison.

Orbiter offers latex interfaces to the following external software packages:
- latex, tikz, metapost
- povray, ffmpeg
- magma
- GAP

Orbiter includes the following software packages (at the source code level)
- Nauty 2.7. (Brendan McKay)
- Jeffrey Leon's partition backtrack software Part
- possolve (Brendan McKay)
- Eigen
- EasyBMP
- DISCRETA (legacy code)

Statistics (as of January 2023):
- Total number of namespaces is 57
- Total number of classes is 552.
- Total number of lines of code is about 980 thousand (excluding the external software packages)

Anton Betten
January 30, 2023

