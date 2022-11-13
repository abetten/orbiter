

for f in $(find -path ./.git -prune -o -name '*.cpp' -print); 
do 
    cat <(echo "
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    ") $f > $f.bk;
    rm $f;
    mv $f.bk $f;
done