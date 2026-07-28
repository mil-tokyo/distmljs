#!/bin/bash
set -eu

OUTDIR=_docbuild

cd "${0%/*}"
cd ..

rm -rf $OUTDIR

# manually written manual
find ./docs -type f | while read -r p
do
relpath=${p#./docs}
mkdir -p ${OUTDIR}${relpath%/*}
echo $relpath
if [[ $relpath == *.md ]]
then
# docs/*.md to _docbuild/*.html
pandoc -f markdown -t html --standalone $p -o ${OUTDIR}${relpath%.md}.html
else
# docs/* (other than markdown; e.g. image) to _docbuild/*
cp -a $p ${OUTDIR}${relpath}
fi
done

# client library generation
npx typedoc src/index.ts --out _docbuild/client

# server library generation
cd distributed/docs
make html
cd ../..
# _build contains the html directory, and docs/index.md links to ./server
cp -r distributed/docs/_build/html _docbuild/server

# zip
rm -rf /tmp/distmljs-document
cp -a _docbuild /tmp/distmljs-document
pushd /tmp
rm -f distmljs-document.zip
zip -r distmljs-document.zip distmljs-document
rm -rf distmljs-document
popd
mv /tmp/distmljs-document.zip .
