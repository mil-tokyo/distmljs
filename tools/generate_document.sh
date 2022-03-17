#!/bin/bash

OUTDIR=_docbuild

cd "${0%/*}"
cd ..

rm -r $OUTDIR

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
cp -r distributed/docs/_build _docbuild/server

# zip
rm -rf /tmp/kakiage-document
cp -a _docbuild /tmp/kakiage-document
pushd /tmp
rm kakiage-document.zip
zip -r kakiage-document.zip kakiage-document
rm -rf kakiage-document
popd
mv /tmp/kakiage-document.zip .
