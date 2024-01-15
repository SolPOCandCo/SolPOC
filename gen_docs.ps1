pdoc --html --force -o docs/html solpoc
pdoc --pdf solpoc > docs/solpoc_doc.md
# pandoc --metadata=title:"SolPOC Function Documentation"  --toc --toc-depth=4 --from=markdown+abbreviations  --pdf-engine=xelatex --variable=mainfont:"DejaVu Sans" --output=docs/solpoc_doc.pdf docs/solpoc_doc.md