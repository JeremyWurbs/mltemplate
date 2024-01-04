#!/usr/bin/env bash
while getopts a:n:u:d: flag
do
    case "${flag}" in
        a) author=${OPTARG};;
        n) name=${OPTARG};;
        u) urlname=${OPTARG};;
        d) description=${OPTARG};;
    esac
done

echo "Author: $author";
echo "Project Name: $name";
echo "Project URL name: $urlname";
echo "Project Description: $description";

echo "Renaming project..."

original_author="Jeremy Wurbs"
original_name="mltemplate"
original_urlname="https://github.com/jeremywurbs/mltemplate/"
original_description="An end-to-end starter template for machine learning projects."

for filename in $(git ls-files)
do
    if [[ $filename == *"workflows"* ]]; then
        continue
    fi
    sed -i "s@$original_author@$author@g" $filename
    sed -i "s@${$original_name^}@${name^}@g" $filename
    sed -i "s@${original_name^^}@${name^^}@g" $filename
    sed -i "s@$original_name@$name@g" $filename
    sed -i "s@$original_urlname@$urlname@g" $filename
    sed -i "s@$original_description@$description@g" $filename
    echo "Renamed $filename"
done

mv mltemplate $name

# This command runs only once on GHA!
rm -rf .github/template.yml
