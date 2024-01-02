#!/usr/bin/env bash
overwrite_template_dir=0

while getopts t:o flag
do
    case "${flag}" in
        t) template=${OPTARG};;
        o) overwrite_template_dir=1;;
    esac
done

if [ -z "${template}" ]; then
    echo "Available templates: flask"
    read -p "Enter template name: " template
fi

repo_urlname=$(basename -s .git `git config --get remote.origin.url`)
repo_name=$(basename -s .git `git config --get remote.origin.url` | tr '-' '_' | tr '[:upper:]' '[:lower:]')
repo_owner=$(git config --get remote.origin.url | awk -F ':' '{print $2}' | awk -F '/' '{print $1}')
echo "Repo name: ${repo_name}"
echo "Repo owner: ${repo_owner}"
echo "Repo urlname: ${repo_urlname}"

if [ -f ".github/workflows/rename_project.yml" ]; then
    .github/rename_project.sh -a "${repo_owner}" -n "${repo_name}" -u "${repo_urlname}" -d "Awesome ${repo_name} created by ${repo_owner}"
fi
