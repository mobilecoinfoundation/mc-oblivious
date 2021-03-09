#!/bin/sh
set -e
set -x

ROOT=`git rev-parse --show-toplevel`

# symlink the precommit hook into .git/hooks
# see https://stackoverflow.com/questions/4592838/symbolic-link-to-a-hook-in-git
cd "$ROOT/.git/hooks" && ln -s -f "../../hooks/pre-commit" .

# define a 'theirs' merge driver which uses unix false utility to always choose
# theirs. In .gitattributes we apply this to Cargo.lock files
git config --local merge.theirs.driver false
