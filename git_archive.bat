set branchname=%1

git tag archive/%branchname% origin/%branchname%
git branch -D origin/%branchname%
git branch -d -r origin/%branchname%
git push --tags
git push origin --delete %branchname%
