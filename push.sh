# 0) 先把内容临时挪出去（防止 git rm 把目录删了）
mv DraCo /tmp/DraCo
mv Repoformer /tmp/Repoformer

# 1) 清理 submodule 记录
git submodule deinit -f DraCo Repoformer
git rm -f DraCo Repoformer
rm -rf .git/modules/DraCo .git/modules/Repoformer

# 2) 从 .gitmodules 里删除对应条目（用编辑器打开删掉 DraCo/Repoformer 那两段）
# 或者我也可以给你 sed 命令，但先手动删最稳

# 3) 把目录放回并删除它们各自内部的 .git（关键）
mv /tmp/DraCo DraCo
mv /tmp/Repoformer Repoformer
rm -rf DraCo/.git Repoformer/.git

# 4) 作为普通文件夹加入并提交
git add DraCo Repoformer .gitmodules
git commit -m "Vendor DraCo and Repoformer into main repo"
git push
