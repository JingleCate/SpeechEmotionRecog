
#   ____                       _       _____                 _   _               ____                
#  / ___| _ __   ___  ___  ___| |__   | ____|_ __ ___   ___ | |_(_) ___  _ __   |  _ \ ___  ___ __ _ 
#  \___ \| '_ \ / _ \/ _ \/ __| '_ \  |  _| | '_ ` _ \ / _ \| __| |/ _ \| '_ \  | |_) / _ \/ __/ _` |
#   ___) | |_) |  __/  __/ (__| | | | | |___| | | | | | (_) | |_| | (_) | | | | |  _ <  __/ (_| (_| |
#  |____/| .__/ \___|\___|\___|_| |_| |_____|_| |_| |_|\___/ \__|_|\___/|_| |_| |_| \_\___|\___\__, |
#        |_|                                                                                   |___/ 


# ------------------------ python venv -----------------------
# Virtual environment 
python -m venv myenv

# Activate virtual env
# Linux
source myenv/bin/activate    # deactivate
# In CMD
myenv/Scripts/activate.bat
# In Powershel
myenv/Scripts/Activate.ps1


# Export dependences
pip freeze > requirements.txt

# Install dependences
pip install -r requirements.txt
pip list 
pip show ***

# ------------ git tips: commit standard规范 -----------------

# <type>(<scope>): <subject>
# type(必须)
# 用于说明git commit的类别，只允许使用下面的标识。
    # feat：新功能（feature）。
    # fix/to：修复bug，可以是QA发现的BUG，也可以是研发自己发现的BUG。
    # fix：产生diff并自动修复此问题。适合于一次提交直接修复问题
    # to：只产生diff不自动修复此问题。适合于多次提交。最终修复问题提交时使用fix
    # docs：文档（documentation）。
    # style：格式（不影响代码运行的变动）。
    # refactor：重构（即不是新增功能，也不是修改bug的代码变动）。
    # perf：优化相关，比如提升性能、体验。
    # test：增加测试。
    # chore：构建过程或辅助工具的变动。
    # revert：回滚到上一个版本。
    # merge：代码合并。
    # sync：同步主线或分支的Bug。
# scope(可选) scope用于说明 commit 影响的范围，比如数据层、控制层、视图层等等，视项目不同而不同。


# 推送到gitee
git push -f git@gitee.com:jinglecath/SpeechEmotionRecog.git

# 强制覆盖本地
git fetch --all
git reset --hard origin/main
git pull



# ------------------------ train & run ------------------------
# 运行 加 & 后台运行
python train.py --help
# don't use checkpoint
python train.py -e 1000 -b 16 -lr 1e-4 -p 10
# use checkpoint
python train.py -e 5000 -b 16 -r True -chp  checkpoints/SSR_epoch_3400_acc_0.576_2nd.pth

# delete specified files except for some files in linux.
find ./checkpoints | grep -v '\(f1.txt \| f2.txt\)' | xargs rm