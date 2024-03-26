python -m venv myenv

# Activate virtual env
# Linux
source venv/bin/activate    # deactivate
# In CMD
env/Scripts/activate.bat
# In Powershel
env/Scripts/Activate.ps1


# Export dependences
pip freeze > requirements.txt
pip install -r requirement.txt
pip list 
pip show ***

#    git tips
# commit standard
# keywords: feat/fix/optim/modify

# 推送到gitee
git push -f git@gitee.com:jinglecath/SpeechEmotionRecog.git

# 强制覆盖本地
git fetch --all
git reset --hard origin/main
git pull


# 运行 加 & 后台运行
python train.py --help

# don't use checkpoint
python train.py -e 1000 -b 16 -lr 1e-4 -p 10

# use checkpoint
python train.py -e 5000 -b 16 -r True -chp  checkpoints/SSR_epoch_3400_acc_0.576_2nd.pth

# delete specified files except for some files
find ./checkpoints | grep -v '\(f1.txt \| f2.txt\)' | xargs rm