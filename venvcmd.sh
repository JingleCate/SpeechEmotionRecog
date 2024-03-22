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

#    git tips
# commit standard
# keywords: feat/fix/optim/modify

# 推送到gitee
git push -f git@gitee.com:jinglecath/SpeechEmotionRecog.git

# 运行
python train.py --help

python train.py --epochs 5000 --batch_size 4 --use_checkpoint True --checkpoint_path  "checkpoints/SSR_epoch_1200.pth"
