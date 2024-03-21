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
# keywords: feat/fix/optim...

# 推送到gitee
git push -f git@gitee.com:jinglecath/SpeechEmotionRecog.git
