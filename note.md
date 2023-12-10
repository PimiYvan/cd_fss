
git clone https://github.com/PimiYvan/cd_fss.git <br/>
chmod +x dataset.sh 
cd cd_fss
python3.9 -m venv env <br/>
source env/bin/activate <br/>
pip install --upgrade pip <br/>
pip install -r requirements.txt <br/>



if you installed a package you should run <br/>
pip freeze > requirements.txt <br/>
if you setting up the project you should run <br/>
pip install -r requirements.txt <br/>

salloc --time=1:0:0 --mem=3G --ntasks=2 --account=def-menna --gres=gpu:1 --nodes=1

export CUDA_VISIBLE_DEVICES=0,1

seff 12945221


seff 12945980
12947371

13022692
13022723
13022839
13075128

git clone https://github.com/slei109/PATNet.git

nano ~/.ssh/config

nano ~/.ssh/id_rsa
ssh -Tv graham.alliancecan.ca