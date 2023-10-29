
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

export CUDA_VISIBLE_DEVICES=0,1

