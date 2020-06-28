PROJECT_NAME=mdp-sir-modelling

rm -rf ./*

git clone https://github.com/danielbdias/$PROJECT_NAME.git
mv ./$PROJECT_NAME/* ./
rm -rf ./$PROJECT_NAME
rm -rf ./sample_data
pip install -q -r requirements.txt
