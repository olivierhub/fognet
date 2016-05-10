This is the code to reproduce the best submit on the fognet challenge
It runs with python2.7 on fedora 23



I) Environment

In order to reproduce easily the environment you can run a docker image:

git clone https://github.com/olivierhub/dockerfiles.git
cd dockerfiles/fedora-sklearn
docker builder -t fedora_sklearn .

And then you can run bash inside the container:
docker run -i -t fedora_sklearn bash

If you don't want to run inside the container you can try installing the following packages:
# fedora packaged python27 modules
dnf install -y python-pip python-scikit-learn opencv-python python-pandas python-psycopg2 python-pillow python-scikit-image python-statsmodels python-matplotlib python-pandas python-scikit-image python-setuptools python-crypto python-backports-ssl_match_hostname python-theano h5py pyephem
# Unpackaged python tools
pip install nolearn ; pip install lasagne ; pip install xgboost ; pip install holidays; pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git ; pip install seaborn

II) Run the code
git clone https://github.com/olivierhub/fognet
cd fognet
python fognet2.py

Depending on your hardware you will wait between 3 and 48 hours. A lot of time is spent running xgboost, which benefits from parallelization, so running, say, on an aws compute instance with 36 vcpus will be about 20 times faster than on your dual core laptop.

III) Result
At the end of the run a file:
fognet_sub2_2.011573.csv
will be generated.
fognet_sub2_source_2.011573.csv is the same file with the "yield_source" column added, with indicates for each submission line the "eval_set" which was used.

The program generates a lot of models builded with the aggregation of various source ( micro, macro data ). 
This is done as followed:
  add_eval_sets( ( list of aggregated sources ), set_name , .. )
   => A search of good hyperparameters is done for this "mini model"

  compare_evaluations2(evaluations_list, .. )
   => the "mini models" ( eval sets ) are ranked thru a 4 fold evaluation process

  generate_valid_sub2(evaluations_list)
   => A submission is generated using the "best model" available for each line


IV) Discussion

The main difficulty for this challenge is that there is a lot of missing data. There is not a single data source which covers the whole period. So instead of making "forecasts of forecasts", we use the best data we have for each submission line.
When micro data is available, the model using it always beat models with only macro data.
When no micro data is available, some combinaisons of the various macro data allow to make a prediction.

The program allows to quickly try different combinations of macro and micro data, and automatically generates a submission using the best performing combination for each line

V) Bugs
GenerateGroupsBy() is buggy because it inputs future data. One I realised it I fixed it and improved the algorithms further, but the best submission I had was at 3.3938 .

VI) Possible enhancements

VI.1) Data
The main issue is the missing micro data.
Worse: the valuable macro data is the one from guelmin and sidi ( because it is close to the fognets probably ), but this macro data is available only at working hours! And of course the fog collection occurs mainly at night.
So for about half the solution we are prediction values for the 18:00 - 06:00 time range with only agadir data.
Having at least better data for the 18:00 - 06:00 period will certainly help. For example sea current and sea temperature would help, but I did not find a source for these.

VI.2) Model / Algorithms
Instead of using xgboost I did some tests with an mlp network, and I was able to observe improvements, at the price of a *very* long compute time ( even with a decent graphic card ).

It was very difficult to have a reliable estimation of the submission score when the solution was scoring better than 3.4.
Probably using more than 4 folds would have helped, also I did a projected scoring based only on a test set, it would have been better to compute an average of folded scores.
My parsing of the macro data could certainly be enhanced. The WeatherParse() functions compute an arbitrary "degre of humidity". 
Also I frequently copy the previous values in order to input some missing values, it is probably possible to make something better.
Scaling data and using null values  always produced worse results.

The whole point of the programm is to overcome the issue of missing data. If the problem was not time-related it would certainly be possible to make good estimation using averages / splines etc, but this would systematically induce future data. Also use "historical means" would probably mean injecting more subtly future data.

What I did in further models was to split the models in "night" models and "day" models. The "night" models had few data but performed better. This is probably because a model predicting "24h" benefits from its good "day" performance on the ranking process.

If I had more time I would have try models based only on the last available data at 18:00 in order to predict the 20:00 - 04:00 period, in order to have very specialized models.

Also the 06:00 prediction can probably be enhanced thanks to "min / max" during the night available only at 6:00.



