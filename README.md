# Speech-Accent-Recognition

### About
Every individual has their own dialects or mannerisms in which they speak. This project revolves around the detection of backgrounds of every individual using their speeches. The goal in this project is to classify various types of accents, specifically foreign accents, by the native language of the speaker. This project allows to detect the demographic and linguistic backgrounds of the speakers by comparing different speech outputs with the speech accent archive dataset in order to determine which variables are key predictors of each accent. The speech accent archive demonstrates that accents are systematic rather than merely mistaken speech. Given a recording of a speaker speaking a known script of English words, this project predicts the speaker’s native language.

### Dataset
All of the speech files used for this project come from the Speech Accent Archive, a repository of spoken English hosted by George Mason University. Over 2000 speakers representing over 100 native languages read a common elicitation paragraph in English:

```
'Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh
snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need 
a small plastic snake and a big toy frog for the kids. She can scoop these things into three red 
bags, and we will go meet her Wednesday at the train station.'
```

The common nature of the dataset makes it ideal for studying accent, being that the wording is provided, and the recording quality is (nearly) uniform across all speakers. Since the dataset was large in the terms of size (approximately 2GB) but the samples were less, so I worked mainly on 3 most spoken accents i.e. English, Mandarin and Arabic.

The dataset contained **.mp3** audio files which were converted to **.wav** audio files which allowed easy extraction of the **MFCC (Mel Frequency Cepstral Coefficients)** features to build a 2-D convolution neural network.

The MFCC was fed into a 2-Dimensional Convolutional Neural Network (CNN) to predict the native language class.

### Dependencies
• [Python 3.x](https://www.python.org/download/releases/2.7/)

• [Keras](https://keras.io/)

• [Numpy](http://www.numpy.org/)

• [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) 

• [Pydub](https://github.com/jiaaro/pydub)

• [Scikit-learn](http://scikit-learn.org/stable/)

• [Librosa](http://librosa.github.io/librosa/)

### Execution
To execute the code, please have all the dependencies installed on your system. Next, change execution directory to the src directory of the code and execute the following python commands - 

• To download language metadata from [The Speech Accent Archive](http://accent.gmu.edu/index.php) and download audio files:
```
python fromwebsite.py bio_data.csv mandarin english arabic
```
• Run getaudio.py to download audio files to the audio directory. All audio files listed in bio_metadata.csv will be downloaded.
```
python getaudio.py bio_data.csv
```
• Run trainmodel.py to train the CNN.
```
python trainmodel.py bio_data.csv model5
```
### Results 
This project was able to achieve an accuracy of **over 75%** and used 13 MFCC features of each sample which were fed into the 2-D CNN.
The final accuracies are: -

• 96.6% when English samples were given 

• 65% when Arabic samples were given

• 50% when Mandarin samples were given

### References
• [Morgan Bryant, Amanda Chow & Sydney Li Classification of Accents of English Speakers by Native Language](http://cs229.stanford.edu/proj2014/Morgan%20Bryant,%20Amanda%20Chow,%20Sydney%20Li,%20Classification%20of%20Accents%20of%20English%20Speakers%20by%20Native%20Language.pdf)
