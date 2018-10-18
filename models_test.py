# this is our big python script that defines all of our functions
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sklearn
import flask
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import networkx as nx
from sklearn.decomposition import PCA
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

############################## make the plots look purty ######################

#to make fonts from plots look normal
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Calibri'
mpl.rcParams['mathtext.it'] = 'Calibri:italic'
mpl.rcParams['mathtext.bf'] = 'Calibri:bold'

font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 36}
        #'sans-serif' : 'Arial Unicode MS'}
mpl.rc('font', **font)
plt.rc('font', size=36)          # controls default text sizes

#mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['axes.linewidth'] = 1.5 #set the value globally
mpl.rcParams['lines.markersize'] = 16
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#mpl.rcParams['xtick.top'] = False
#mpl.rcParams['ytick.right'] = False

############################# fin ##########################################

staticDir = r'/home/eric/Documents/Insight/WebApp/static'
os.chdir(staticDir)

# open image path array
with open('imagePathArray.pickle', 'rb') as file:
    imagePathList = pickle.load(file)

# open font family array
with open('fontFamily.pickle', 'rb') as file:
    fontFamily = pickle.load(file)

# open dimensionally reduced feature vectors
with open('featureDimRed.pickle', 'rb') as file:
    featureDimRed = pickle.load(file)

# open full feature vector file
with open('featureVecsFull.pickle', 'rb') as file:
    featureVecFull = pickle.load(file)

# open CLEANED list of font names
with open('fullNames.pickle', 'rb') as file:
    NameVec = pickle.load(file)

# open tSNE results
with open('tsne_results2.pickle', 'rb') as file:
    tSNE_vecs = pickle.load(file)

# open weights arrays
with open('weight_array', 'rb') as file:
    weights = pickle.load(file)

##### define the weighted cosine similarity metric #########
def weighted_cosine_metric(fn1,fn2,weights):
    '''
    function that determines the contrast metric for generating font recommendations.
    assume fn2 is a large array. the first step is applying the function along each row.
    the parameter u is the weight of the contrasting part. start small--maybe 0.1
    '''
    # calculate element wise product of the 2 arrays
    f_tot = fn1*fn2*weights

    #calculate row-wise sum of the negative and positive arrays
    f_sum = np.sum(f_tot,axis=1)

    # normalize the font array (# of items = # of rows)
    norm_fn2 = np.linalg.norm(fn2,axis = 1)
    # normalize the input font vector
    norm_fn1 = np.linalg.norm(fn1)

    calc_sim = []
    # for each entry, compute the weighted similarity.
    for norm in range(len(fn2)):
        calc_sim.append(f_sum[norm]/(norm_fn1*norm_fn2[norm]))


    calc_sim_array = np.asarray(calc_sim)

    return calc_sim_array


def model_output(user_select,NameVec,featureDimRed,tSNE_vecs,imagePathList,weights):

    # extract idx of user font
    userIdx = [idx for idx,s in enumerate(NameVec) if user_select in s]
    userIdx = np.asarray(userIdx)

    #perform cosine similarity and obtain rankings and scores
    simFonts = weighted_cosine_metric(featureDimRed[userIdx[0]],featureDimRed, weights)
    recResults = (-simFonts).argsort()[2:6]
    # print(recResults)
    scoreResults = simFonts[recResults]
    # print(scoreResults)

    # return font names
    fullName2 = np.asarray(NameVec)
    nameResult = fullName2[recResults]
    # print(nameResult)

    # find the font images
    imagePathArray2 = np.asarray(imagePathList)
    imgPaths = imagePathArray2[recResults]


    # convert the outputs to a list
    nameList = nameResult.tolist()
    scoreList = scoreResults.tolist()
    scoreList = [ '%.5f' % score for score in scoreList ]
    imgList = imgPaths.tolist()
    imgList2 =  [path[43:] for path in imgPaths]

    #also grab user input path
    inputPath = imagePathArray2[userIdx[0]]
    inputPath2 =  inputPath[43:]

    return(nameList, scoreList, imgList2, inputPath2, userIdx)

################# function to make the tSNE plot ############################
# make tSNE figure

def makeplot(tSNE_vecs, userIdx, ):
fig, ax = plt.subplots(1,1, figsize = (8,7))
for ft in range(len(tSNE_vecs)):
    ax.plot(tSNE_vecs[ft,0],tSNE_vecs[ft,1],'o', markersize = 10, c = 'k', alpha = 0.1)

ax.plot(tSNE_vecs[userIdx[0],0],tSNE_vecs[userIdx[0],1],'o', markersize = 20, c = 'r')
ax.plot(tSNE_vecs[recResults,0],tSNE_vecs[recResults,1],'o', markersize = 10, c = 'b', alpha = 0.9)

ax.set_xlabel("tSNE axis 1")
ax.set_ylabel("tSNE axis 2")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_title("|Z| vs. $\omega$")
ax.set_xlim([-150,150]);
ax.set_ylim([-150,150]);
figDir = '/home/eric/Documents/Insight/WebApp/static/'
figName = 'tSNE_figResult.png'
plt.savefig(figDir + figName,dpi = 300, bbox_inches = 'tight')
