

from efficientnet_pytorch import EfficientNet
import glob
import numpy as np
import cv2  
import torch
import pickle
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from moviepy.editor import VideoFileClip
import pandas as pd

def getVidFramesAndFpsDur(vidName):
    video_clip = VideoFileClip(vidName)
    frmAll=list(video_clip.iter_frames(with_times=False))
    frmAll,fps,duration=frmAll,video_clip.fps,video_clip.duration
    del video_clip
    return frmAll,fps,duration

def getVisualFeatures(vid,visualModel,sampleRate):
    frmAll,fps,duration=getVidFramesAndFpsDur(vid)
    fps=int(fps)
    sampledFrames=[]
    nSamples=int(fps*.10)
    for i in range(int(duration)):
            smpl=np.sort(np.random.choice(np.arange(fps),nSamples,replace=False))
            for idx in smpl:
                sampledFrames.append(frmAll[i*fps:(i+1)*fps][idx])
    featuresArrMain=[]
    for f in range(len(sampledFrames)):                       
        frmRz=cv2.resize(sampledFrames[f], (224,224))
        frmRzPyTensor=torch.Tensor(frmRz.reshape([1, 3, 224, 224]))
        features = visualModel.extract_features(frmRzPyTensor)
        featuresArr=features.detach().numpy()
        featuresArr=featuresArr.reshape((1280,7,7))
        featuresArrMain.append(featuresArr)
    featuresArrMain=np.array(featuresArrMain)
    return  featuresArrMain,fps,duration,nSamples

def getVisualFramesResizeSample(vid,sampleRate,frameSize=(224,224)):
    frmAll,fps,duration=getVidFramesAndFpsDur(vid)
    fps=int(fps)
    sampledFrames=[]
    nSamples=int(fps*.10)
    for i in range(int(duration)):
            smpl=np.sort(np.random.choice(np.arange(fps),nSamples,replace=False))
            for idx in smpl:
                sampledFrames.append(frmAll[i*fps:(i+1)*fps][idx])
    framesMain=[]
    for f in range(len(sampledFrames)):                       
        frmRz=cv2.resize(sampledFrames[f], frameSize)
        framesMain.append(frmRz)
    framesMain=np.array(framesMain)
    return  framesMain,fps,duration,nSamples

    
def getAvgGt(gtDf,vidName):
    subGt=gtDf.loc[vidName]
    allUserArr=[]
    for i in range(len(subGt)):
        allUserArr.append(subGt.iloc[i,-1].split(","))
    allUserArr=np.array(allUserArr)
    allUserArr=allUserArr.astype(int)
    meanGt=allUserArr.mean(axis=0)
    return meanGt



def writeVisualFeaturesAlongGtForTvSum(groundTruth,basedir,baseOut,model,sampleRate=0.10):
    gtDf = pd.read_csv(groundTruth, sep="\t",header=None,index_col=0)
    allVideos=glob.glob(basedir)
    for v in range(len(allVideos)):
        vid=allVideos[v]
        vidName=vid.split("/")[-1].split(".")[0]
        meanGt=getAvgGt(gtDf,vidName)
        frmAll,fps,duration=getVidFramesAndFpsDur(vid)
        fps=int(fps)
        sampledFrames=[]
        sampledFramesGt=[]
        nSamples=int(fps*sampleRate)
        for i in range(int(duration)):
            smpl=np.sort(np.random.choice(np.arange(fps),nSamples,replace=False))
            for idx in smpl:
                sampledFrames.append(frmAll[i*fps:(i+1)*fps][idx])
                sampledFramesGt.append(meanGt[i*fps:(i+1)*fps][idx])
        featuresArrMain=[]
        featuresArrMainAvgFrames=[]
        for f in range(len(sampledFrames)):                       
            frmRz=cv2.resize(sampledFrames[f], (224,224))
            frmRzPyTensor=torch.Tensor(frmRz.reshape([1, 3, 224, 224]))
            features = model.extract_features(frmRzPyTensor)
            featuresArr=features.detach().numpy()
            featuresArr=featuresArr.reshape((1280,7,7))
            featuresArrMain.append(featuresArr)
            featuresArrMainAvgFrames.append(frmRz)
            print("visual feature for frame: ",f," in: ",vidName)
        featuresArrMain=np.array(featuresArrMain)
        sampledFramesGt=np.array(sampledFramesGt)
        featuresArrMainAvgFrames=np.array(featuresArrMainAvgFrames)
        pickle.dump(featuresArrMain,open(baseOut+vidName+"_"+str(nSamples)+"_featuresArrMain.p","wb"))
        pickle.dump(sampledFramesGt,open(baseOut+vidName+"_"+str(nSamples)+"_sampledFramesGt.p","wb"))
        pickle.dump(featuresArrMainAvgFrames,open(baseOut+vidName+"_"+str(nSamples)+"_featuresArrMainAvgFrames.p","wb"))
        print("features and gt written for: ",vidName)
        print("frames and gt written for: ",vidName)



def writeVisualFeaturesAlongGtTvSumForAllExtractors(basedir,groundTruth,baseOutEff,baseOutVgg,baseOutIncepV3,modelEff,modelVgg,modelInceptionV3,nSamples=2,strt=0):
    gtDf = pd.read_csv(groundTruth, sep="\t",header=None,index_col=0)
    allVideos=glob.glob(basedir)
    for v in range(strt,len(allVideos)):
        vid=allVideos[v]
        vidName=vid.split("/")[-1].split(".")[0]
        meanGt=getAvgGt(gtDf,vidName)
        frmAll,fps,duration=getVidFramesAndFpsDur(vid)
        sampledFrames=[]
        sampledFramesGt=[]
        fps=int(fps)
        for i in range(int(duration)):
            smpl=np.sort(np.random.choice(np.arange(fps),nSamples,replace=False))
            for idx in smpl:
                sampledFrames.append(frmAll[i*fps:(i+1)*fps][idx-1])
                sampledFramesGt.append(meanGt[i*fps:(i+1)*fps][idx-1])
        featuresArrMainVGG=[]
        featuresArrMainIncepV3=[]
        featuresArrMainAvgFrames299=[]
        featuresArrMainAvgFrames224=[]
        featuresArrMainEff=[]
        for f in range(len(sampledFrames)):                       
            frmRz224=cv2.resize(sampledFrames[f], (224,224))
            frmRz299=cv2.resize(sampledFrames[f], (299,299))
            frmRzPyTensor=torch.Tensor(frmRz224.reshape([1, 3, 224, 224]))
            featuresEff = modelEff.extract_features(frmRzPyTensor)
            featuresArrEff=featuresEff.detach().numpy()
            featuresArrEff=featuresArrEff.reshape((1280,7,7))
            featuresArrMainEff.append(featuresArrEff)
            frmRz299Org=frmRz299
            frmRz224Org=frmRz224
            frmRz299=np.expand_dims(frmRz299, axis=0)
            frmRz224=np.expand_dims(frmRz224, axis=0)
            frmRz224 = preprocess_input(frmRz224)
            vgg16_feature = modelVgg.predict(frmRz224)
            inceptionv3_feature = modelInceptionV3.predict(frmRz299)
            featuresArrVgg=vgg16_feature
            featuresArrIncepV3=inceptionv3_feature
            featuresArrMainVGG.append(featuresArrVgg)
            featuresArrMainIncepV3.append(featuresArrIncepV3)
            featuresArrMainAvgFrames299.append(frmRz299Org)
            featuresArrMainAvgFrames224.append(frmRz224Org)
        featuresArrMainVGG=np.array(featuresArrMainVGG)
        featuresArrMainIncepV3=np.array(featuresArrMainIncepV3)
        featuresArrMainEff=np.array(featuresArrMainEff)
        featuresArrMainAvgFrames299=np.array(featuresArrMainAvgFrames299)
        featuresArrMainAvgFrames224=np.array(featuresArrMainAvgFrames224)
        sampledFramesGt=np.array(sampledFramesGt)
        
        pickle.dump(featuresArrMainEff,open(baseOutEff[0]+vidName+"_"+baseOutEff[1]+".p","wb"))
        pickle.dump(featuresArrMainVGG,open(baseOutVgg[0]+vidName+"_"+baseOutVgg[1]+".p","wb"))
        pickle.dump(featuresArrMainIncepV3,open(baseOutIncepV3[0]+vidName+"_"+baseOutIncepV3[1]+".p","wb"))
        pickle.dump(featuresArrMainAvgFrames224,open(baseOutVgg[0]+vidName+"_"+baseOutVgg[1]+"Frames.p","wb"))
        pickle.dump(featuresArrMainAvgFrames299,open(baseOutIncepV3[0]+vidName+"_"+baseOutIncepV3[1]+"Frames.p","wb"))
        pickle.dump(sampledFramesGt,open(baseOutEff[0]+vidName+"_"+"smpGt"+".p","wb"))
        pickle.dump(sampledFramesGt,open(baseOutVgg[0]+vidName+"_"+"smpGt"+".p","wb"))
        pickle.dump(sampledFramesGt,open(baseOutIncepV3[0]+vidName+"_"+"smpGt"+".p","wb"))
        print("features and gt written for: ",vidName," vid:",v)
        print("frames and gt written for: ",vidName," vid:",v)
        del featuresArrMainVGG
        del featuresArrMainIncepV3
        del featuresArrMainEff
        del featuresArrMainAvgFrames224
        del featuresArrMainAvgFrames299
        

basedir="./../data/tvSum50/video/*mp4"
baseOutEff=["./../data/efficientNetB0Featues/","efficientNetB0Featues"]
baseOutVgg=["./../data/vgg16Featues/","vgg16Featues"]
baseOutIncepV3=["./../data/InceptionV3Featues/","InceptionV3Featues"]
groundTruth="./../data/tvSum50/data/ydata-tvsum50-anno.tsv"




modelVgg = VGG16(weights='imagenet', include_top=False)
#model.summary()
modelInceptionV3=InceptionV3(weights='imagenet', include_top=False)

modelEff= EfficientNet.from_pretrained('efficientnet-b0')
#model2.summary()


strt=0
writeVisualFeaturesAlongGtTvSumForAllExtractors(basedir,groundTruth,baseOutEff,baseOutVgg,baseOutIncepV3,modelEff,modelVgg,modelInceptionV3,nSamples=2,strt=strt)

